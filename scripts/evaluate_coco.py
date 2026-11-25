"""Supplementary COCO-style bbox evaluation on corrected VisDrone YOLO GT.

Uses maxDets=[1,10,300], original-image areas and category IDs 0..9. No
official VisDrone ignored-region filtering is applied to predictions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def resolve(path: Path | str) -> Path:
    path = Path(path)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def image_paths(data_path: Path, split: str):
    import yaml
    cfg = yaml.safe_load(resolve(data_path).read_text(encoding="utf-8-sig"))
    root = resolve(cfg.get("path", "."))
    key = "test" if split == "test-dev" else split
    if key not in cfg:
        raise ValueError(f"Dataset YAML has no {key!r} split")
    values = cfg[key] if isinstance(cfg[key], list) else [cfg[key]]
    files = []
    for value in values:
        source = Path(value)
        source = source if source.is_absolute() else root / source
        if source.is_dir():
            files.extend(path.resolve() for path in source.rglob("*")
                         if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
        elif source.is_file() and source.suffix.lower() == ".txt":
            for line in source.read_text(encoding="utf-8-sig").splitlines():
                if line.strip():
                    image = Path(line.strip())
                    files.append(image.resolve() if image.is_absolute() else (source.parent / image).resolve())
        else:
            raise FileNotFoundError(f"Split source unavailable: {source}")
    files = sorted(set(files), key=lambda path: path.as_posix())
    if not files or any(not path.is_file() for path in files):
        raise ValueError("Split must contain existing image files")
    if len({path.name for path in files}) != len(files):
        raise ValueError("Duplicate image basenames prevent portable image mapping")
    return files


def build_ground_truth(data_path: Path, split: str):
    import yaml
    from PIL import Image
    cfg = yaml.safe_load(resolve(data_path).read_text(encoding="utf-8-sig"))
    names = cfg["names"]
    if isinstance(names, dict):
        names = [names[key] for key in sorted(names, key=int)]
    if len(names) != 10:
        raise ValueError("Paper protocol requires ten categories")
    files = image_paths(data_path, split)
    gt = {"info": {}, "licenses": [], "images": [], "annotations": [],
          "categories": [{"id": index, "name": name} for index, name in enumerate(names)]}
    hashes = []
    for image_id, image in enumerate(files, 1):
        with Image.open(image) as frame:
            width, height = frame.size
        # Match Ultralytics's final /images/ -> /labels/ transformation.
        components = list(image.parts)
        indices = [index for index, item in enumerate(components) if item == "images"]
        if not indices:
            raise ValueError(f"Image path has no images directory: {image.name}")
        components[indices[-1]] = "labels"
        label = Path(*components).with_suffix(".txt")
        if not label.is_file():
            raise FileNotFoundError(f"Corrected YOLO label unavailable: {label.name}")
        gt["images"].append({"id": image_id, "file_name": image.name, "width": width, "height": height})
        hashes.append({"image": image.name, "image_sha256": digest(image), "label_sha256": digest(label)})
        for number, line in enumerate(label.read_text(encoding="utf-8-sig").splitlines(), 1):
            if not line.strip():
                continue
            values = [float(value) for value in line.split()]
            if len(values) != 5 or not all(math.isfinite(value) for value in values):
                raise ValueError(f"{label.name}:{number}: invalid YOLO row")
            category, cx, cy, bw, bh = values
            if category != int(category) or not 0 <= category <= 9 or bw <= 0 or bh <= 0:
                raise ValueError(f"{label.name}:{number}: invalid category or dimensions")
            if any(value < 0 or value > 1 for value in (cx, cy, bw, bh)):
                raise ValueError(f"{label.name}:{number}: normalized coordinates outside [0,1]")
            box = [(cx - bw / 2) * width, (cy - bh / 2) * height, bw * width, bh * height]
            gt["annotations"].append({"id": len(gt["annotations"]) + 1, "image_id": image_id,
                                      "category_id": int(category), "bbox": box,
                                      "area": box[2] * box[3], "iscrowd": 0})
    return files, gt, hashes


def validate_predictions(predictions, gt):
    if not isinstance(predictions, list):
        raise ValueError("Predictions must be a COCO detection JSON list")
    image_ids = {image["id"] for image in gt["images"]}
    dropped = []
    valid = []
    for index, row in enumerate(predictions):
        if row.get("image_id") not in image_ids or row.get("category_id") not in range(10):
            raise ValueError(f"Prediction {index}: unmapped image/category ID")
        box, score = row.get("bbox"), row.get("score")
        if not isinstance(box, list) or len(box) != 4:
            raise ValueError(f"Prediction {index}: bbox must be xywh")
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in [*box, score]):
            raise ValueError(f"Prediction {index}: nonfinite coordinates/score")
        if not 0 <= score <= 1 or box[2] < 0 or box[3] < 0:
            raise ValueError(f"Prediction {index}: invalid score/dimensions")
        # Zero-area predictions are retained: saved paper predictions include
        # these and COCO can score them. Official TXT conversion handles them
        # separately with an explicitly audited drop-invalid-boxes option.
        valid.append({"image_id": int(row["image_id"]), "category_id": int(row["category_id"]),
                      "bbox": box, "score": score})
    return valid


def coco_metrics(gt, predictions):
    import numpy as np
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    coco = COCO()
    coco.dataset = gt
    coco.createIndex()
    if predictions:
        detections = coco.loadRes(predictions)
    else:
        detections = COCO()
        detections.dataset = {"images": gt["images"], "categories": gt["categories"], "annotations": []}
        detections.createIndex()
    evaluator = COCOeval(coco, detections, "bbox")
    evaluator.params.maxDets = [1, 10, 300]
    evaluator.evaluate()
    evaluator.accumulate()
    precision = evaluator.eval["precision"]
    # COCOeval.summarize() hardcodes maxDets=100 in its AP row. Index the
    # precision tensor directly to report the requested maxDets=300 AP.
    def ap(area: str, threshold=None):
        values = precision[:, :, :, evaluator.params.areaRngLbl.index(area), 2]
        if threshold is not None:
            values = values[np.isclose(evaluator.params.iouThrs, threshold)]
        valid = values[values > -1]
        return float(valid.mean()) if valid.size else None
    return {"ap": ap("all"), "ap50": ap("all", 0.5), "ap75": ap("all", 0.75),
            "ap_s": ap("small"), "ap_m": ap("medium"), "ap_l": ap("large")}, {
        "maxDets": [1, 10, 300], "iou_thresholds": evaluator.params.iouThrs.tolist(),
        "recall_thresholds": evaluator.params.recThrs.tolist(),
        "area_ranges": evaluator.params.areaRng,
        "category_ids": [category["id"] for category in gt["categories"]]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("datasets/VisDrone2019-DET-corrected/data.yaml"))
    parser.add_argument("--split", choices=("val", "test", "test-dev"), default="val")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--weights", type=Path, help="Run inference using this selected checkpoint")
    source.add_argument("--predictions", type=Path, help="CPU-only scoring of existing COCO predictions")
    parser.add_argument("--image-map", type=Path, help="Require stored image IDs/names/dimensions to match GT")
    parser.add_argument("--device", default="0")
    parser.add_argument("--output", type=Path, required=True, help="New directory for GT/predictions/metrics")
    parser.add_argument("--require-paper-count", action="store_true")
    args = parser.parse_args()
    files, gt, hashes = build_ground_truth(args.data, args.split)
    expected = {"train": 6471, "val": 548, "test": 1610, "test-dev": 1610}
    if args.require_paper_count and len(files) != expected[args.split]:
        raise ValueError(f"Split has {len(files)} images; expected {expected[args.split]}")
    if args.image_map:
        image_map = json.loads(resolve(args.image_map).read_text(encoding="utf-8-sig"))
        images = image_map["images"] if isinstance(image_map, dict) else image_map
        if images != gt["images"]:
            raise ValueError("Saved image map differs from reconstructed GT image map")
    output = resolve(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Output must be a new or empty directory")
    environment = {"python": sys.version, "platform": platform.platform()}
    settings = {"imgsz": 832, "batch": 1, "rect": False, "conf": 0.001, "iou": 0.7,
                "max_det": 300, "half": False, "augment": False, "agnostic_nms": False,
                "save": False, "verbose": False, "device": args.device}
    identity = {}
    if args.predictions:
        source_path = resolve(args.predictions)
        predictions = json.loads(source_path.read_text(encoding="utf-8-sig"))
        identity["predictions_sha256"] = digest(source_path)
        settings["device"] = "existing_predictions_cpu_scoring"
    else:
        os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics_settings"))
        os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
        sys.path.insert(0, str(ROOT))
        import cv2
        import torch
        import ultralytics
        from ultralytics import YOLO
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        cv2.setNumThreads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        weights = resolve(args.weights)
        before = digest(weights)
        model = YOLO(str(weights))
        identity.update({"weight_sha256": before, "native_end2end": bool(getattr(model.model, "end2end", False))})
        environment.update({"torch": torch.__version__, "cuda": torch.version.cuda,
                            "ultralytics": ultralytics.__version__, "opencv": cv2.__version__})
        predictions = []
        with torch.inference_mode():
            for image_id, path in enumerate(files, 1):
                image = cv2.imread(str(path))
                if image is None:
                    raise ValueError(f"Cannot decode image: {path.name}")
                result = model.predict(image, **settings)[0]
                for box, score, category in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist(),
                                                result.boxes.cls.cpu().tolist()):
                    x1, y1, x2, y2 = box
                    predictions.append({"image_id": image_id, "category_id": int(category),
                                        "bbox": [x1, y1, x2 - x1, y2 - y1], "score": score})
        if digest(weights) != before:
            raise RuntimeError("Checkpoint changed during evaluation")
    predictions = validate_predictions(predictions, gt)
    metrics, params = coco_metrics(gt, predictions)
    reported = ("ap", "ap_s") if args.split == "val" else ("ap", "ap50", "ap_s")
    metrics = {key: metrics[key] for key in reported}
    from importlib.metadata import version
    environment["pycocotools"] = version("pycocotools")
    dump(output / "ground_truth.json", gt)
    dump(output / "dataset_hashes.json", hashes)
    dump(output / "predictions.json", predictions)
    dump(output / "summary.json", {"status": "COMPLETE", "protocol": "corrected_visdrone_coco300",
                                   "split": args.split, "images": len(files), "gt_boxes": len(gt["annotations"]),
                                   "metrics": metrics, "metrics_percent": {key: 100 * value if value is not None else None
                                                                            for key, value in metrics.items()},
                                   "coco_params": params, "settings": settings, "identity": identity,
                                   "environment": environment, "script_sha256": digest(Path(__file__)),
                                   "official_visdrone_ignore_filtering": False,
                                   "timing_measured": False})
    print(json.dumps({"status": "COMPLETE", "metrics_percent": {key: 100 * value if value is not None else None
                                                                for key, value in metrics.items()}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
