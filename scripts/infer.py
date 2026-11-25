"""Run native-head FP32 inference and save COCO detections in original coordinates."""
from __future__ import annotations

import argparse
from pathlib import Path

from _common import PREDICT_SETTINGS, configure_precision, digest, dump, init_runtime, resolve, runtime_environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=Path("weights/gsf_yolo/seed0.pt"))
    parser.add_argument("--source", type=Path, required=True, help="Image file or directory")
    parser.add_argument("--device", default="0")
    parser.add_argument("--output", type=Path, default=Path("runs/inference"))
    args = parser.parse_args()
    init_runtime()
    configure_precision()
    import cv2
    import torch
    from ultralytics import YOLO
    source, output, weights = resolve(args.source), resolve(args.output), resolve(args.weights)
    if output.exists():
        raise FileExistsError("Use a new --output directory")
    files = [source] if source.is_file() else sorted(path for path in source.iterdir()
             if path.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"))
    if not files:
        raise ValueError("No images found")
    before_hash = digest(weights)
    model = YOLO(str(weights))
    model.model.eval()
    predictions, images = [], []
    settings = dict(PREDICT_SETTINGS, device=args.device)
    with torch.inference_mode():
        for image_id, path in enumerate(files, 1):
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError(f"Cannot decode {path.name}")
            result = model.predict(image, **settings)[0]
            images.append(dict(id=image_id, file_name=path.name, width=image.shape[1], height=image.shape[0]))
            for box, score, cls in zip(result.boxes.xyxy.cpu().tolist(),
                                       result.boxes.conf.cpu().tolist(), result.boxes.cls.cpu().tolist()):
                x1, y1, x2, y2 = box
                predictions.append(dict(image_id=image_id, category_id=int(cls),
                    bbox=[x1, y1, x2-x1, y2-y1], score=score))
    dump(output / "predictions.json", predictions)
    dump(output / "image_map.json", dict(images=images))
    dump(output / "summary.json", dict(weights_sha256=before_hash,
         native_end2end=bool(getattr(model.model, "end2end", False)), settings=settings,
         images=len(images), detections=len(predictions), environment=runtime_environment(),
         predictions_sha256=digest(output / "predictions.json")))
    if digest(weights) != before_hash:
        raise RuntimeError("Checkpoint changed during inference")


if __name__ == "__main__":
    main()
