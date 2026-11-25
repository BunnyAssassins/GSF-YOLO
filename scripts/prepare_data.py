"""Convert original VisDrone DET annotations to the paper's corrected YOLO labels.

This script does not download or redistribute VisDrone. It creates a new dataset
directory and records every original image/annotation hash and conversion count.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAMES = ["pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle",
         "awning-tricycle", "bus", "motor"]
EXPECTED = {"train": 6471, "val": 548, "test-dev": 1610}


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def convert_annotation(path: Path, width: int, height: int):
    rows, counts = [], {"input": 0, "excluded_category_or_score": 0,
                       "nonpositive_box": 0, "rounded_nonpositive_box": 0,
                       "clipped": 0, "retained": 0}
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split(",")
        if len(fields) != 8:
            raise ValueError(f"{path.name}:{number}: expected eight annotation columns")
        values = [float(value) for value in fields]
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"{path.name}:{number}: nonfinite annotation value")
        x, y, bw, bh, score, category, _, _ = values
        if category != int(category):
            raise ValueError(f"{path.name}:{number}: category must be an integer")
        counts["input"] += 1
        if score == 0 or not 1 <= category <= 10:
            counts["excluded_category_or_score"] += 1
            continue
        x1, y1 = max(0.0, min(float(width), x)), max(0.0, min(float(height), y))
        x2, y2 = max(0.0, min(float(width), x + bw)), max(0.0, min(float(height), y + bh))
        if (x1, y1, x2, y2) != (x, y, x + bw, y + bh):
            counts["clipped"] += 1
        if x2 <= x1 or y2 <= y1:
            counts["nonpositive_box"] += 1
            continue
        coordinates = [(x1 + x2) / (2 * width), (y1 + y2) / (2 * height),
                       (x2 - x1) / width, (y2 - y1) / height]
        formatted = [f"{value:.6f}" for value in coordinates]
        if float(formatted[2]) <= 0 or float(formatted[3]) <= 0:
            counts["rounded_nonpositive_box"] += 1
            continue
        rows.append(f"{int(category) - 1} " + " ".join(formatted))
        counts["retained"] += 1
    return "\n".join(rows) + ("\n" if rows else ""), counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True,
                        help="Parent of VisDrone2019-DET-{train,val,test-dev} directories")
    parser.add_argument("--output", type=Path, default=Path("datasets/VisDrone2019-DET-corrected"))
    parser.add_argument("--splits", nargs="+", choices=tuple(EXPECTED), default=list(EXPECTED))
    parser.add_argument("--images-mode", choices=("copy", "hardlink"), default="copy")
    parser.add_argument("--require-paper-counts", action="store_true",
                        help="Reject image counts different from the paper's 6471/548/1610")
    args = parser.parse_args()
    from PIL import Image
    import yaml
    raw_root, output = resolve(args.raw_root), resolve(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Output must be a new directory or an empty directory")
    sources = {}
    for split in args.splits:
        source = raw_root / f"VisDrone2019-DET-{split}"
        images = sorted((source / "images").glob("*.jpg"))
        if not images:
            raise FileNotFoundError(f"No JPG images for split {split}")
        if args.require_paper_counts and len(images) != EXPECTED[split]:
            raise ValueError(f"{split}: {len(images)} images; paper requires {EXPECTED[split]}")
        for image in images:
            annotation = source / "annotations" / (image.stem + ".txt")
            if not annotation.is_file():
                raise FileNotFoundError(f"Missing annotation: {annotation.name}")
        sources[split] = (source, images)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"protocol": "visdrone_corrected_yolo_six_decimals_v1",
                "images_mode": args.images_mode, "class_names": NAMES,
                "label_decimal_places": 6, "splits": {}, "complete": False}
    for split, (source, images) in sources.items():
        image_output, label_output = output / split / "images", output / split / "labels"
        image_output.mkdir(parents=True)
        label_output.mkdir(parents=True)
        records, totals = [], {}
        for image in images:
            annotation = source / "annotations" / (image.stem + ".txt")
            with Image.open(image) as frame:
                width, height = frame.size
            content, counts = convert_annotation(annotation, width, height)
            destination = image_output / image.name
            if args.images_mode == "hardlink":
                os.link(image, destination)
            else:
                shutil.copy2(image, destination)
            label = label_output / (image.stem + ".txt")
            label.write_text(content, encoding="utf-8", newline="\n")
            for key, value in counts.items():
                totals[key] = totals.get(key, 0) + value
            records.append({"image": image.name, "width": width, "height": height,
                            "image_sha256": digest(image), "annotation_sha256": digest(annotation),
                            "label_sha256": digest(label), "counts": counts})
        manifest["splits"][split] = {"images": len(images), "paper_expected_images": EXPECTED[split],
                                     "counts": totals, "files": records}
        (output / f"{split}_images.txt").write_text(
            "".join(f"./{split}/images/{image.name}\n" for image in images), encoding="utf-8")
    config = {"path": output.as_posix(), "names": NAMES, "nc": len(NAMES)}
    for split in args.splits:
        config["test" if split == "test-dev" else split] = f"{split}/images"
    (output / "data.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    manifest["complete"] = True
    (output / "conversion_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "COMPLETE", "output": str(output),
                      "split_counts": {key: value["images"] for key, value in manifest["splits"].items()}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
