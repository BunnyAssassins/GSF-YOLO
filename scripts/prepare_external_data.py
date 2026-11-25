"""Build native-framework COCO views of corrected YOLO labels without modifying data."""
from __future__ import annotations

import argparse
from pathlib import Path

from _common import digest, dump, resolve
from evaluate_coco import build_ground_truth


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("datasets/VisDrone2019-DET-corrected/data.yaml"))
    parser.add_argument("--output", type=Path, default=Path("datasets/VisDrone2019-DET-coco"))
    args = parser.parse_args()
    output = resolve(args.output)
    if output.exists():
        raise FileExistsError("Choose a new --output directory")
    audit = {}
    for split in ("train", "val"):
        files, gt, hashes = build_ground_truth(args.data, split)
        destination = output / f"instances_{split}.json"
        dump(destination, gt)
        audit[split] = dict(images=len(files), gt_boxes=len(gt["annotations"]), annotation_sha256=digest(destination), source_hashes=hashes)
    dump(output / "conversion_audit.json", dict(splits=audit, category_ids=list(range(10)), original_data_modified=False))


if __name__ == "__main__":
    main()
