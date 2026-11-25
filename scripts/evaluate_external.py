"""Evaluate released native RTMDet/RT-DETRv2 checkpoints with supplementary COCO300."""
from __future__ import annotations

import argparse
from pathlib import Path

from _common import configure_precision, digest, dump, init_runtime, resolve, runtime_environment
from evaluate_coco import build_ground_truth, coco_metrics, validate_predictions
from external_predictors import detection_records, load_predictor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--framework", choices=("auto", "rtmdet", "rtdetrv2", "ultralytics"), default="auto")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--data", type=Path, default=Path("datasets/VisDrone2019-DET-corrected/data.yaml"))
    parser.add_argument("--split", choices=("val",), default="val")
    parser.add_argument("--device", default="0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output, weights = resolve(args.output), resolve(args.weights)
    if output.exists():
        raise FileExistsError("Choose a new --output directory")
    init_runtime()
    configure_precision()
    import cv2
    import torch
    files, gt, hashes = build_ground_truth(args.data, args.split)
    before = digest(weights)
    predictor = load_predictor(weights, args.framework, args.device, args.config)
    predictions = []
    with torch.inference_mode():
        for image_id, path in enumerate(files, 1):
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError(f"Cannot decode {path.name}")
            predictions.extend(detection_records(predictor(image), predictor.framework, image_id))
    predictions = validate_predictions(predictions, gt)
    metrics, params = coco_metrics(gt, predictions)
    metrics = {"ap_s": metrics["ap_s"]}
    if digest(weights) != before:
        raise RuntimeError("Checkpoint changed during evaluation")
    dump(output / "ground_truth.json", gt)
    dump(output / "dataset_hashes.json", hashes)
    dump(output / "predictions.json", predictions)
    dump(output / "summary.json", dict(status="COMPLETE", framework=predictor.framework,
        protocol="corrected_visdrone_coco300_native_external_preprocessing", settings=predictor.settings,
        metrics=metrics, metrics_percent={key: 100*value if value is not None else None for key,value in metrics.items()},
        coco_params=params, weight_sha256=before, environment=runtime_environment(),
        official_visdrone_ignore_filtering=False, timing_measured=False))


if __name__ == "__main__":
    main()
