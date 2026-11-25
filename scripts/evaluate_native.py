"""Run native Ultralytics validation; this is distinct from COCO300 evaluation."""
from __future__ import annotations

import argparse
from pathlib import Path

from _common import configure_precision, digest, dump, init_runtime, resolve, runtime_environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=Path("weights/gsf_yolo/seed0.pt"))
    parser.add_argument("--data", type=Path, default=Path("datasets/VisDrone2019-DET-corrected/data.yaml"))
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=6)
    parser.add_argument("--output", type=Path, default=Path("runs/native_validation"))
    args = parser.parse_args()
    init_runtime()
    configure_precision()
    import torch
    from ultralytics import YOLO
    weights, output = resolve(args.weights), resolve(args.output)
    if output.exists():
        raise FileExistsError("Use a new --output directory for each evaluation")
    before_hash = digest(weights)
    model = YOLO(str(weights))
    model.model.eval()
    settings = dict(data=str(resolve(args.data)), split=args.split, imgsz=832,
                    batch=args.batch, device=args.device, conf=0.001, iou=0.7,
                    max_det=300, rect=False, half=False, augment=False,
                    plots=False, save_json=False, verbose=False,
                    project=str(output.parent), name=output.name, exist_ok=False)
    with torch.inference_mode():
        metrics = model.val(**settings)
    if digest(weights) != before_hash:
        raise RuntimeError("Checkpoint file changed during evaluation")
    dump(output / "summary.json", dict(protocol="native_ultralytics_validation",
        metrics={key: float(value) for key, value in metrics.results_dict.items()
                 if key in {"metrics/mAP50(B)", "metrics/mAP50-95(B)"}},
        weights_sha256=before_hash, settings=settings, environment=runtime_environment(),
        note="A new validation run; manuscript native AP values are the original selected training epoch logs."))


if __name__ == "__main__":
    main()
