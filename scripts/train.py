"""Train one published architecture with the recorded seed-specific arguments."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from _common import ROOT, dump, init_runtime, read_yaml, resolve

MODELS = ("p2_baseline", "dsf_refinement", "p2_p3_widening_without_dsf",
          "dsf_p2_widening", "dsf_p3_widening", "gsf_yolo")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, default="gsf_yolo")
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--config", type=Path, help="Override the recorded seed-specific training YAML")
    parser.add_argument("--data", type=Path, default=Path("datasets/VisDrone2019-DET-corrected/data.yaml"))
    parser.add_argument("--project", type=Path, default=Path("runs/reproduction"))
    parser.add_argument("--device", default="0")
    parser.add_argument("--resume", type=Path, help="Resume this user's own unstripped last.pt, retaining optimizer state")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved training arguments without loading a model")
    args = parser.parse_args()
    config = resolve(args.config or Path("configs/recorded_args") / args.model / f"seed{args.seed}.yaml")
    settings = read_yaml(config)
    # Original args are provided as documentary evidence. A fresh run starts
    # from the architecture; resumption needs the user's own optimizer state.
    for key in ("task", "mode", "model", "data", "project", "name", "save_dir", "resume", "device"):
        settings.pop(key, None)
    settings.update(data=str(resolve(args.data)), project=str(resolve(args.project)),
                    name=f"{args.model}__seed{args.seed}", seed=args.seed,
                    device=args.device, resume=bool(args.resume), exist_ok=False)
    architecture = resolve(Path("configs/models") / f"{args.model}.yaml")
    payload = dict(model=str(architecture), source_config=str(config), training=settings,
                   historical_resume_replayed=False if not args.resume else True)
    if args.dry_run:
        print(json.dumps(payload, indent=2))
        return
    if not resolve(args.data).is_file():
        raise FileNotFoundError("Prepare the dataset first with scripts/prepare_data.py")
    if not args.resume and (resolve(args.project) / settings["name"]).exists():
        raise FileExistsError("Run directory exists; choose a new --project or resume its last.pt")
    init_runtime()
    from ultralytics import YOLO
    model = YOLO(str(resolve(args.resume) if args.resume else architecture))
    if args.resume:
        # Ultralytics restores its checkpoint arguments and optimizer state.
        model.train(resume=True, data=settings["data"], device=args.device)
    else:
        model.train(**settings)
    run = Path(model.trainer.save_dir)
    dump(run / "release_invocation.json", payload)


if __name__ == "__main__":
    main()
