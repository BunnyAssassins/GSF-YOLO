"""Launch one disclosed native external recipe; preview unless explicitly requested."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from _common import ROOT, init_runtime, read_yaml, resolve
from evaluate_coco import image_paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--framework", choices=("rtmdet", "rtdetrv2"), required=True)
    parser.add_argument("--data", type=Path, default=Path("datasets/VisDrone2019-DET-corrected/data.yaml"))
    parser.add_argument("--annotations", type=Path, default=Path("datasets/VisDrone2019-DET-coco"))
    parser.add_argument("--project", type=Path, default=Path("runs/native_external_reproduction"))
    parser.add_argument("--device", default="0")
    parser.add_argument("--start-training", action="store_true")
    args = parser.parse_args()
    source = resolve("configs/external/rtmdet_tiny.py" if args.framework == "rtmdet" else "configs/external/rtdetrv2_r18.yml")
    work = resolve(args.project) / ("rtmdet_tiny__seed0" if args.framework == "rtmdet" else "rtdetrv2_r18__seed0")
    tool = resolve("external/mmdetection/tools/train.py" if args.framework == "rtmdet" else "external/rtdetrv2/tools/train.py")
    print(json.dumps(dict(framework=args.framework, config=str(source), seed=0, epochs=100,
                         data=str(resolve(args.data)), work_dir=str(work), start_training=args.start_training), indent=2))
    if not args.start_training:
        print("Preview complete. Add --start-training only when ready to run the native recipe.")
        return
    if work.exists():
        raise FileExistsError("Choose a new --project; published inference checkpoints cannot resume optimization")
    for split in ("train", "val"):
        if not (resolve(args.annotations) / f"instances_{split}.json").is_file():
            raise FileNotFoundError("Run scripts/prepare_external_data.py first")
    from benchmark_latency import guard
    guard([])
    init_runtime()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.device, PYTHONIOENCODING="utf-8")
    if args.framework == "rtmdet":
        package = ROOT / "external/mmdetection"
        sys.path.insert(0, str(package))
        from mmengine.config import Config
        config = Config.fromfile(str(source))
        for split, loader_key in (("train", "train_dataloader"), ("val", "val_dataloader")):
            files = image_paths(args.data, split)
            if len({path.parent for path in files}) != 1:
                raise ValueError("Native recipe requires one image directory per split")
            dataset = config[loader_key].dataset
            dataset.ann_file = str(resolve(args.annotations) / f"instances_{split}.json")
            dataset.data_prefix.img = str(files[0].parent) + os.sep
        config.test_dataloader = config.val_dataloader
        config.val_evaluator.ann_file = str(resolve(args.annotations) / "instances_val.json")
        config.test_evaluator = config.val_evaluator
        generated = work / "resolved_config.py"
        generated.parent.mkdir(parents=True)
        config.dump(str(generated))
        command = [sys.executable, str(tool), str(generated), "--work-dir", str(work), "--launcher", "none"]
    else:
        package = ROOT / "external/rtdetrv2"
        sys.path.insert(0, str(package))
        from src.core import YAMLConfig
        import yaml
        config = YAMLConfig(str(source)).yaml_cfg
        for split, loader_key in (("train", "train_dataloader"), ("val", "val_dataloader")):
            files = image_paths(args.data, split)
            if len({path.parent for path in files}) != 1:
                raise ValueError("Native recipe requires one image directory per split")
            config[loader_key]["dataset"]["img_folder"] = str(files[0].parent) + os.sep
            config[loader_key]["dataset"]["ann_file"] = str(resolve(args.annotations) / f"instances_{split}.json")
        config.pop("__include__", None)
        generated = work / "resolved_config.yml"
        generated.parent.mkdir(parents=True)
        generated.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        command = [sys.executable, str(tool), "-c", str(generated), "--output-dir", str(work),
                   "--seed", "0", "--device", "cuda:0", "--use-amp"]
    env["PYTHONPATH"] = str(package) + os.pathsep + str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(command, cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
