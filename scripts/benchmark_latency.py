"""Balanced, synchronized decoded-image latency benchmark with external telemetry."""
from __future__ import annotations

import argparse
import csv
import gc
import os
import statistics
import subprocess
import time
from pathlib import Path

from _common import PREDICT_SETTINGS, configure_precision, digest, dump, init_runtime, resolve, runtime_environment


def gpu_pids():
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, check=True)
    return [int(item.strip()) for item in result.stdout.splitlines() if item.strip().isdigit()]


def guard(ambient):
    competing = set(gpu_pids()) - {os.getpid()} - set(ambient)
    if competing:
        raise RuntimeError(f"Competing GPU compute processes: {sorted(competing)}")


def telemetry():
    import psutil
    fields = ("name", "driver_version", "memory.used", "utilization.gpu", "temperature.gpu",
              "power.draw", "clocks.current.graphics", "clocks.current.memory", "pstate")
    result = subprocess.run(["nvidia-smi", "--query-gpu=" + ",".join(fields), "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, check=True)
    values = next(csv.reader(result.stdout.splitlines()))
    return dict(time=time.time(), gpu={key: value.strip() for key, value in zip(fields, values)},
                cpu_percent=psutil.cpu_percent(interval=None), ram_percent=psutil.virtual_memory().percent,
                compute_pids=gpu_pids())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, nargs="+", required=True, help="Pass all compared checkpoints together")
    parser.add_argument("--framework", choices=("auto", "ultralytics", "rtmdet", "rtdetrv2"), default="auto",
                        help="Auto identifies the framework from published checkpoint directories")
    parser.add_argument("--images", type=Path, required=True, help="Validation image directory, sorted by filename")
    parser.add_argument("--device", default="0")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--expected-images", type=int, default=548)
    parser.add_argument("--allow-ambient-pid", type=int, action="append", default=[], help="Explicitly recorded idle desktop contexts")
    parser.add_argument("--output", type=Path, default=Path("runs/balanced_latency"))
    args = parser.parse_args()
    if args.rounds < 1 or args.warmup < 1:
        parser.error("Rounds and warm-up count must be positive")
    init_runtime()
    configure_precision()
    import cv2
    import torch
    from external_predictors import framework_for, load_predictor
    if not torch.cuda.is_available() or args.device == "cpu":
        raise RuntimeError("This benchmark requires CUDA; CPU timings are a different protocol")
    device = torch.device("cuda:" + str(args.device))
    output = resolve(args.output)
    if output.exists():
        raise FileExistsError("Use a new --output directory")
    files = sorted(path for path in resolve(args.images).iterdir() if path.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if len(files) != args.expected_images:
        raise ValueError(f"Expected {args.expected_images} images, found {len(files)}")
    weights = [resolve(path) for path in args.weights]
    if len(set(weights)) != len(weights):
        raise ValueError("Duplicate checkpoint paths")
    identities = [dict(checkpoint=path.name, model=path.parent.name, sha256=digest(path),
                       framework=framework_for(path, args.framework)) for path in weights]
    guard(args.allow_ambient_pid)
    before = telemetry()
    if float(before["gpu"]["utilization.gpu"]) > 5:
        raise RuntimeError("GPU utilization exceeds the idle threshold (5%)")
    orders = []
    for repeat in range(args.rounds):
        order = list(range(len(weights)))
        if repeat % 3 == 1:
            order.reverse()
        elif repeat % 3 == 2:
            offset = len(order) // 2
            order = order[offset:] + order[:offset]
        orders.append(order)
    dump(output / "plan.json", dict(checkpoints=identities, orders=orders, images=[path.name for path in files],
        images_sha256=[digest(path) for path in files], rounds=args.rounds, warmup=args.warmup,
        ambient_pids=args.allow_ambient_pid, settings=dict(PREDICT_SETTINGS, device=args.device)))
    measurements = [[] for _ in weights]
    settings = dict(PREDICT_SETTINGS, device=args.device)
    for repeat, order in enumerate(orders):
        for index in order:
            guard(args.allow_ambient_pid)
            predictor = load_predictor(weights[index], args.framework, args.device)
            warm = cv2.imread(str(files[0]))
            snapshots, records = [telemetry()], []
            with torch.inference_mode():
                for _ in range(args.warmup):
                    predictor(warm)
                torch.cuda.synchronize(device)
                for image_id, path in enumerate(files, 1):
                    if image_id % 100 == 1:
                        guard(args.allow_ambient_pid)
                        snapshots.append(telemetry())
                    image = cv2.imread(str(path))
                    if image is None:
                        raise ValueError(f"Cannot decode {path.name}")
                    torch.cuda.synchronize(device)
                    start = time.perf_counter_ns()
                    result = predictor(image)
                    torch.cuda.synchronize(device)
                    elapsed = (time.perf_counter_ns() - start) / 1e6
                    record = dict(image_id=image_id, file=path.name, pipeline_ms=elapsed)
                    if hasattr(result, "speed"):
                        record["native_speed_ms"] = {key: float(value) for key,value in result.speed.items()}
                    records.append(record)
            guard(args.allow_ambient_pid)
            snapshots.append(telemetry())
            values = [record["pipeline_ms"] for record in records]
            mean_ms = statistics.mean(values)
            measurements[index].append(mean_ms)
            dump(output / f"checkpoint{index}_round{repeat}.json", dict(identity=identities[index], round=repeat,
                measurements=records, telemetry=snapshots, mean_ms=mean_ms,
                environment=runtime_environment(), settings=predictor.settings, native_end2end=predictor.native_end2end))
            print(f"Round {repeat+1}: {weights[index].parent.name}/{weights[index].name} {mean_ms:.4f} ms", flush=True)
            del predictor, result
            gc.collect()
            torch.cuda.empty_cache()
            if digest(weights[index]) != identities[index]["sha256"]:
                raise RuntimeError("Checkpoint changed during measurement")
    checkpoint_means = [statistics.mean(values) for values in measurements]
    groups = {}
    for identity, mean_ms in zip(identities, checkpoint_means):
        groups.setdefault(identity["model"], []).append(mean_ms)
    summary = {model: dict(checkpoint_count=len(values), mean_ms=statistics.mean(values),
                          sample_sd_across_checkpoints_ms=statistics.stdev(values) if len(values)>1 else None,
                          fps=1000/statistics.mean(values)) for model, values in groups.items()}
    dump(output / "summary.json", dict(status="COMPLETE", technical_rounds=args.rounds,
        checkpoint_round_means_ms=measurements, checkpoint_means_ms=checkpoint_means, groups=summary,
        timing_scope="Decoded-image preprocessing, transfer, native-head inference and postprocessing; excludes decode, disk I/O, telemetry, drawing, and serialization",
        aggregation="Equal-weight mean over technical rounds per checkpoint, then mean and sample SD over training checkpoints; technical rounds are not training seeds"))


if __name__ == "__main__":
    main()
