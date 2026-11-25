"""Reconstruct selected-epoch native AP means and sample SDs from released CSV logs."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

from _common import digest, dump, resolve


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs", type=Path, default=Path("results/training_logs"))
    parser.add_argument("--output", type=Path, default=Path("outputs/native_training_summary.json"))
    args = parser.parse_args()
    output = resolve(args.output)
    if output.exists():
        raise FileExistsError("Choose a new --output path")
    records, groups = [], {}
    for selection_path in sorted(resolve(args.logs).glob("*/seed*/selection.json")):
        source = selection_path.parent / "results.csv"
        if not source.is_file():
            continue  # external-framework JSONL logs have their own parser
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        epoch = int(selection["selected_epoch_one_based"])
        with source.open(encoding="utf-8-sig", newline="") as stream:
            rows = [{key.strip(): value for key, value in row.items()} for row in csv.DictReader(stream)]
        matches = [row for row in rows if int(row["epoch"]) == epoch]
        if len(matches) != 1:
            raise ValueError(f"Selected epoch not uniquely present in {selection_path.parent.name}")
        selected = matches[0]
        values = dict(ap50=100*float(selected["metrics/mAP50(B)"]),
                      ap50_95=100*float(selected["metrics/mAP50-95(B)"]))
        max_ap = max(float(row["metrics/mAP50-95(B)"]) for row in rows)
        if abs(float(selected["metrics/mAP50-95(B)"]) - max_ap) > 1e-5:
            raise ValueError("Selected epoch is not the maximum native validation AP50:95")
        record = dict(configuration=selection["configuration"], seed=selection["seed"],
                      selected_epoch_one_based=epoch, metrics_percent=values,
                      results_csv_sha256=digest(source), selection_sha256=digest(selection_path))
        records.append(record)
        groups.setdefault(record["configuration"], []).append(record)
    summaries = {}
    for name, items in groups.items():
        summaries[name] = dict(training_seed_count=len(items),
            metrics_percent={key: dict(mean=statistics.mean(item["metrics_percent"][key] for item in items),
                  sample_sd=statistics.stdev(item["metrics_percent"][key] for item in items) if len(items)>1 else None)
                  for key in ("ap50", "ap50_95")})
    dump(output, dict(protocol="native_checkpoint_selected_training_logs", records=records, summary=summaries,
         note="These logged scores are distinct from newly evaluated COCO300 scores. SD is undefined for one training seed."))
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
