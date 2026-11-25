"""Download the public paper assets and verify published SHA256 values."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "https://github.com/BunnyAssassins/GSF-YOLO/releases/download/v1.0-paper/"


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def records(kind):
    if kind == "weights":
        with (ROOT / "manifests/checkpoints.csv").open(encoding="utf-8-sig", newline="") as f:
            return [{"model": r["configuration"], "seed": int(r["seed"]), "path": r["file"],
                     "download_url": r["download_url"], "sha256": r["published_checkpoint_sha256"]}
                    for r in csv.DictReader(f)]
    return json.loads((ROOT / "manifests/predictions.json").read_text(encoding="utf-8"))["files"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("weights", "predictions"))
    parser.add_argument("--models", nargs="+", help="Filter by descriptive model slug")
    parser.add_argument("--seeds", nargs="+", type=int, help="Filter by training seed")
    parser.add_argument("--split", choices=("validation", "test-dev", "testdev"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    selected = [r for r in records(args.kind)
                if (not args.models or r["model"] in args.models)
                and (not args.seeds or int(r["seed"]) in args.seeds)
                and (not args.split or r.get("split") == ("test-dev" if args.split == "testdev" else args.split))]
    if not selected:
        raise ValueError("No assets match the requested filters")
    for row in selected:
        destination = (ROOT / row["path"]).resolve()
        destination.relative_to(ROOT)
        if not row["download_url"].startswith(PREFIX):
            raise ValueError("Unexpected asset origin in manifest")
        if args.dry_run:
            print(row["path"])
            continue
        if destination.exists():
            if digest(destination) != row["sha256"]:
                raise ValueError(f"Existing asset hash mismatch: {row['path']}")
            print("Verified", row["path"])
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".download")
        with urlopen(row["download_url"], timeout=120) as response, temporary.open("xb") as out:
            for block in iter(lambda: response.read(1024 * 1024), b""):
                out.write(block)
        if digest(temporary) != row["sha256"]:
            raise ValueError(f"Downloaded asset hash mismatch: {row['path']}")
        temporary.replace(destination)
        print("Downloaded and verified", row["path"])


if __name__ == "__main__":
    main()
