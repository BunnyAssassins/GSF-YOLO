"""Reproduce paired ROI gate/injection means and image-bootstrap intervals."""
from pathlib import Path
import argparse
import csv
import json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"reproduced/regional_injection.json")
    args=parser.parse_args()
    with (ROOT/"results/summary/gate_injection_image_pairs.csv").open(encoding="utf-8-sig",newline="") as f:
        records=list(csv.DictReader(f))
    archived=json.loads((ROOT/"results/summary/gate_injection_statistics.json").read_text())
    output=[]
    for row in archived["region_statistics"]:
        selected=[r for r in records if r["metric"]==row["metric"] and r["group"]==row["group"]]
        values=np.array([[float(r["seed0"]),float(r["seed1"])] for r in selected])
        assert len(values)==row["n_images"]
        means=values.mean(axis=0)
        joint=values.mean(axis=1)
        rng=np.random.default_rng(20261017)
        indices=rng.integers(len(values),size=(2000,len(values)))
        interval=np.percentile(joint[indices].mean(axis=1),[2.5,97.5])
        mean=float(values.mean());sd=float(means.std(ddof=1))
        assert abs(mean-row["mean"])<1e-12
        assert np.allclose(interval,row["ci95"],atol=1e-12,rtol=0)
        output.append(dict(metric=row["metric"],group=row["group"],seed0=float(means[0]),seed1=float(means[1]),
            mean=mean,sample_sd=sd,ci95=interval.tolist(),n_images=len(values),n_unique_pairs=row["n_unique_pairs"]))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(dict(region_statistics=output,resamples=2000,rng_seed=20261017,
        uncertainty="Image sampling conditional on fixed checkpoints; paired image draws across seeds"),indent=2),encoding="utf-8")
    print("PASS",len(output),"regional bootstrap rows")


if __name__=="__main__":main()
