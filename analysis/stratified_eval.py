"""Recompute three-seed size/category/density COCO300 strata, or aggregate archived metrics."""
from pathlib import Path
import argparse
from collections import Counter
import copy
import gzip
import json
import statistics
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def load(path):
    opener=gzip.open if Path(path).suffix==".gz" else open
    with opener(path,"rt",encoding="utf-8-sig") as f:return json.load(f)


def ap(ev, area=0, category=None):
    values=ev.eval["precision"][:,:,:,area,ev.params.maxDets.index(300)]
    if category is not None:values=values[:,:,ev.params.catIds.index(category)]
    values=values[values>-1]
    return float(values.mean()) if values.size else None


def subset_accumulation(ev, ids):
    original=ev._paramsEval
    wanted=set(ids)
    indexes=[i for i,image_id in enumerate(original.imgIds) if image_id in wanted]
    assert len(indexes)==len(wanted)
    sub=copy.copy(ev)
    sub.params=copy.deepcopy(original)
    sub.params.imgIds=[original.imgIds[i] for i in indexes]
    sub._paramsEval=copy.deepcopy(sub.params)
    matches=np.asarray(ev.evalImgs,dtype=object).reshape(len(original.catIds),len(original.areaRng),len(original.imgIds))
    sub.evalImgs=matches[:,:,indexes].reshape(-1).tolist()
    sub.eval={}
    sub.accumulate()
    return sub


def aggregate_saved(source):
    output=[]
    for row in source["rows"]:
        record={k:v for k,v in row.items() if k not in ("baseline","gsf_yolo","paired_delta")}
        for model in ("baseline","gsf_yolo","paired_delta"):
            values=row[model]["seeds"]
            mean,sd=statistics.mean(values),statistics.stdev(values)
            assert abs(mean-row[model]["mean"])<1e-12
            assert abs(sd-row[model]["sample_sd"])<1e-12
            record[model]={"seeds":values,"mean":mean,"sample_sd":sd}
        output.append(record)
    return output


def from_predictions(args):
    import csv
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    gt=load(args.ground_truth)
    with args.training_density.open(encoding="utf-8-sig",newline="") as f:training=list(csv.DictReader(f))
    densities=[int(r["n_gt"])*1e6/(int(r["width"])*int(r["height"])) for r in training]
    cutpoints=np.quantile(densities,[1/3,2/3],method="linear").tolist()
    counts=Counter(a["image_id"] for a in gt["annotations"])
    groups={k:[] for k in ("low","medium","high")}
    for im in gt["images"]:
        rho=counts[im["id"]]*1e6/(im["width"]*im["height"])
        groups["low" if rho<=cutpoints[0] else "medium" if rho<=cutpoints[1] else "high"].append(im["id"])
    coco=COCO(str(args.ground_truth));records=[]
    for model in ("p2_baseline","gsf_yolo"):
        for seed in range(3):
            path=args.prediction_dir/f"val_{model}_seed{seed}.json.gz"
            predictions=load(path)
            ev=COCOeval(coco,coco.loadRes(predictions),"bbox")
            ev.params.maxDets=[1,10,300];ev.evaluate();ev.accumulate()
            metrics={"overall":100*ap(ev),"size_small":100*ap(ev,1),"size_medium":100*ap(ev,2),"size_large":100*ap(ev,3)}
            for category in ev.params.catIds:metrics[f"class_{category}"]=100*ap(ev,category=category)
            for group,ids in groups.items():
                sub=subset_accumulation(ev,ids)
                metrics[f"density_{group}_AP"]=100*ap(sub)
                metrics[f"density_{group}_AP_S"]=100*ap(sub,1)
            records.append(dict(model=model,seed=seed,metrics_percent=metrics))
            print(model,seed,metrics["overall"],flush=True)
    archived=load(ROOT/"results/summary/gsf_stratified_statistics.json")
    output=[]
    for source_row in archived["rows"]:
        row={k:v for k,v in source_row.items() if k not in ("baseline","gsf_yolo","paired_delta")}
        a=[next(r for r in records if r["model"]=="p2_baseline" and r["seed"]==s)["metrics_percent"][row["key"]] for s in range(3)]
        b=[next(r for r in records if r["model"]=="gsf_yolo" and r["seed"]==s)["metrics_percent"][row["key"]] for s in range(3)]
        for model,values in [("baseline",a),("gsf_yolo",b),("paired_delta",[y-x for x,y in zip(a,b)])]:
            row[model]=dict(seeds=values,mean=statistics.mean(values),sample_sd=statistics.stdev(values))
        output.append(row)
    return dict(rows=output,records=records,density_cutpoints=cutpoints,maxDets=[1,10,300],official_ignored_region_filtering=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-saved",action="store_true",help="CPU-only reaggregation of released per-seed metrics")
    parser.add_argument("--ground-truth",type=Path)
    parser.add_argument("--training-density",type=Path,default=ROOT/"results/stratified/training_density.csv")
    parser.add_argument("--prediction-dir",type=Path,default=ROOT/"assets/predictions")
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.from_saved:
        result={"rows":aggregate_saved(load(ROOT/"results/summary/gsf_stratified_statistics.json")),"reaggregation_only":True}
    else:
        if not args.ground_truth:parser.error("--ground-truth required for prediction-based evaluation")
        result=from_predictions(args)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False),encoding="utf-8")
    print("PASS",len(result["rows"]),"strata")


if __name__=="__main__":main()
