"""CPU-only aggregation of published per-run accuracy and timing evidence."""
from pathlib import Path
import argparse
import csv
import json
import statistics
ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"reproduced/tables")
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    data=json.loads((ROOT/"results/summary/table2_precision_cost_statistics.json").read_text(encoding="utf-8"))
    rows=[]
    for model in data["models"]:
        runs=sorted(model["runs"],key=lambda r:r["seed"])
        assert [r["seed"] for r in runs]==[0,1,2]
        row=dict(model=model["experiment"],seeds="0,1,2",parameters=model["parameters"],gflops=model["gflops"])
        for key,source in [("native_ap", "logged_ap_percent"),("native_ap50","logged_ap50_percent"),("coco_ap_s","ap_s_percent")]:
            values=[r[source] for r in runs]
            row[key]=statistics.mean(values);row[key+"_sample_sd"]=statistics.stdev(values)
        latencies=[statistics.mean(r["rounds_ms"]) for r in runs]
        assert abs(statistics.mean(latencies)-model["mean_latency_ms"])<1e-10
        row["latency_ms"]=statistics.mean(latencies);row["latency_sample_sd_ms"]=statistics.stdev(latencies)
        row["fps"]=1000/row["latency_ms"]
        rows.append(row)
    with (args.output/"ablation_unrounded.csv").open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    gate=json.loads((ROOT/"results/testdev/final_gate.json").read_text(encoding="utf-8"))
    gate_rows=[]
    for mode in ("native","valid_spatial_mean","valid_shuffle_0","valid_shuffle_l1_0"):
        rs=[gate["runs"][f"{mode}__seed{s}"] for s in range(3)]
        row=dict(condition=mode,seeds="0,1,2")
        for key in ("coco_ap","coco_ap50","ap_s"):
            values=[100*r["coco_no_ignore"][key] for r in rs]
            row[key]=statistics.mean(values);row[key+"_sample_sd"]=statistics.stdev(values)
            native=[100*gate["runs"][f"native__seed{s}"]["coco_no_ignore"][key] for s in range(3)]
            delta=[v-n for v,n in zip(values,native)]
            row[key+"_paired_delta"]=statistics.mean(delta);row[key+"_paired_delta_sd"]=statistics.stdev(delta)
        gate_rows.append(row)
    with (args.output/"testdev_gate_unrounded.csv").open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(gate_rows[0]));w.writeheader();w.writerows(gate_rows)
    external=json.loads((ROOT/"results/summary/external_comparison_statistics.json").read_text())
    timing=json.loads((ROOT/"results/summary/table5_unified_timing_statistics.json").read_text())
    by_timing={r["experiment"]:r for r in timing["models"]}
    external_rows=[]
    for model in ("yolo26n","p2_baseline","yolo11n","yolo26s","gsf_yolo","rtmdet_tiny","rtdetrv2_r18"):
        source=external["models"][model];cost=source["complexity"]
        seeds=by_timing[model]["training_seeds"]
        if model in ("rtmdet_tiny","rtdetrv2_r18"):
            native=source["native_record"]["native_metrics_percent"]
            aps,ap50s=[native["ap"]],[native["ap50"]]
            small=[source["native_record"]["ap_s_coco300_percent"]]
        else:
            aps,ap50s=[],[]
            for seed in seeds:
                path=ROOT/f"results/training_logs/{model}/seed{seed}/results.csv"
                with path.open(encoding="utf-8-sig",newline="") as f:logs=list(csv.DictReader(f))
                best=max(logs,key=lambda r:float(r["metrics/mAP50-95(B)"]))
                aps.append(100*float(best["metrics/mAP50-95(B)"]))
                ap50s.append(100*float(best["metrics/mAP50(B)"]))
            small=[100*r["metrics"]["ap_s"] for r in source["ap_s_source"]["records"]]
        row=dict(model=model,seeds=",".join(map(str,seeds)),native_ap=statistics.mean(aps),
            native_ap50=statistics.mean(ap50s),coco_ap_s=statistics.mean(small),
            parameters=cost["parameters"],gflops=cost["gflops"],latency_ms=by_timing[model]["latency_mean_ms"])
        for key,values in [("native_ap",aps),("native_ap50",ap50s),("coco_ap_s",small)]:
            row[key+"_sample_sd"]=statistics.stdev(values) if len(values)>1 else None
        row["latency_sample_sd_ms"]=by_timing[model]["latency_seed_sample_sd_ms"]
        external_rows.append(row)
    with (args.output/"external_unrounded.csv").open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(external_rows[0]));w.writeheader();w.writerows(external_rows)
    with (ROOT/"results/summary/final_model_testdev_source.csv").open(encoding="utf-8-sig",newline="") as f:final=list(csv.DictReader(f))
    m5=json.loads((ROOT/"results/testdev/dsf_p2_widening.json").read_text())
    arch_rows=[]
    for model in ("p2_baseline","dsf_p2_widening","gsf_yolo"):
        source=[r["coco_no_ignore"] for r in m5["runs"] if r["experiment_id"]==model] if model=="dsf_p2_widening" else [r for r in final if r["experiment_id"]==model]
        row=dict(model=model,seeds="0,1,2")
        for key in ("coco_ap","coco_ap50","ap_s"):
            values=[100*float(r[key]) for r in source]
            assert len(values)==3
            row[key]=statistics.mean(values);row[key+"_sample_sd"]=statistics.stdev(values)
        arch_rows.append(row)
    with (args.output/"testdev_architecture_unrounded.csv").open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(arch_rows[0]));w.writeheader();w.writerows(arch_rows)
    gain={"comparison":"gsf_yolo minus dsf_p2_widening","seeds":"0,1,2"}
    for key,stats in m5["paired_M6_minus_M5"].items():
        gain[key+"_paired_delta_pp"]=100*statistics.mean(stats["values"])
        gain[key+"_paired_delta_sd_pp"]=100*statistics.stdev(stats["values"])
    with (args.output/"testdev_widening_paired_gain.csv").open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(gain));w.writeheader();w.writerow(gain)
    validation=json.loads((ROOT/"results/summary/table5_seed01_statistics.json").read_text())
    validation_rows=[]
    for source in validation["rows"]:
        row=dict(condition=source["mode"],seeds="0,1")
        for key in ("ap_s","small_recall","fp_per_image","background_per_image"):
            values=[source[key]["seed0"],source[key]["seed1"]]
            row[key]=statistics.mean(values);row[key+"_sample_sd"]=statistics.stdev(values)
        validation_rows.append(row)
    with (args.output/"validation_gate_unrounded.csv").open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(validation_rows[0]));w.writeheader();w.writerows(validation_rows)
    appendix=json.loads((ROOT/"results/testdev/dsf_refinement_supplement.json").read_text())
    reference=json.loads((ROOT/"results/summary/final_model_testdev_statistics.json").read_text())
    appendix_rows=[]
    conditions=[("p2_baseline",None),("dsf_refinement:spatial_mean","valid_spatial_mean"),
        ("dsf_refinement:spatial_shuffle","valid_shuffle_0"),("dsf_refinement:L1_matched_shuffle","valid_shuffle_l1_0"),
        ("dsf_refinement","native"),("gsf_yolo",None)]
    for label,mode in conditions:
        runs=[appendix["runs"][f"{mode}__seed{s}"] if mode else reference["runs"][f"{label}__seed{s}"] for s in range(3)]
        row=dict(condition=label,seeds="0,1,2")
        for key in ("coco_ap","coco_ap50","ap_s"):
            values=[100*r["coco_no_ignore"][key] for r in runs]
            row[key]=statistics.mean(values);row[key+"_sample_sd"]=statistics.stdev(values)
        for key in ("small_recall","fp_per_image","background_per_image"):
            values=[r["diagnostics_conf025"][key]*(100 if key=="small_recall" else 1) for r in runs]
            row[key]=statistics.mean(values);row[key+"_sample_sd"]=statistics.stdev(values)
        appendix_rows.append(row)
    with (args.output/"appendix_testdev_unrounded.csv").open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(appendix_rows[0]));w.writeheader();w.writerows(appendix_rows)
    print("PASS",len(rows),"ablation rows,",len(external_rows),"external rows,",len(arch_rows),"test-dev architecture rows,",len(gate_rows),"test-dev gate rows,",len(validation_rows),"validation gate rows and",len(appendix_rows),"Appendix-B rows")


if __name__=="__main__":main()
