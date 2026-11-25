"""Portable redraw of the archived DSF+refinement functional diagnostics."""
from pathlib import Path
import argparse,csv,json,os
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR",str(ROOT/".runtime/matplotlib"))


def csv_rows(path):
    with path.open(encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,default=ROOT/"reproduced/gate_summary")
    args=p.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    fig,axes=plt.subplots(3,2,figsize=(8,9))
    a,b,c,d,e,unused=axes.flatten()
    transitions=csv_rows(ROOT/"results/gate_figure/fig4a_transitions.csv")
    a.bar([r["transition"] for r in transitions],[float(r["mean"]) for r in transitions],yerr=[float(r["sample_sd"]) for r in transitions],color=["#458B8B","#9B794D"],capsize=3)
    a.set(title="(a) Detection transitions",ylabel="Small targets")
    curves=csv_rows(ROOT/"results/gate_figure/fig4b_curves.csv")
    for model,label,color in [("M0","P2 baseline","#65717E"),("M1","DSF + refinement","#24598A")]:
        selected=[r for r in curves if r["model"]==model]
        b.errorbar([float(r["budget"]) for r in selected],[float(r["mean"]) for r in selected],yerr=[float(r["sample_sd"]) for r in selected],marker="o",color=color,label=label)
    b.set(title="(b) Recall at FP budgets",xlabel="FP/image budget",ylabel="Small-object recall (%)");b.legend(fontsize=8)
    interventions=csv_rows(ROOT/"results/gate_figure/fig4c_interventions.csv")
    modes=["zero","one","valid_spatial_mean","valid_shuffle_0","valid_shuffle_l1_0","native"]
    selected=[next(r for r in interventions if r["mode"]==m) for m in modes]
    c.barh(["Zero","Unit","Spatial mean","Shuffle","L1-matched shuffle","Native"],[float(r["mean"]) for r in selected],xerr=[float(r["sample_sd"]) for r in selected],color="#6786A6",capsize=3)
    c.set(title="(c) Frozen gate controls",xlabel="Small-object AP (%)");c.invert_yaxis()
    regional=json.loads((ROOT/"results/summary/gate_injection_statistics.json").read_text())
    for ax,metric,title in [(d,"gate_mean","(d) Matched gate contrast"),(e,"injection_abs","(e) Matched injection contrast")]:
        rows=[r for r in regional["region_statistics"] if r["metric"]==metric]
        means=np.array([r["mean"] for r in rows]);lo=np.array([r["ci95"][0] for r in rows]);hi=np.array([r["ci95"][1] for r in rows])
        ax.errorbar(range(3),means,yerr=np.stack([means-lo,hi-means]),fmt="o",capsize=4,color="#24598A")
        ax.axhline(0,color="#999999",linewidth=.7);ax.set(xticks=range(3),xticklabels=[r["group"] for r in rows],title=title,ylabel="Object minus background")
    unused.axis("off")
    unused.text(0,.7,"DSF + refinement; training seeds 0 and 1\nPanels a-c: mean and sample SD\nPanels d-e: paired image-bootstrap 95% intervals\nFunctional interventions keep parameters frozen",fontsize=9)
    fig.tight_layout();args.output.parent.mkdir(parents=True,exist_ok=True)
    for suffix in (".png",".pdf",".svg"):fig.savefig(args.output.with_suffix(suffix),dpi=300)
    ratio_rows=[r for r in regional["region_statistics"] if r["metric"]=="injection_to_detail"]
    ratio_fig,ratio_ax=plt.subplots(figsize=(5.5,3.3))
    means=np.array([r["mean"] for r in ratio_rows]);lo=np.array([r["ci95"][0] for r in ratio_rows]);hi=np.array([r["ci95"][1] for r in ratio_rows])
    ratio_ax.errorbar(range(3),means,yerr=np.stack([means-lo,hi-means]),fmt="D",capsize=4,color="#24598A",label="Two-checkpoint mean")
    for key,marker in [("seed0","o"),("seed1","s")]:ratio_ax.scatter(range(3),[r[key] for r in ratio_rows],marker=marker,facecolors="white",edgecolors="#24598A",label=key)
    ratio_ax.axhline(0,color="#999999",linewidth=.7)
    ratio_ax.set(xticks=range(3),xticklabels=[r["group"] for r in ratio_rows],ylabel="GT minus background injection/detail ratio")
    ratio_ax.legend(fontsize=8);ratio_fig.tight_layout()
    for suffix in (".png",".pdf",".svg"):ratio_fig.savefig(args.output.parent/("gate_injection_ratio"+suffix),dpi=300)
    print("Generated gate diagnostic figure")


if __name__=="__main__":main()
