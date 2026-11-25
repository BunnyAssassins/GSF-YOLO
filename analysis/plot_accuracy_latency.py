"""Generate the paper's six-configuration accuracy-latency comparison from released records."""
from pathlib import Path
import argparse
import json
import os
import statistics
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR",str(ROOT/".runtime/matplotlib"))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"reproduced/accuracy_latency")
    args=parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    data=json.loads((ROOT/"results/summary/ablation_accuracy_cost_statistics.json").read_text(encoding="utf-8"))
    plt.rcParams.update({"font.size":8,"pdf.fonttype":42,"svg.fonttype":"none","axes.spines.top":False,"axes.spines.right":False})
    fig,ax=plt.subplots(figsize=(7,4.2))
    colors=["#65717E","#6786A6","#9B794D","#458B8B","#8877A5","#24598A"]
    for model,color in zip(data["models"],colors):
        xs=[statistics.mean(r["rounds_ms"]) for r in model["runs"]]
        ys=[r["ap_s_percent"] for r in model["runs"]]
        ax.scatter(xs,ys,s=20,facecolors="white",edgecolors=color)
        x,y=statistics.mean(xs),statistics.mean(ys)
        ax.errorbar(x,y,xerr=statistics.stdev(xs),yerr=statistics.stdev(ys),fmt="o",color=color,capsize=3,label=model["label"])
    ax.set(xlabel="Prediction-call latency (ms)",ylabel="Small-object AP (%)")
    ax.legend(fontsize=7,loc="lower right")
    fig.tight_layout();args.output.parent.mkdir(parents=True,exist_ok=True)
    for suffix in (".pdf",".svg",".png"):fig.savefig(args.output.with_suffix(suffix),dpi=300)
    print("Generated accuracy-latency figure")


if __name__=="__main__":main()
