"""Custom confidence-threshold diagnostics and ignored-region handling; not official VisDrone scoring."""
from pathlib import Path
import argparse
from collections import defaultdict
import gzip
import json
import math
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from analysis.diagnostics import diagnose, aggregate


class Dataset:
    def __init__(self, ground_truth, raw_annotations):
        self.gt=json.loads(ground_truth.read_text(encoding="utf-8-sig"))
        self.images=self.gt["images"]
        self.annotations=defaultdict(list)
        for ann in self.gt["annotations"]:
            self.annotations[ann["image_id"]].append(ann)
        self.ignores={}
        for im in self.images:
            path=raw_annotations/(Path(im["file_name"]).stem+".txt")
            ignored=[]
            for line in path.read_text(encoding="utf-8-sig").splitlines():
                row=[float(v) for v in line.rstrip(",").split(",")]
                if len(row)<8:
                    raise ValueError("Raw VisDrone annotation must have eight columns")
                x,y,w,h,score,category=row[:6]
                if category in (0,11) or score==0:
                    ignored.append([x,y,w,h])
            self.ignores[im["id"]]=ignored

    def ignore_integral(self, im):
        mask=np.zeros((im["height"],im["width"]),dtype=np.uint8)
        for x,y,w,h in self.ignores[im["id"]]:
            x0,y0=max(0,math.floor(x)),max(0,math.floor(y))
            x1,y1=min(im["width"],math.ceil(x+w)),min(im["height"],math.ceil(y+h))
            if x1>x0 and y1>y0:
                mask[y0:y1,x0:x1]=1
        return cv2.integral(mask,sdepth=cv2.CV_64F)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth",type=Path,required=True)
    parser.add_argument("--raw-annotations",type=Path,required=True)
    parser.add_argument("--predictions",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--confidence",type=float,default=.25)
    args=parser.parse_args()
    opener=gzip.open if args.predictions.suffix==".gz" else open
    with opener(args.predictions,"rt",encoding="utf-8-sig") as f:
        predictions=json.load(f)
    ds=Dataset(args.ground_truth,args.raw_annotations)
    rows,objects=diagnose(ds,predictions)
    summary,perimage=aggregate(rows,ds,args.confidence)
    args.output.mkdir(parents=True,exist_ok=True)
    summary={k:v for k,v in summary.items() if k in {"confidence","n_gt","n_small","small_recall","fp_per_image","background_per_image"}}
    for name,data in [("summary",summary),("per_image",perimage),("per_object",objects)]:
        (args.output/(name+".json")).write_text(json.dumps(data,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps(summary,indent=2))


if __name__=="__main__":main()
