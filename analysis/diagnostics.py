"""Diagnostic functions from the frozen-checkpoint study (custom ignore rules, not official scoring)."""
from collections import defaultdict
import math
import numpy as np
def size_group(area):
    # Disjoint diagnostic groups; COCO AP's own boundary behavior is left unchanged.
    return 'small' if area < 1024 else ('medium' if area < 9216 else 'large')


def xyxy(boxes):
    a = np.asarray(boxes, dtype=np.float64).reshape(-1, 4).copy()
    a[:, 2:] += a[:, :2]
    return a


def iou_matrix(a, b):
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    wh = np.maximum(np.minimum(a[:, None, 2:], b[None, :, 2:]) -
                    np.maximum(a[:, None, :2], b[None, :, :2]), 0)
    inter = wh.prod(2)
    area_a = np.maximum(a[:, 2:] - a[:, :2], 0).prod(1)
    area_b = np.maximum(b[:, 2:] - b[:, :2], 0).prod(1)
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-12)


def split_by_image(preds):
    result = defaultdict(list)
    for p in preds:
        result[p['image_id']].append(p)
    return result


def integral_value(integral, x, y):
    h, w = np.array(integral.shape) - 1
    x, y = np.clip(x, 0, w), np.clip(y, 0, h)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.minimum(x0+1, w), np.minimum(y0+1, h)
    dx, dy = x-x0, y-y0
    return ((1-dx)*(1-dy)*integral[y0, x0] + dx*(1-dy)*integral[y0, x1]
            + (1-dx)*dy*integral[y1, x0] + dx*dy*integral[y1, x1])


def ignore_ioa(boxes, integral):
    if not len(boxes):
        return np.zeros(0)
    x0,y0,x1,y1 = boxes.T
    area = np.maximum((x1-x0)*(y1-y0), 1e-12)
    overlap = (integral_value(integral,x1,y1) - integral_value(integral,x0,y1)
               - integral_value(integral,x1,y0) + integral_value(integral,x0,y0))
    return np.clip(overlap/area, 0, 1)


def classify(im, anns, predictions, integral):
    """The manuscript's score-ordered, same-class one-to-one threshold diagnostic."""
    preds = sorted(predictions, key=lambda p: -p['score'])
    gb = xyxy([a['bbox'] for a in anns])
    pb = xyxy([p['bbox'] for p in preds])
    gc = np.array([a['category_id'] for a in anns])
    overlaps = iou_matrix(pb, gb)
    ioas = ignore_ioa(pb, integral)
    matched = np.zeros(len(anns), dtype=bool)
    gt_scores = np.zeros(len(anns))
    gt_ious = np.zeros(len(anns))
    rows = []
    for j,p in enumerate(preds):
        ov = overlaps[j]
        same = gc == p['category_id']
        candidates = np.flatnonzero(same & ~matched & (ov >= .5))
        match = -1
        if len(candidates):
            match = int(candidates[np.argmax(ov[candidates])])
            matched[match] = True
            gt_scores[match] = p['score']
            gt_ious[match] = ov[match]
            kind = 'tp'
        elif ioas[j] > .5:
            kind = 'ignored'
        elif np.any(same & (ov >= .5)):
            kind = 'duplicate'
        elif np.any((~same) & (ov >= .5)):
            kind = 'classification'
        elif np.any(same & (ov >= .1)):
            kind = 'localization'
        elif np.any(ov >= .1):
            kind = 'classification_localization'
        else:
            kind = 'background'
        rows.append(dict(image_id=im['id'], file_name=im['file_name'], score=p['score'],
                         category_id=p['category_id'], error=kind,
                         gt_id=anns[match]['id'] if match >= 0 else -1,
                         small_tp=int(match >= 0 and anns[match]['area'] < 1024),
                         max_iou=float(ov.max()) if len(ov) else 0., ignore_ioa=float(ioas[j]),
                         x=float(pb[j,0]),y=float(pb[j,1]),w=float(pb[j,2]-pb[j,0]),h=float(pb[j,3]-pb[j,1])))
    return rows, gt_scores, gt_ious


def aggregate(rows, ds, conf):
    counts = defaultdict(int)
    perimage = {im['id']: dict(image_id=im['id'], tp=0,fp=0,background=0,small_tp=0,
                             n_gt=len(ds.annotations[im['id']]),
                             n_small=sum(a['area']<1024 for a in ds.annotations[im['id']])) for im in ds.images}
    for row in rows:
        if row['score'] <= conf:
            continue
        kind = row['error']
        counts[kind] += 1
        p = perimage[row['image_id']]
        if kind == 'tp':
            p['tp'] += 1
            p['small_tp'] += row['small_tp']
        elif kind != 'ignored':
            p['fp'] += 1
            p['background'] += int(kind=='background')
    n_gt = sum(p['n_gt'] for p in perimage.values())
    n_small = sum(p['n_small'] for p in perimage.values())
    tp = sum(p['tp'] for p in perimage.values())
    fp = sum(p['fp'] for p in perimage.values())
    small_tp = sum(p['small_tp'] for p in perimage.values())
    return dict(confidence=conf,n_gt=n_gt,n_small=n_small,
                small_recall=small_tp/max(n_small,1),
                fp_per_image=fp/len(ds.images),background_per_image=counts['background']/len(ds.images)), list(perimage.values())


def operating_points(rows, ds):
    rows = sorted([r for r in rows if r['error']!='ignored'],key=lambda r:-r['score'])
    if not rows:
        return {}
    tp = np.cumsum([r['error']=='tp' for r in rows])
    fp = np.arange(1,len(rows)+1)-tp
    bg = np.cumsum([r['error']=='background' for r in rows])
    stp = np.cumsum([r['small_tp'] for r in rows])
    n_gt = sum(len(v) for v in ds.annotations.values())
    n_small = sum(a['area']<1024 for anns in ds.annotations.values() for a in anns)
    scores = np.array([r['score'] for r in rows])
    # Only endpoints of tied scores correspond to attainable threshold operating points.
    ends = np.r_[np.flatnonzero(scores[:-1]!=scores[1:]),len(scores)-1]
    result = {'fp_per_image_budgets':[]}
    def at(i):
        return dict(threshold_inclusive=float(scores[i]),
                    small_recall=float(stp[i]/max(n_small,1)),fp_per_image=float(fp[i]/len(ds.images)))
    for budget in [1,5,10,20,50]:
        feasible = ends[fp[ends]/len(ds.images)<=budget]
        result['fp_per_image_budgets'].append(dict(budget=budget,measured=at(feasible[-1]) if len(feasible) else None))
    return result


def paired_bootstrap(a,b,n=2000):
    """Image cluster bootstrap of micro-averaged recall and counts/image, fixed image set."""
    rng=np.random.default_rng(20261017)
    fields=['tp','fp','background','small_tp','n_gt','n_small']
    x=np.array([[r[k] for k in fields] for r in a],dtype=float)
    y=np.array([[r[k] for k in fields] for r in b],dtype=float)
    def metrics(z):
        s=z.sum(0)
        return np.array([s[3]/max(s[5],1)])
    actual=metrics(y)-metrics(x)
    values=[]
    for _ in range(n):
        ix=rng.integers(0,len(x),len(x))
        values.append(metrics(y[ix])-metrics(x[ix]))
    lo,hi=np.quantile(values,[.025,.975],axis=0)
    return {k:dict(delta=float(d),ci95=[float(l),float(h)]) for k,d,l,h in
            zip(['small_recall'],actual,lo,hi)}


def diagnose(ds, predictions):
    grouped=split_by_image(predictions)
    rows=[]; objects=[]
    for im in ds.images:
        anns=ds.annotations[im['id']]
        current,scores,ious=classify(im,anns,grouped[im['id']],ds.ignore_integral(im))
        rows.extend(current)
        objects.extend(dict(image_id=im['id'],gt_id=a['id'],category_id=a['category_id'],area=a['area'],
                            size=size_group(a['area']),matched_score=float(s),matched_iou=float(u))
                       for a,s,u in zip(anns,scores,ious))
    return rows,objects


def roi_weights(box,shape):
    h,w=shape
    x0,y0,x1,y1=box
    l,t=max(0,int(math.floor(x0))),max(0,int(math.floor(y0)))
    r,b=min(w,int(math.ceil(x1))),min(h,int(math.ceil(y1)))
    if r<=l or b<=t:
        return None
    wx=np.maximum(np.minimum(np.arange(l,r)+1,x1)-np.maximum(np.arange(l,r),x0),0)
    wy=np.maximum(np.minimum(np.arange(t,b)+1,y1)-np.maximum(np.arange(t,b),y0),0)
    weights=wy[:,None]*wx[None,:]
    return (slice(t,b),slice(l,r)),weights


def roi_stats(maps,box):
    region=roi_weights(box,maps['gate_mean'].shape)
    if region is None:
        return None
    slices,weights=region
    weight=weights.sum()
    if weight<=0:
        return None
    av=lambda key:float((maps[key][slices]*weights).sum()/weight)
    gm=av('gate_mean')
    d,u=av('detail_abs'),av('injection_abs')
    return dict(gate_mean=gm,detail_abs=d,injection_abs=u,
                injection_to_detail=u/(d+1e-8),feature_area=float(weight))


def transform_boxes(boxes,im):
    ratio=min(832/im['height'],832/im['width'])
    nw,nh=round(im['width']*ratio),round(im['height']*ratio)
    left,top=round((832-nw)/2-.1),round((832-nh)/2-.1)
    a=np.asarray(boxes,dtype=float).reshape(-1,4)*ratio
    a+=np.array([left,top,left,top])
    return a/4,dict(ratio=ratio,left=left,top=top,resized_width=nw,resized_height=nh)


