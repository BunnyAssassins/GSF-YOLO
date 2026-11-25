# Derived from Ultralytics; AGPL-3.0. See the repository LICENSE.
from __future__ import annotations
from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.utils.ops import xywh2xyxy
from ultralytics.utils.tal import TaskAlignedAssigner, dist2bbox, make_anchors
from .metrics import bbox_iou
from .tal import bbox2dist

class DFLoss(nn.Module):

    def __init__(self, reg_max: int=16) -> None:
        super().__init__()
        self.reg_max = reg_max

    def __call__(self, pred_dist: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.clamp_(0, self.reg_max - 1 - 0.01)
        tl = target.long()
        tr = tl + 1
        wl = tr - target
        wr = 1 - wl
        return (F.cross_entropy(pred_dist, tl.view(-1), reduction='none').view(tl.shape) * wl + F.cross_entropy(pred_dist, tr.view(-1), reduction='none').view(tl.shape) * wr).mean(-1, keepdim=True)

class BboxLoss(nn.Module):

    def __init__(self, reg_max=16, bbox_loss='ciou'):
        super().__init__()
        if bbox_loss.lower() != 'ciou':
            raise ValueError('The manuscript uses CIoU box loss')
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None

    def forward(self, pred_dist: torch.Tensor, pred_bboxes: torch.Tensor, anchor_points: torch.Tensor, target_bboxes: torch.Tensor, target_scores: torch.Tensor, target_scores_sum: torch.Tensor, fg_mask: torch.Tensor, imgsz: torch.Tensor, stride: torch.Tensor, anchor_weights: torch.Tensor | None=None) -> tuple[torch.Tensor, torch.Tensor]:
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        if anchor_weights is not None:
            expanded_anchor_weights = anchor_weights.expand(fg_mask.shape[0], -1, -1)
            weight = weight * expanded_anchor_weights[fg_mask]
        pred_fg = pred_bboxes[fg_mask]
        target_fg = target_bboxes[fg_mask]
        iou_loss = 1.0 - bbox_iou(pred_fg, target_fg, xywh=False, CIoU=True)
        box_loss = iou_loss
        loss_iou = (box_loss * weight).sum() / target_scores_sum
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            target_ltrb = bbox2dist(anchor_points, target_bboxes)
            target_ltrb = target_ltrb * stride
            target_ltrb[..., 0::2] /= imgsz[1]
            target_ltrb[..., 1::2] /= imgsz[0]
            pred_dist = pred_dist * stride
            pred_dist[..., 0::2] /= imgsz[1]
            pred_dist[..., 1::2] /= imgsz[0]
            loss_dfl = F.l1_loss(pred_dist[fg_mask], target_ltrb[fg_mask], reduction='none').mean(-1, keepdim=True) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        return (loss_iou, loss_dfl)

class v8DetectionLoss:

    def __init__(self, model, tal_topk: int=10, tal_topk2: int | None=None):
        device = next(model.parameters()).device
        h = model.args
        m = model.model[-1]
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.hyp = h
        self.stride = m.stride
        self.nc = m.nc
        self.no = m.nc + m.reg_max * 4
        self.reg_max = m.reg_max
        self.device = device
        self.use_dfl = m.reg_max > 1
        self.assigner = TaskAlignedAssigner(topk=tal_topk, num_classes=self.nc, alpha=0.5, beta=6.0, stride=self.stride.tolist(), topk2=tal_topk2)
        bbox_loss = getattr(h, 'bbox_loss', 'ciou').lower()
        if bbox_loss != 'ciou':
            raise ValueError('This manuscript uses CIoU box loss')
        self.bbox_loss = BboxLoss(m.reg_max, bbox_loss=bbox_loss).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

    def preprocess(self, targets: torch.Tensor, batch_size: int, scale_tensor: torch.Tensor) -> torch.Tensor:
        (nl, ne) = targets.shape
        if nl == 0:
            out = torch.zeros(batch_size, 0, ne - 1, device=self.device)
        else:
            i = targets[:, 0]
            (_, counts) = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)
            for j in range(batch_size):
                matches = i == j
                if (n := matches.sum()):
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points: torch.Tensor, pred_dist: torch.Tensor) -> torch.Tensor:
        if self.use_dfl:
            (b, a, c) = pred_dist.shape
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def get_assigned_targets_and_loss(self, preds: dict[str, torch.Tensor], batch: dict[str, Any]) -> tuple:
        loss = torch.zeros(3, device=self.device)
        (pred_distri, pred_scores) = (preds['boxes'].permute(0, 2, 1).contiguous(), preds['scores'].permute(0, 2, 1).contiguous())
        (anchor_points, stride_tensor) = make_anchors(preds['feats'], self.stride, 0.5)
        level_gains = [float(getattr(self.hyp, 'p2_loss_gain', 1.0)), float(getattr(self.hyp, 'p3_loss_gain', 1.0))]
        level_gains.extend([1.0] * max(len(preds['feats']) - len(level_gains), 0))
        anchor_weights = torch.cat([torch.full((feat.shape[2] * feat.shape[3],), level_gains[i], device=self.device) for (i, feat) in enumerate(preds['feats'])]).view(1, -1, 1)
        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(preds['feats'][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        targets = torch.cat((batch['batch_idx'].view(-1, 1), batch['cls'].view(-1, 1), batch['bboxes']), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        (gt_labels, gt_bboxes) = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)
        (_, target_bboxes, target_scores, fg_mask, target_gt_idx) = self.assigner(pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype), anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)
        target_scores_sum = max(target_scores.sum(), 1)
        loss[1] = (self.bce(pred_scores, target_scores.to(dtype)) * anchor_weights).sum() / target_scores_sum
        if fg_mask.sum():
            (loss[0], loss[2]) = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes / stride_tensor, target_scores, target_scores_sum, fg_mask, imgsz, stride_tensor, anchor_weights)
        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl
        return ((fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor), loss, loss.detach())

    def parse_output(self, preds: dict[str, torch.Tensor] | tuple[torch.Tensor, dict[str, torch.Tensor]]) -> torch.Tensor:
        return preds[1] if isinstance(preds, tuple) else preds

    def __call__(self, preds: dict[str, torch.Tensor] | tuple[torch.Tensor, dict[str, torch.Tensor]], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        return self.loss(self.parse_output(preds), batch)

    def loss(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = preds['boxes'].shape[0]
        (loss, loss_detach) = self.get_assigned_targets_and_loss(preds, batch)[1:]
        return (loss * batch_size, loss_detach)

class E2ELoss:

    def __init__(self, model, loss_fn=v8DetectionLoss):
        self.one2many = loss_fn(model, tal_topk=10)
        self.one2one = loss_fn(model, tal_topk=7, tal_topk2=1)
        self.updates = 0
        self.total = 1.0
        self.o2m = 0.8
        self.o2o = self.total - self.o2m
        self.o2m_copy = self.o2m
        self.final_o2m = 0.1

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        preds = self.one2many.parse_output(preds)
        (one2many, one2one) = (preds['one2many'], preds['one2one'])
        loss_one2many = self.one2many.loss(one2many, batch)
        loss_one2one = self.one2one.loss(one2one, batch)
        return (loss_one2many[0] * self.o2m + loss_one2one[0] * self.o2o, loss_one2one[1])

    def update(self) -> None:
        self.updates += 1
        self.o2m = self.decay(self.updates)
        self.o2o = max(self.total - self.o2m, 0)

    def decay(self, x) -> float:
        return max(1 - x / max(self.one2one.hyp.epochs - 1, 1), 0) * (self.o2m_copy - self.final_o2m) + self.final_o2m
