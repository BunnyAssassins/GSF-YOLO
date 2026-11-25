# Copyright (c) OpenMMLab. All rights reserved.
# Registration/import surface scoped to the manuscript external baselines.
from .gfocal_loss import QualityFocalLoss, DistributionFocalLoss
from .cross_entropy_loss import CrossEntropyLoss
from .iou_loss import GIoULoss, IoULoss, DIoULoss, CIoULoss
from .utils import reduce_loss, weight_reduce_loss, weighted_loss
