# Derived from Ultralytics; AGPL-3.0. See the repository LICENSE.
"""PyTorch inference adapter for the manuscript's detection checkpoints."""
import numpy as np
import torch
import torch.nn as nn

from ultralytics.utils import YAML
from ultralytics.utils.nms import non_max_suppression


def check_class_names(names):
    if isinstance(names, list):
        names = dict(enumerate(names))
    names = {int(key): str(value) for key, value in names.items()}
    if names and (min(names) < 0 or max(names) >= len(names)):
        raise KeyError("Detection category indices must be consecutive from zero")
    return names


def default_class_names(data=None):
    if data:
        try:
            return check_class_names(YAML.load(data)["names"])
        except (OSError, KeyError, TypeError):
            pass
    return {index: f"class{index}" for index in range(999)}


class AutoBackend(nn.Module):
    def __init__(self, model="yolo26n.pt", device=torch.device("cpu"), dnn=False,
                 data=None, fp16=False, fuse=True, verbose=True):
        super().__init__()
        if dnn:
            raise ValueError("The manuscript runtime uses native PyTorch inference")
        self.nn_module = isinstance(model, torch.nn.Module)
        if self.nn_module:
            model = model.fuse(verbose=verbose) if fuse else model
            model = model.to(device)
        else:
            from ultralytics.nn.tasks import load_checkpoint
            model, _ = load_checkpoint(model, device=device, fuse=fuse)
        model.half() if fp16 else model.float()
        for parameter in model.parameters():
            parameter.requires_grad = False
        self.model = model
        self.device = device
        self.fp16 = bool(fp16)
        self.stride = max(int(model.stride.max()), 32)
        self.ch = model.yaml.get("channels", 3)
        self.names = check_class_names(model.module.names if hasattr(model, "module") else model.names)
        self.end2end = getattr(model, "end2end", False)
        self.pt = True
        self.jit = self.imx = self.triton = self.nhwc = False
        self.dynamic = False
        self.metadata = None

    def forward(self, im, augment=False, visualize=False, embed=None, **kwargs):
        if self.fp16 and im.dtype != torch.float16:
            im = im.half()
        output = self.model(im, augment=augment, visualize=visualize, embed=embed, **kwargs)
        return list(output) if isinstance(output, (list, tuple)) else output

    def from_numpy(self, value):
        return torch.tensor(value).to(self.device) if isinstance(value, np.ndarray) else value

    def warmup(self, imgsz=(1, 3, 640, 640)):
        if self.device.type != "cpu":
            image = torch.empty(*imgsz, dtype=torch.float16 if self.fp16 else torch.float32, device=self.device)
            self.forward(image)
            boxes = torch.rand(1, 84, 16, device=self.device)
            boxes[:, :4] *= imgsz[-1]
            non_max_suppression(boxes)
