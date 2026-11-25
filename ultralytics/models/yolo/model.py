# Derived from Ultralytics; AGPL-3.0. See the repository LICENSE.
from __future__ import annotations
from pathlib import Path
from ultralytics.engine.model import Model
from ultralytics.nn.tasks import DetectionModel
from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator, DetectionPredictor

class YOLO(Model):
    def __init__(self, model: str | Path = "yolo26n.pt", task: str | None = None, verbose=False):
        if task not in (None, "detect"):
            raise ValueError("The manuscript runtime supports object detection")
        super().__init__(model=model, task="detect", verbose=verbose)

    @property
    def task_map(self):
        return {"detect": {"model": DetectionModel, "trainer": DetectionTrainer,
                           "validator": DetectionValidator, "predictor": DetectionPredictor}}
