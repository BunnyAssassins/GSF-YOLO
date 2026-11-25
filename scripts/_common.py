"""Portable paths, provenance, and runtime configuration for release scripts."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def resolve(path):
    path = Path(path).expanduser()
    return path if path.is_absolute() else ROOT / path


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def read_yaml(path):
    import yaml
    return yaml.safe_load(Path(path).read_text(encoding="utf-8-sig"))


def init_runtime():
    sys.path.insert(0, str(ROOT))
    os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".cache" / "ultralytics"))
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")


def configure_precision():
    import cv2
    import torch
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


def runtime_environment():
    import cv2
    import torch
    import ultralytics
    # A CPU prediction can set CUDA_VISIBLE_DEVICES=-1 after an availability
    # check has been cached. Query visible device count before requesting a name.
    gpu = None
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        gpu = torch.cuda.get_device_name(0)
    return dict(python=sys.version, platform=platform.platform(), torch=torch.__version__,
                cuda=torch.version.cuda, ultralytics=ultralytics.__version__, opencv=cv2.__version__,
                gpu=gpu)


PREDICT_SETTINGS = dict(imgsz=832, batch=1, conf=0.001, iou=0.7, max_det=300,
                        half=False, rect=False, augment=False, agnostic_nms=False,
                        save=False, verbose=False)
