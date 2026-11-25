"""Native inference adapters matching the manuscript's external timing calls."""
from __future__ import annotations

import sys
from pathlib import Path

from _common import PREDICT_SETTINGS, ROOT, resolve


def framework_for(weights, framework="auto"):
    if framework != "auto":
        return framework
    name = Path(weights).parent.name.lower()
    if name == "rtmdet_tiny":
        return "rtmdet"
    if name == "rtdetrv2_r18":
        return "rtdetrv2"
    return "ultralytics"


class NativePredictor:
    def __init__(self, model, predict, framework, settings, native_end2end=None):
        self.model = model
        self._predict = predict
        self.framework = framework
        self.settings = settings
        self.native_end2end = native_end2end

    def __call__(self, image):
        return self._predict(image)


def load_predictor(weights, framework="auto", device="0", config=None):
    import torch
    weights = resolve(weights)
    framework = framework_for(weights, framework)
    torch_device = torch.device(device if device == "cpu" or str(device).startswith("cuda") else f"cuda:{device}")
    if framework == "rtmdet":
        sys.path.insert(0, str(ROOT / "external/mmdetection"))
        from mmengine.config import Config
        from mmdet.apis import inference_detector, init_detector
        cfg = Config.fromfile(str(resolve(config or "configs/external/rtmdet_tiny.py")))
        cfg.test_dataloader.dataset.pipeline = [operation for operation in cfg.test_dataloader.dataset.pipeline
                                                if operation["type"] != "LoadAnnotations"]
        model = init_detector(cfg, str(weights), device="cpu").float().eval().to(torch_device)
        settings = dict(imgsz=832, batch=1, dtype="FP32", resize="keep_ratio_then_pad_832",
                        score_thr=0.001, nms_iou=0.7, max_det=300,
                        timing_call="mmdet.apis.inference_detector(decoded_Bgr_array)")
        return NativePredictor(model, lambda image: inference_detector(model, image).pred_instances,
                               framework, settings)
    if framework == "rtdetrv2":
        sys.path.insert(0, str(ROOT / "external/rtdetrv2"))
        import cv2
        from PIL import Image
        from torchvision.transforms import functional as transforms
        from src.core import YAMLConfig
        cfg = YAMLConfig(str(resolve(config or "configs/external/rtdetrv2_r18.yml")))
        model = cfg.model.cpu().float().eval()
        checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
        state = checkpoint["ema"]["module"] if checkpoint.get("ema") else checkpoint["model"]
        model.load_state_dict(state, strict=True)
        model = model.to(torch_device)
        postprocess = cfg.postprocessor

        def predict(image):
            height, width = image.shape[:2]
            rgb = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            tensor = transforms.pil_to_tensor(transforms.resize(rgb, [832, 832]))
            tensor = tensor.float().div_(255).unsqueeze(0).to(torch_device)
            return postprocess(model(tensor), torch.tensor([[width, height]], device=torch_device))[0]

        settings = dict(imgsz=832, batch=1, dtype="FP32", resize="direct_832x832_PIL_bilinear",
                        normalization="RGB_float_divide_255", max_det=300, nms=False,
                        timing_call="RGB conversion_resize_transfer_forward_native_postprocessor")
        return NativePredictor(model, predict, framework, settings)
    if framework != "ultralytics":
        raise ValueError(f"Unknown framework {framework}")
    from ultralytics import YOLO
    model = YOLO(str(weights))
    settings = dict(PREDICT_SETTINGS, device=device)
    return NativePredictor(model, lambda image: model.predict(image, **settings)[0],
                           framework, settings, bool(getattr(model.model, "end2end", False)))


def detection_records(result, framework, image_id):
    """Convert native outputs after inference, outside any timed interval."""
    if framework == "rtmdet":
        boxes, scores, labels = result.bboxes.cpu().tolist(), result.scores.cpu().tolist(), result.labels.cpu().tolist()
    elif framework == "rtdetrv2":
        boxes, scores, labels = result["boxes"].cpu().tolist(), result["scores"].cpu().tolist(), result["labels"].cpu().tolist()
    else:
        boxes, scores, labels = result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist(), result.boxes.cls.cpu().tolist()
    records = []
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box
        records.append(dict(image_id=image_id, category_id=int(label), score=float(score),
                            bbox=[x1, y1, x2-x1, y2-y1]))
    return records
