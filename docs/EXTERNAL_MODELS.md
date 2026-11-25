# Native external models

RTMDet-tiny and RT-DETRv2-S/R18 use their native frameworks and disclosed recipes. They share the corrected full-image dataset, 832 input and 100-epoch budget, but their optimizers, augmentation, preprocessing and postprocessing differ from the YOLO runs. Each external model was trained with seed0. A common COCO300 scoring step does not remove those recipe differences.

## Source versions and licenses

| Framework | Verified upstream revision | Included source |
|---|---|---|
| [MMDetection](https://github.com/open-mmlab/mmdetection) 3.3.0 | `44ebd17b145c2372c4b700bfb9cb20dbd28ab64a` | RTMDet dependency subset in `external/mmdetection/mmdet` and `external/mmdetection/tools/train.py` |
| [RT-DETR](https://github.com/lyuwenyu/RT-DETR), RT-DETRv2 PyTorch | `df2a96c8c3a098567a1149bf0248da6efe6d495d` | R18 dependency subset in `external/rtdetrv2/src` and `external/rtdetrv2/tools/train.py` |
| [MMCV](https://github.com/open-mmlab/mmcv) 2.1.0 | `57c4e25e06e2d4f8a9357c84bcd24089a284dc88` | Operator library is an environment dependency; compatibility patch is included separately |

Both model upstreams use Apache-2.0; their original licenses are retained in the corresponding source directories. The published code contains scoped dependency subsets with adapted registration imports. `external/SOURCE_PROVENANCE.json` records upstream commits, subset changes and current file SHA-256 values. `configs/external/rtmdet_tiny.py` and `configs/external/rtdetrv2_r18.yml` are resolved, independent manuscript configurations; the unused upstream configuration trees are omitted.

An upstream checkout can be obtained independently with `git clone` followed by `git checkout <revision>` using the revisions above. The vendored subset avoids requiring a moving upstream branch to load the published models.

## Dependencies

Use the root environment specification for PyTorch and torchvision. RTMDet additionally requires **MMCV with compiled operators**, MMEngine and the runtime dependencies in `external/mmdetection/requirements/runtime.txt`. The measured environment used MMCV2.1.0 and MMEngine0.10.7. `mmcv-lite` cannot provide the required operator implementation. MMDetection3.3.0 requires MMCV below2.2.0; changing the Torch/CUDA build may require rebuilding MMCV for that build.


```bash
pip install mmengine==0.10.7
pip install -r external/mmdetection/requirements/runtime.txt
pip install faster-coco-eval==1.7.0 tensorboard scipy PyYAML pycocotools
```

Install or build MMCV2.1.0 against the chosen Torch/CUDA environment rather than letting an unconstrained package command replace core packages. The original Windows/Torch2.9.1/CUDA13.0 build used `external/mmcv_torch291_windows_headers.patch`: four unused `torch/script.h` includes were replaced by `torch/types.h` to resolve a Windows NVCC header ambiguity. It does not change operator implementations. This patch addresses that specific build, and should not be applied automatically on every platform.

## Native input/output adapters

`scripts/external_predictors.py` copies the original measurement calls:

- RTMDet uses `inference_detector` on a decoded OpenCV BGR array, removes `LoadAnnotations` from the inference pipeline, retains aspect ratio while resizing to832, pads to832, and uses native score threshold0.001, NMS IoU0.7 and cap300.
- RT-DETRv2 converts BGR to RGB/PIL, resizes directly to832×832 with torchvision/PIL bilinear interpolation, converts to floating tensor divided by255, transfers to the device and runs the native model/postprocessor. The postprocessor restores original-image coordinates and selects300 queries. It uses no NMS.
- Ultralytics keeps the native checkpoint head and its square letterbox preprocessing.

The benchmark synchronizes around the entire native prediction call. Detection conversion/serialization happens outside the timed interval. These native preprocessing differences are recorded with each round; the shared input budget does not imply identical transforms.

```bash
python scripts/evaluate_external.py --weights weights/rtmdet_tiny/seed0.pth --split val --device 0 --output outputs/external/rtmdet_coco300
python scripts/evaluate_external.py --weights weights/rtdetrv2_r18/seed0.pth --split val --device 0 --output outputs/external/rtdetr_coco300
python scripts/benchmark_latency.py --weights weights/gsf_yolo/seed0.pt weights/rtmdet_tiny/seed0.pth weights/rtdetrv2_r18/seed0.pth --images datasets/VisDrone2019-DET-corrected/val/images --framework auto --output outputs/latency/native_external
```

Framework auto-selection uses the published checkpoint parent directory names. Use `--framework rtmdet` or `--framework rtdetrv2` when loading a checkpoint from another directory. Evaluation can use `--device cpu`; CPU inference does not reproduce workstation GPU latency. Supplementary evaluation still uses corrected original-image ground truth and COCO `maxDets=[1,10,300]`, without official ignored-region filtering.

## Native training recipes

The parameterized configs are `configs/external/rtmdet_tiny.py` and `configs/external/rtdetrv2_r18.yml`. RTMDet uses AdamW with learning rate `0.004*6/256=0.00009375`, weight decay0.05, CachedMosaic/MixUp/HSV augmentation, EMA and a switch at epoch80. RT-DETRv2 uses AdamW1e-4, weight decay1e-4 (zero for norm/bn), photometric/zoom/IoU-crop augmentation and a stop at epoch80. Both use batch6, 100epochs, seed0, AMP and no pretrained backbone initialization. Their native validation evaluator/checkpoint-selection protocol remains distinct from supplementary COCO300.

```bash
python scripts/prepare_external_data.py
python scripts/train_external.py --framework rtmdet
python scripts/train_external.py --framework rtdetrv2
```

The two commands preview only. Add `--start-training` to run the selected upstream trainer in a new output directory. The launcher resolves the selected dataset and annotation views without modifying source data. Published inference checkpoints have no optimizer state and cannot resume these runs. Original native logs and selection metadata are retained under `results/training_logs/`.
