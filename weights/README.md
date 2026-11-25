# Publication checkpoints

The 32 publication checkpoints are distributed as assets of the [`v1.0-paper` release](https://github.com/BunnyAssassins/GSF-YOLO/releases/tag/v1.0-paper). The six primary architecture configurations and three fusion/refinement controls each have training seeds 0, 1 and 2; the five external detector configurations each have seed 0.

[`../manifests/checkpoints.csv`](../manifests/checkpoints.csv) records the download URL, local destination, selected epoch, framework, original checkpoint SHA256, published checkpoint SHA256 and tensor verification for every checkpoint. Download each asset to its `file` destination in that manifest. Asset names use `<configuration>__seed<seed>.pt` or `.pth`.

The published files contain inference parameters and the architecture information needed by the native loader. Local paths, repository author metadata, optimizer state, scaler state and other training-only metadata were removed. The original research checkpoints were preserved. Every published checkpoint was reloaded and every inference tensor was compared with the original, including tensor values, shape and dtype. Tensor-equivalence flags and published hashes are recorded in the checkpoint manifest.

These files are **inference checkpoints**. They do not contain the optimizer state required to resume a historical training run. Training configuration and logs are available under [`../results/training_logs`](../results/training_logs).

## Native loading

Run from the repository root after installing its dependencies and downloading the weights.

```python
from ultralytics import YOLO
model = YOLO("weights/gsf_yolo/seed0.pt")
```

The MMDetection checkpoint retains `state_dict`, `ema_state_dict` and public dataset metadata. Use the released RTMDet configuration with `mmdet.apis.init_detector`; its ordinary MMEngine checkpoint loader can load the sanitized file.

The RT-DETRv2 checkpoint retains both `model` and `ema["module"]` state dictionaries. The publication inference uses `ema["module"]`, loaded with `strict=True` into the released R18 architecture. Use the repository's inference/evaluation entry points so that native preprocessing is retained.
