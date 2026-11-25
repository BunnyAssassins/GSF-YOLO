# Training configuration

The six manuscript configurations are distributed in `configs/models/`:

| File stem | Configuration |
|---|---|
| `p2_baseline` | YOLO26-P2 baseline |
| `dsf_refinement` | DSF fusion followed by C3k2 refinement |
| `p2_p3_widening_without_dsf` | P2/P3 widening, with DSF replaced by C3k2 |
| `dsf_p2_widening` | DSF and P2 widening |
| `dsf_p3_widening` | DSF and P3 widening |
| `gsf_yolo` | DSF and P2/P3 widening |

The released YAML files explicitly set `nc: 10` and retain only the `n` compound scale used in the manuscript experiments. Their backbone, neck and prediction-head definitions specify the six studied configurations and their channel allocation. Formatting and unused configuration metadata have been normalized for the release. Released checkpoints contain the corresponding ten-class trained detectors.

## Fresh training

```bash
python scripts/train.py --model gsf_yolo --seed 0 --dry-run
python scripts/train.py --model gsf_yolo --seed 0 --data datasets/VisDrone2019-DET-corrected/data.yaml --device 0
```

Run seeds 0,1,2 independently for every desired configuration, in sequence on the same GPU. Fresh runs use a new output directory and refuse to reuse an existing run. The launcher builds the architecture from YAML and passes the recorded arguments; `pretrained=True` in the original arguments does not identify an external pretrained checkpoint. No COCO initialization weights are loaded by this YAML construction.

The formal main settings are full-image input 832, 100 epochs, batch 6, CIoU box loss, deterministic execution, linear learning-rate schedule (`cos_lr=False`, `lrf=0.01`), base learning rate 0.01, momentum 0.937, configured weight decay 0.0005, three warm-up epochs, mosaic 0.6 disabled over the final 20 epochs, translation 0.05, scale augmentation 0.3 and horizontal flip 0.5. The compatible runtime implements the original MuSGD optimizer and the early `optimizer=auto` selection. Training AMP settings are retained in the per-run configurations; FP32 evaluation is specified separately.

`configs/recorded_args/<configuration>/seedN.yaml` records the detection training and validation arguments relevant to the manuscript. Task-unrelated defaults are omitted, and local path fields and run names are normalized for the release; retained scientific parameter values remain as recorded for each run. `configs/recorded_args/provenance.json` lists original source hashes, optimizer, batch and warm-up bias learning rate. Early auto-optimizer runs recorded warm-up bias learning rate 0.1, while explicit MuSGD runs commonly recorded 0.0. The bundled auto-optimizer selection resets the effective warm-up bias learning rate to 0.0; the initial configured value is not its effective runtime value. Historical data-loader worker counts and recovery flags are omitted from this distribution. The native logs and checkpoint training metadata under `results/training_logs/` retain the reported native metrics and relevant training records.

## Resumption

```bash
python scripts/train.py --model gsf_yolo --seed 0 --resume runs/reproduction/gsf_yolo__seed0/weights/last.pt --data datasets/VisDrone2019-DET-corrected/data.yaml --device 0
```

Resumption loads the user's own unstripped `last.pt` before calling `train(resume=True)` so that optimizer state and checkpoint training arguments are restored. The published inference checkpoints have no optimizer state. They support evaluation and inference; they cannot recreate an interrupted historical optimization trajectory. A new full training run with the recorded configuration is a reproducibility experiment, and bitwise agreement across devices or software stacks is not promised.

## Model selection and native scores

The checkpoint is selected by the highest native validation AP50:95. AP50 is read from the same selected epoch. These selected-epoch native scores differ in provenance from a fresh validation run and from the supplementary COCO300 scores. Preserve the supplied logs and selection metadata when reproducing manuscript tables. `python scripts/summarize_training_logs.py` reconstructs the CSV-backed native results with the selected epochs and correct per-configuration training seed counts. External-framework JSONL logs require their own metric parser.

## Inference and native validation

```bash
python scripts/infer.py --weights weights/gsf_yolo/seed0.pt --source datasets/VisDrone2019-DET-corrected/val/images --output outputs/inference/gsf_seed0
python scripts/evaluate_native.py --weights weights/gsf_yolo/seed0.pt --data datasets/VisDrone2019-DET-corrected/data.yaml --split val --output outputs/native/gsf_seed0
```

Inference saves original-coordinate COCO detections and an image map. Native validation saves its own metrics and protocol label. Both preserve the model's native end-to-end head and use the bundled runtime.

## Balanced latency

```bash
python scripts/benchmark_latency.py --weights weights/p2_baseline/seed0.pt weights/p2_baseline/seed1.pt weights/p2_baseline/seed2.pt weights/gsf_yolo/seed0.pt weights/gsf_yolo/seed1.pt weights/gsf_yolo/seed2.pt --images datasets/VisDrone2019-DET-corrected/val/images --output outputs/latency/baseline_gsf
```

The benchmark requires an idle CUDA GPU, 548 sorted validation images, 50 warm-up predictions per checkpoint/round and three technical rounds. Every supplied checkpoint is measured in the same session. Checkpoint order is forward in round one, reversed in round two and rotated in round three. GPU telemetry and CPU/RAM sampling occur outside the synchronized timed interval. Concurrent GPU compute causes the run to stop; explicitly permitted idle desktop contexts are recorded via `--allow-ambient-pid`.

Raw per-image times and each technical round are retained. Technical-round means are first averaged per checkpoint. Configuration means and sample SDs then use the training-checkpoint means, so technical repeats do not inflate the training seed count. FPS uses the unrounded mean latency. The entry point supports bundled Ultralytics, RTMDet and RT-DETRv2 checkpoints through `--framework auto`. External models retain their native preprocessing and postprocessing as in the original measurements; see `EXTERNAL_MODELS.md` for their recipe differences and dependencies.
