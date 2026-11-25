# Analyses reported in the manuscript

The release covers the manuscript's active main sections and two included appendices. `manifests/paper_results.json` maps each retained result to its source label and text anchor. Archived checkpoint hashes identify original experimental weights; `manifests/checkpoints.csv` maps them to metadata-sanitized release weights.

## Recompute reported tables and plots

```bash
python analysis/reproduce_tables.py --output reproduced/tables
python analysis/stratified_eval.py --from-saved --output reproduced/stratified.json
python analysis/regional_injection.py --output reproduced/regional_injection.json
python analysis/plot_accuracy_latency.py --output reproduced/accuracy_latency
python analysis/plot_gate_summary.py --output reproduced/gate_summary
```

The result reconstruction command reproduces the six-configuration ablation, five-configuration fusion/refinement comparison, seven-model external comparison, three-model test-dev comparison and final-GSF gate interventions, two-seed validation gate comparison, and three-seed supplementary test-dev controls. Native validation AP/AP50 and supplementary COCO AP_S have separate sources. Only baseline and final GSF have three training seeds in the external comparison; the other five models have seed 0.

The component comparison uses the existing baseline and DSF + refinement records plus nine completed runs of fusion-only, refinement-only and fixed-sum fusion, all at P2 width 32. Each run contributes the native AP/AP50 from its validation-selected epoch. `results/summary/fusion_component_statistics.json` records means, sample SDs and paired contrasts without assigning table numbers. The mean gated-minus-fixed-sum AP difference is 0.153 percentage points, comparable in magnitude to training-seed variation. No supplementary AP_S or latency result is reported for these three controls.

Stratified validation reproduces the reported overall AP, size AP_S/AP_M/AP_L, density AP/AP_S and ten class AP values. Density cutpoints use linear quantiles at 1/3 and 2/3 of valid targets per original-image megapixel over 6,471 training images. The 548 validation images form groups of 57, 77 and 414. Training-density inputs and validation image lists are in `results/stratified/`.

Regional reproduction recalculates nine size-by-metric ROI contrasts: mean gate, mean absolute injected residual and injection/detail ratio (`eq:regional_injection_statistics`). It uses 2,000 paired image-bootstrap draws over the same two DSF+refinement checkpoints. Object/background mean quantities and geometric pair identities are retained in `results/regional/seed0` and `seed1`. The matched GT–background samples comprise 10,328/4,979/600 pairs from 524/526/268 images. The plotting command redraws `fig:gate_analysis` and the included `fig:gate_injection_ratio` from reported values.

## Saved prediction assets

`manifests/predictions.json` gives 24 unique archives with Release URLs and compressed/uncompressed SHA256 hashes. Download them into `assets/predictions/`. They cover baseline/GSF validation seeds 0--2, baseline/DSF+P2 widening/GSF test-dev seeds 0--2, and three additional frozen-GSF conditions per test-dev seed. Native final-GSF gate rows reuse the same per-seed architecture predictions because archived bytes are identical.

Predictions use original-image xywh boxes, supplied image IDs and model categories 0--9. The protocol uses 832 input, batch 1, FP32, confidence 0.001 and up to 300 outputs/image. Supplementary AP uses IoU 0.50:0.05:0.95, maxDets=[1,10,300], corrected original-image GT and no official prediction-level ignored-region filtering.

```bash
python analysis/stratified_eval.py --ground-truth data/val_coco.json --prediction-dir assets/predictions --output reproduced/stratified_from_predictions.json
```

## Frozen gate interventions

```bash
python analysis/gate_interventions.py --weights weights/gsf_yolo/seed0.pt --images data/VisDrone2019-DET-test-dev/images --image-map data/testdev_coco.json --output reproduced/gate_seed0 --device 0
```

Run each seed separately. The final-model test-dev analysis uses native, valid spatial mean, valid shuffle and L1-matched shuffle conditions. The algorithm uses CPU generator seed `1000003*image_id`, a shared permutation across channels, unchanged letterbox padding and one scalar matching the injected residual G*S over valid cells (`eq:gate_l1_matching`). Replay and L1 checks are saved.

The DSF+refinement validation gate comparison uses seeds 0 and 1 with native, zero/unit and the three valid-region interventions. Constant controls and full-canvas technical draws are defined in `app:gate_interventions`; `--include-constant --include-full-canvas` applies only to that 32-channel configuration. Full-canvas shuffle seeds are `1000003*image_id+r` for r=0,1,2. Their algorithm is included without publishing additional numerical results absent from the manuscript. Supplementary test-dev DSF+refinement uses seeds 0--2 and four native/valid-region conditions. All interventions freeze model parameters.

## Threshold diagnostics reported in the paper

```bash
python analysis/threshold_diagnostics.py --ground-truth data/testdev_coco.json --raw-annotations data/VisDrone2019-DET-test-dev/annotations --predictions assets/predictions/testdev_gsf_yolo_seed0.json.gz --output reproduced/diagnostics_seed0
```

The DSF+refinement validation and supplementary test-dev diagnostics report small-object recall, FP/image and background FP/image at score>0.25. Score-ordered same-class one-to-one matching uses IoU>=0.5, before ignore filtering. Unmatched predictions are ignored when category0/11 or score0 regions cover over half their area. Background FP has maximum IoU<0.1 with valid GT. Per-seed small-recall bootstrap intervals for GSF minus baseline are in `results/summary/final_model_testdev_statistics.json`.

## Reported computational cost

`results/timing/raw/` has 23 checkpoints × three technical rounds ×548 pipeline measurements, with 50 warm-ups/round, supporting the detector comparison and small-object AP-versus-latency analysis. Pipeline latency includes preprocessing, transfer, inference and postprocessing with synchronization; decoding and disk access are excluded. Average rounds within checkpoint first, then weight checkpoint means equally. Sample SD is across training checkpoint means; technical repeats do not create training seeds.

Input settings, GPU model, driver, clocks, power and temperature are retained as measurement conditions. Hardware identifiers, process identifiers, private paths and unrelated applications are excluded. Parameter/GFLOPs values refer to the paper's unfused dual-head graph.
