# GSF-YOLO

Code and reproducibility records for **Gated Semantic Fusion and High-Resolution
Capacity Allocation for UAV Small-Object Detection**.

GSF-YOLO combines high-resolution P2 detail features with aligned P3–P5 semantic
features through a spatial–channel gate and an additive semantic residual.
A separate C3k2 block refines the fused P2 feature. Joint P2/P3 widening changes
the prediction widths from (32, 64, 128, 256) to (40, 80, 128, 256), while
retaining the four detection strides and their spatial grids.
Widening also propagates through connected transformations and the
detection-head classification widths.

This repository covers the experiments and analyses in the manuscript's main
text and two included appendices. Result-to-manuscript mappings are provided
in [paper_results.json](manifests/paper_results.json).

## Installation

```bash
git clone https://github.com/BunnyAssassins/GSF-YOLO.git
cd GSF-YOLO
python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

The reported latency measurements used Python 3.10.19, PyTorch 2.9.1, CUDA 13.0,
OpenCV 4.9.0, a modified Ultralytics 8.4.22 runtime, and an NVIDIA GeForce
RTX 5060 Ti. [requirements-lock.txt](requirements-lock.txt) records the release
preparation environment, where OpenCV metadata is 4.10.0; the pip installation
recipe uses 4.10.0.84. Retained runtime files are identified by SHA256 in
[source_snapshot.json](manifests/source_snapshot.json).

See [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for the recorded protocols and
[EXTERNAL_MODELS.md](docs/EXTERNAL_MODELS.md) for RTMDet/RT-DETR dependencies.

## Dataset

Obtain [VisDrone2019-DET](https://github.com/VisDrone/VisDrone-Dataset) from the
dataset maintainers under their use conditions. Images and original
annotations are not redistributed. Arrange the downloaded directories as:

```text
datasets/VisDrone2019-DET/
  VisDrone2019-DET-train/{images,annotations}/
  VisDrone2019-DET-val/{images,annotations}/
  VisDrone2019-DET-test-dev/{images,annotations}/
```

Prepare the corrected ten-class YOLO labels and dataset configuration:

```bash
python scripts/prepare_data.py --raw-root datasets/VisDrone2019-DET --output datasets/VisDrone2019-DET-corrected --require-paper-counts
```

The manuscript uses 6,471 training images, 548 validation images and 1,610
test-dev images. Conversion clips boxes to image boundaries, retains valid
categories 1–10, remaps them to 0–9, and writes normalized labels to six decimal
places. It also records image and label hashes. See
[DATA_PROTOCOL.md](docs/DATA_PROTOCOL.md) for the conversion and ignore rules.

## Training

Preview the seed-specific settings, then train GSF-YOLO:

```bash
python scripts/train.py --model gsf_yolo --seed 0 --dry-run
python scripts/train.py --model gsf_yolo --seed 0 --data datasets/VisDrone2019-DET-corrected/data.yaml --device 0
```

The main ablation settings are input size 832, batch 6, 100 epochs, MuSGD,
base learning rate 0.01, a linear learning-rate schedule, and a three-epoch
warm-up. Each checkpoint is selected by the highest native validation
AP@0.50:0.95; AP@0.50 is taken from the same epoch. The launcher reads the
recorded seed-specific arguments in `configs/recorded_args/`.

| Model argument | Configuration |
|---|---|
| `p2_baseline` | YOLO26-P2 baseline |
| `dsf_refinement` | DSF followed by C3k2 refinement |
| `p2_p3_widening_without_dsf` | P2/P3 widening without DSF |
| `dsf_p2_widening` | DSF and P2 widening |
| `dsf_p3_widening` | DSF and P3 widening |
| `gsf_yolo` | DSF and joint P2/P3 widening |

Run training seeds 0, 1 and 2 separately for each configuration. The five
external comparison models each use seed 0. Training instructions are documented in
[TRAINING_REPRODUCIBILITY.md](docs/TRAINING_REPRODUCIBILITY.md); released
inference checkpoints omit optimizer state.

## Evaluation

Run a new native validation of a released checkpoint:

```bash
python scripts/evaluate_native.py --weights weights/gsf_yolo/seed0.pt --data datasets/VisDrone2019-DET-corrected/data.yaml --split val --device 0 --output outputs/native/gsf_seed0
```

Run the supplementary COCO-style evaluation on validation or test-dev:

```bash
python scripts/evaluate_coco.py --split val --weights weights/gsf_yolo/seed0.pt --device 0 --output outputs/validation/gsf_seed0 --require-paper-count
python scripts/evaluate_coco.py --split test-dev --weights weights/gsf_yolo/seed0.pt --device 0 --output outputs/testdev/gsf_seed0 --require-paper-count
```

The supplementary evaluator uses corrected ground truth, original-image areas,
832 input, batch 1, FP32, confidence 0.001 and `maxDets=[1,10,300]`. It does
not apply official VisDrone ignored-region filtering. Validation Tables 2/5
use selected-epoch native AP/AP50 and supplementary AP_S; test-dev Table 6
uses COCO-style AP/AP50/AP_S. See
[EVALUATION_PROTOCOLS.md](docs/EVALUATION_PROTOCOLS.md) for these distinctions
and the separate threshold diagnostics.

## Latency

Measure all compared baseline and GSF checkpoints together on an idle GPU:

```bash
python scripts/benchmark_latency.py --weights weights/p2_baseline/seed0.pt weights/gsf_yolo/seed0.pt weights/p2_baseline/seed1.pt weights/gsf_yolo/seed1.pt weights/p2_baseline/seed2.pt weights/gsf_yolo/seed2.pt --images datasets/VisDrone2019-DET-corrected/val/images --device 0 --output outputs/latency/baseline_gsf
```

The reported protocol uses 832 input, batch 1, FP32 and GPU synchronization.
Each checkpoint has three technical rounds with 50 warm-ups and 548 measured
images per round. The timed prediction call includes native preprocessing,
transfer, inference and postprocessing; decoding and disk access occur outside
the interval. Checkpoint order is reversed in round two and rotated in round
three. Average technical rounds per checkpoint before calculating the
three-seed mean and sample SD. FPS is 1,000 divided by unrounded latency in ms.

## Reproduce the reported results

```bash
python scripts/summarize_training_logs.py
python analysis/reproduce_tables.py --output reproduced/tables
python analysis/stratified_eval.py --from-saved --output reproduced/stratified.json
python analysis/regional_injection.py --output reproduced/regional_injection.json
python analysis/plot_accuracy_latency.py --output reproduced/accuracy_latency
python analysis/plot_gate_summary.py --output reproduced/gate_summary
```

The retained records support the six-configuration ablation, external
comparison, stratified validation, frozen gate interventions, matched-region
bootstrap analysis and test-dev evaluation. Validation gate diagnostics use
DSF + refinement seeds 0/1; the final-GSF test-dev controls and supplementary
DSF + refinement test-dev controls use seeds 0–2. Frozen controls replace gates
within trained checkpoints. See [ANALYSIS.md](docs/ANALYSIS.md) for the specific
conditions and result inputs.

## Main VisDrone results

The validation comparison follows manuscript Table 5, with one decimal place.
AP/AP50 are selected-epoch native scores; AP_S is supplementary COCO-style AP.
Means and sample SDs use training seeds 0–2.

| Method | Seeds | Native AP@0.50:0.95 (%) | Native AP@0.50 (%) | Supplementary AP_S (%) | Params (M) | GFLOPs | Latency (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| YOLO26-P2 baseline | 0, 1, 2 | 23.5 ± 0.2 | 38.9 ± 0.3 | 13.6 ± 0.2 | 2.5 | 12.7 | 14.5 ± 0.2 |
| GSF-YOLO | 0, 1, 2 | 26.0 ± 0.2 | 43.0 ± 0.5 | 16.9 ± 0.3 | 2.7 | 20.7 | 15.3 ± 0.4 |

Parameters and GFLOPs describe the native unfused dual-head model. Latency
includes the complete prediction call under the protocol above. Hardware,
software and timing scope should accompany any reproduced latency.

The test-dev architecture comparison follows manuscript Table 6(a), with two
decimal places. All three models use training seeds 0–2 and supplementary
COCO-style AP without official ignored-region filtering.

| Method | Seeds | AP@0.50:0.95 (%) | AP@0.50 (%) | AP_S (%) |
|---|---|---:|---:|---:|
| P2 baseline | 0, 1, 2 | 16.56 ± 0.12 | 29.34 ± 0.17 | 8.59 ± 0.12 |
| DSF + P2 widening | 0, 1, 2 | 17.98 ± 0.05 | 32.09 ± 0.15 | 10.63 ± 0.03 |
| GSF-YOLO | 0, 1, 2 | 18.38 ± 0.18 | 32.71 ± 0.37 | 10.87 ± 0.14 |

## Repository layout

```text
configs/       six model YAMLs, seed-specific arguments and external recipes
ultralytics/   compatible detection training and inference runtime
external/      RTMDet/RT-DETR dependencies, source identities and licenses
scripts/       data preparation, training, evaluation, inference and latency
analysis/      table reconstruction, gate, stratified and regional analyses
results/       reported summaries, retained training logs and timing records
manifests/     result mappings, checkpoint/prediction records and SHA256 values
weights/       checkpoint destinations populated from the paper Release
docs/          data, training, evaluation and analysis protocols
```

## Data and weights

The [paper Release](https://github.com/BunnyAssassins/GSF-YOLO/releases/tag/v1.0-paper)
contains 23 checkpoints and 24 distinct prediction archives, plus `SHA256SUMS`.
The checkpoints comprise the six three-seed ablation configurations and five
single-seed external detectors. Final-GSF native-gate predictions also serve
as the corresponding final-model architecture predictions.

Download the Release assets directly to the destinations recorded in
[checkpoints.csv](manifests/checkpoints.csv) and
[predictions.json](manifests/predictions.json), and verify their published
SHA256 values with:

```bash
python scripts/download_assets.py weights
python scripts/download_assets.py predictions
```

These commands first verify existing local files and reuse files with matching
SHA256 values. Missing files are downloaded from the Release URLs and verified
before being placed at their destinations. Filters include
`--models`, `--seeds` and `--dry-run`. Published checkpoints retain the original
inference tensors and remove private serialization metadata. See
[SOURCE_AND_PRIVACY.md](docs/SOURCE_AND_PRIVACY.md).

## Citation

Citation metadata is provided in [CITATION.cff](CITATION.cff). Please cite the
final journal article after publication and acknowledge the underlying
frameworks when using their code.

## License

The GSF-YOLO code and bundled Ultralytics-derived runtime are distributed under
the [AGPL-3.0 License](LICENSE). The external framework subsets retain their
respective [MMDetection](external/mmdetection/LICENSE) and
[RT-DETR](external/rtdetrv2/LICENSE) licenses.
