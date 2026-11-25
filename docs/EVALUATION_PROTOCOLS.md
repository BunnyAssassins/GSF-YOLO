# Evaluation used in the manuscript

## Native validation

Tables 2 and 5 use native AP50:95 and AP50 from the same checkpoint-selected
epoch. `scripts/summarize_training_logs.py` restores these values from the
retained logs. Baseline/GSF have three training seeds in Table 5; the five
external configurations have seed 0.

## Supplementary COCO300

Corrected six-decimal YOLO labels are reconstructed in original-image
coordinates. Category IDs are 0–9. COCO bbox IoU thresholds are
0.50:0.05:0.95 and maxDets=[1,10,300]. Original-image area defines small,
medium and large ranges. Predictions do not undergo official ignored-region
filtering in this AP calculation.

Tables 2/5 use supplementary AP_S. Table 4 additionally reports overall,
size, category and density AP for baseline/GSF. Table 6 uses test-dev
AP/AP50/AP_S. The requested precision-tensor slices are evaluated directly;
native and supplementary AP retain distinct labels.

Prediction uses 832 input, batch 1, FP32, confidence 0.001, at most 300 outputs,
and native preprocessing/head/postprocessing. External native recipes are
specified in `docs/EXTERNAL_MODELS.md`.

## Threshold diagnostics

Table 3, Figure 5 and Appendix B use confidence greater than 0.25 and descending
score order. A prediction matches the highest-IoU unmatched GT of its category
when IoU is at least 0.5. Valid-GT matching precedes ignore filtering. An
unmatched prediction is ignored when the custom ignored mask covers more than
half its area. Remaining unmatched predictions are false positives; background
false positives have maximum IoU below 0.1 with all valid GT.

The outputs are small-object recall, FP/image and background FP/image for the
manuscript's specified models/seeds/conditions. Paired-image bootstrap follows
the fixed-checkpoint sampling rules in the included appendices.

## Prediction-call timing

Figure 4 and Table 5 use serial native prediction on an otherwise idle RTX
5060 Ti at 832 input, batch 1 and FP32. Each checkpoint has three technical
rounds, with 50 warm-ups and 548 images per round. GPU-synchronized timing
includes preprocessing, transfer, inference and postprocessing. Decoding, disk
access, drawing, telemetry and serialization are outside the timed interval.

Average technical rounds per checkpoint, then calculate the training-checkpoint
mean and sample SD. FPS is 1000 divided by unrounded latency in milliseconds.
