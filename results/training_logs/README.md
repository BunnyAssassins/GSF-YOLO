# Publication training records

Each configuration/seed directory corresponds to a checkpoint in [`../../manifests/checkpoints.csv`](../../manifests/checkpoints.csv). The directories contain native training logs and the selected validation epoch used for publication. The main six configurations and three fusion/refinement controls have three training seeds each; the five external configurations each have one seed.

For Ultralytics runs:

- `results.csv` retains per-epoch native AP50:95/AP50 and technical loss/learning-rate records. Unreported overall precision and recall columns are omitted.
- `configured_args.yaml` records the run's saved configured arguments.
- `checkpoint_train_args.yaml` records the arguments saved in the selected checkpoint.
- `selection.json` identifies the selected epoch and checkpoint-selection rule.

Configured and checkpoint-stored arguments are separate historical records. Their differences are retained. Automatic optimizer settings are preserved as recorded and should not be interpreted as evidence that all native frameworks used the same optimizer or recipe.

For RTMDet, `scalars.jsonl` retains native AP50:95/AP50 and technical training records; `training_config.txt` records the resolved native configuration. Unreported native AP75, size AP and category-precision fields are omitted. The training checkpoint's message hub and optimizer state are omitted from the public inference weight.

For RT-DETRv2, `epochs.jsonl` retains native JSON epoch records. Its `test_coco_eval_bbox` array contains the first two native metrics, AP50:95 and AP50; unreported size/AP75/AR entries are omitted. The native `epoch` field is zero-based; the checkpoint manifest's `selected_epoch_one_based` is one-based.

Paths and run identifiers are normalized for publication. Retained loss, learning-rate and reported native detection metric values preserve their original values. Selected epochs are recorded in `selection.json` and the checkpoint manifest. Saved configuration files describe historical runs; the portable training entry point and model/data configurations elsewhere in the repository should be used for a new run.
