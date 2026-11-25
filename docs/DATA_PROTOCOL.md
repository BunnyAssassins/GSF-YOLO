# VisDrone data preparation

Obtain VisDrone2019-DET from the [dataset maintainers](https://github.com/VisDrone/VisDrone-Dataset). Images and original annotations are not redistributed in this repository. Keep the provider's license and use conditions.

Arrange the original archives as follows:

```text
datasets/VisDrone2019-DET/
  VisDrone2019-DET-train/{images,annotations}/
  VisDrone2019-DET-val/{images,annotations}/
  VisDrone2019-DET-test-dev/{images,annotations}/
```

From the repository root, create corrected full-image data:

```bash
python scripts/prepare_data.py --raw-root datasets/VisDrone2019-DET --output datasets/VisDrone2019-DET-corrected --require-paper-counts
```

The default copies images into a new directory. `--images-mode hardlink` explicitly selects hard links on the same filesystem. The source images and annotations are never edited. The output must be new or empty; an interrupted conversion remains incomplete and should not be used as publication evidence.

The paper uses 6,471 train images, 548 validation images, and 1,610 test-dev images. All are full images; there is no image slicing. Image resizing to 832 occurs in the model loader, not during conversion. `--splits val` can prepare a single split for evaluation.

The converter reads eight-column original DET annotations. It excludes annotation score 0 and categories outside 1–10 (including ignored region 0 and category 11). It clips box boundaries to the image, drops nonpositive boxes, remaps categories 1–10 to 0–9, and writes normalized center-x, center-y, width and height to six decimal places. Any width/height that rounds to zero is separately counted and discarded. The class order is pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor.

`data.yaml` contains local dataset paths and train/val/test split entries; `test` means test-dev. `conversion_manifest.json` records each image and original annotation SHA-256, corrected label SHA-256, image dimensions, clipping/drop counts, image counts, and completion status. Plain split manifests retain filename order. These generated local files are runtime provenance and are not automatically added to Git.

For the provided example configuration, set `path` in `configs/data/visdrone.example.yaml` to the converted dataset, or use the generated `datasets/VisDrone2019-DET-corrected/data.yaml` directly. Relative paths in the public scripts are resolved from the repository root. Keep original annotations separately: threshold-based diagnostic ignore masks require them.

Validation selects checkpoints and supports analysis. The test-dev evaluation uses fixed selected checkpoints with no test-dev tuning. These roles do not establish that training, validation and test-dev scenes are statistically independent.
