# Reproducing the manuscript

The main text and included gate/test-dev appendices define the retained
experiments. `manifests/paper_results.json` maps each result to source locations,
inputs and commands. Runtime and installation files support those experiments.

Separately trained fusion-only, refinement-only and fixed-sum fusion controls
provide nine complete 100-epoch logs and validation-selected inference
checkpoints. Together with the existing baseline and DSF + refinement records,
they support the component comparison at the original P2 width. These results
use native AP/AP50 only.

Validation ablation and detector comparisons use native AP/AP50 with
supplementary AP_S. Stratified validation uses baseline/GSF overall, size,
category and density COCO AP. Test-dev analyses compare three architectures
and four final-GSF frozen gate conditions. DSF+refinement validation gate
diagnostics and regional analyses use two checkpoints; the supplementary
test-dev analysis uses three-seed DSF+refinement controls and threshold diagnostics.

Keep recorded launch arguments, checkpoint arguments and selected epochs
distinct. External models use different training recipes. Fresh training does not replay an
interrupted optimizer trajectory; released inference weights omit optimizer
state.

Calculate sample SD over the specified training seeds and paired differences
within seeds. Technical timing repetitions do not add training seeds. Keep
unrounded inputs and apply manuscript display precision after aggregation.

Saved predictions permit CPU metric recomputation. The COCO300 and threshold
diagnostic use different ignore treatment and retain separate labels.
Download assets using their published SHA256 values.
