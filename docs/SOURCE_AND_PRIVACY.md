# Runtime source and checkpoint identity

The runtime derives from working Ultralytics 8.4.22 code at
https://github.com/ultralytics/ultralytics. It retains the implementation and
dependencies needed by the manuscript's detection architectures: DSF, parsing,
native dual head, loss and trainer. `manifests/source_snapshot.json` identifies
the retained files.

Six model configurations specify fusion, separate C3k2 refinement and connected
P2/P3 allocation. External source identities and licenses accompany their
runtime dependencies.

Published checkpoint copies remove private path/Git metadata and retain original
inference tensors. Original and published file hashes are both recorded in
`manifests/checkpoints.csv`; use published hashes to verify downloads. Private
locations, GPU UUIDs and process identifiers are absent from shared records.
Dataset images are obtained from VisDrone under its provider's terms.
