# CPU runtime qualification — 2026-10-04

The exact Kaggle2 CPU-only kernel 136999674 version 1 completed package import
qualification under isolated Python 3.12.14 despite a Python 3.13 system image.
The imported numpy 1.26.4, torch 2.4.1+cu121, torch-geometric 2.6.1, ogb 1.3.6
and rdkit 2025.9.5 matched the frozen workload. Source identity matched the
corrected pre-execution release. Runtime wall cost was 62.4920608997345 seconds;
device cost was not applicable, CPU resource-hours and queue time unmeasured.

This accepted NO_TRAIN infrastructure evidence did not instantiate a model,
open graph roles or execute inference. It did not establish GPU calibration,
numerical reproducibility, a causal claim or a training Replay pair.

It justified releasing the already planned unchanged frozen GPU audit v2,
with the original reproduction barrier and cap retained. No training successor
was justified by package import success alone.

Retrieval exposed an operational defect: environment/cache files were included
in worker outputs. Initial bulk/recursive listing was stopped locally, with
partial files preserved. The installed SDK's version-specific exact-file
download endpoint then retained only the two required JSON artifacts. This
affected local retrieval, not the remote qualification or its scientific scope.
