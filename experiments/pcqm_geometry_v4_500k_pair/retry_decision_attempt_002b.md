# Continuation startup repair, 2026-09-29

Kaggle1 accepted the two-dataset, T4x2 submission and recorded both sources in
the pulled kernel metadata, but the first continuation version ended `ERROR`
at `prepare_resume()` before source installation, preflight or training.
The fixed path `/kaggle/input/molgap-geometry-v4-500k-a1-resume-s42` was not
present. The exact mount directory name was never observed. The remote kernel
listed no output files. No new optimizer steps or T4 training hours are claimed.

The infrastructure-only repair searches `/kaggle/input` for the uniquely named
checkpoint manifest and still verifies all ten pinned file hashes, both arm
identities, source SHA and epoch cursors before releasing training. If the
dataset is absent, it fails closed and reports visible top-level mount names.
The frozen 192-module scientific source archive remains byte-identical at
`8bea0dab3b8b1755352f7f11e7ae7708b87de0731fe32aefabb9725e60112367`
from commit `555775ce8fd14966a3566b0e9ff732725279df9f`; only the outer
Kaggle entrypoint changes. Both original five-epoch checkpoints and the new
16/26-hour limits are unchanged. This is a retry of the already authorized
dual-arm continuation, not a new scientific question.
