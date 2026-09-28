# Geometry-channel screen status

Attempt 001 stopped during remote GPU preflight, before either arm trained.
Attempt 002, Kaggle1 kernel
`nothingnessvoid/molgap-geometry-channels-100k-s42-v2` (ID 136152423),
reached `KernelWorkerStatus.COMPLETE`. Both arms passed independent mechanical
acceptance against their retained artifacts. The paired development endpoint
fails the frozen 3.0 meV gain gate: enabling the angle channel worsened MAE by
4.789 meV. See [the attempt-002 decision](decision_attempt_002.md) and
[paired result](results/paired_comparison_attempt_002.json).

The remote trace does not contain the live-weight development metric required
by strict V5 mechanism comparison, and the frozen Spec gives the two arms
different feature identities. The pair is **not dual replay-ready**. Canonical
RML terminal closure and replay-pool publication remain pending; do not label
either arm replay-ready or release a successor on the strength of this run.
