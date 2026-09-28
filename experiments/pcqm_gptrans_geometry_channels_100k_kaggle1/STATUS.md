# Geometry-channel screen status

Attempt 001 stopped during remote GPU preflight, before either arm trained.
Attempt 002, Kaggle1 kernel
`nothingnessvoid/molgap-geometry-channels-100k-s42-v2` (ID 136152423),
reached `KernelWorkerStatus.COMPLETE`. Both arms passed independent mechanical
acceptance against their retained artifacts. The paired development endpoint
fails the frozen 3.0 meV gain gate: enabling the angle channel worsened MAE by
4.789 meV. See [the attempt-002 decision](decision_attempt_002.md) and
[paired result](results/paired_comparison_attempt_002.json). The
[post-hoc failure analysis](results/failure_analysis_attempt_002.md) separates
observed convergence, EMA lag and molecule-level error patterns.

The canonical RML terminal transaction is complete for all four trajectories:
the first attempt's two arms are `INFRASTRUCTURE_ONLY`, attempt 002's
`distance_only` reference is `CLOSED`, and `distance_angle` is
`NEGATIVE_UNDER_CONTRACT`. The remote trace lacks live-weight development
metrics, and the frozen Spec gives the two training arms different feature
identities. Both 60-epoch traces remain explicitly excluded from replay; the
pair is **not dual replay-ready**. See [RML closure](RML_CLOSURE.md).

Saved-prediction comparison with two accepted pure-2D references gives a
[contextual positive signal for distance only](results/distance_vs_pure2d_context_attempt_002.md).
It does not promote that arm under the frozen angle-increment question or the
V5 strict cross-experiment gate. No successor, protected-role evaluation, or
production change is released by this result. This closed negative experiment
branch was archived at commit `629d0549`. This Desktop copy contains canonical
evidence and pointers without the rejected runner.
