# Geometry-channel terminal RML closure

All four prospective trajectories have immutable `rml_finalized/` records.
Attempt 001 stopped at remote preflight before contracted training; each arm
closed as `INFRASTRUCTURE_ONLY` with no training trace or replay entry.
Attempt 002 completed 60 epochs independently per arm and passed mechanical
acceptance. `molgap.experiment_cli terminal --execute` closed the same-job
reference first, then the angle candidate, using the frozen Spec and the
[terminal descriptor](terminal_descriptor_attempt_002.json).

| Attempt | Arm | Terminal outcome | Replay status |
| --- | --- | --- | --- |
| 001 | `distance_only` | `INFRASTRUCTURE_ONLY` | no training trace |
| 001 | `distance_angle` | `INFRASTRUCTURE_ONLY` | no training trace |
| 002 | `distance_only` | `CLOSED` | excluded: remote live-development metric unavailable |
| 002 | `distance_angle` | `NEGATIVE_UNDER_CONTRACT` | excluded: remote live-development metric unavailable and frozen feature identities differ |

The attempt-002 canonical traces contain only observed per-epoch live training
and EMA development MAE. Cumulative optimizer steps and sample presentations
are sums of the remote trace's 60 measured per-epoch increments, ending at
46,860 and 5,998,080 respectively. Only the actual final checkpoint is named;
no per-epoch checkpoint or live-development metric was inferred. Measured
single-T4 training and development evaluation intervals are 2.759468 and
2.866277 T4-device-hours. Setup, CPU and queue costs remain unmeasured.

The RML `replay_pool.json` contains **zero geometry-channel entries**; its
exclusions state the blockers. `research_memory check --frozen` passed after
terminal closure. The final decision remains the negative angle-increment
[attempt-002 decision](decision_attempt_002.md). The distance-only outcome is
a completed same-job reference, and its cross-job pure-2D comparison is
[contextual](results/distance_vs_pure2d_context_attempt_002.md), not a new
frozen promotion decision.
