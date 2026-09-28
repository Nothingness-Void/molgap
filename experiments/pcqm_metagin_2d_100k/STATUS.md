# MetaGIN2D operational status

The prospective V5 prelaunch and private executable source are frozen. The
CPU-only sidecar kernel `kaseichou/molgap-metagin-2d-sidecar-v1:v1` completed;
its full 150,000-row recomputation acceptance and local shard hashes are in
[`results/cpu_sidecar_acceptance.json`](results/cpu_sidecar_acceptance.json).
The private accepted sidecar dataset is
`kaseichou/molgap-metagin-2d-hop-cache-v1` (version 1, ready).

The seed-42 GPU attempt v1 was allocated Tesla T4 x2 despite the P100
request. It used only device 0 and stopped before epoch 0 because its cold
first-step extrapolation exceeded the six-hour gate. See the
[failure record](results/v1_preflight_failure.md). This is an infrastructure/
cost-screen outcome, not a scientific loss or a completed V5 comparison.

A short train-role-only [runtime diagnosis](runtime_profile_protocol.md) was
attempted in profile v1. It stopped before optimizer timing because the
consumer incorrectly required its executable-source commit to equal the
already accepted CPU sidecar's producer commit. The source and sidecar are
independently immutable assets; profile v2 pins both identities separately.
See the [profile failure record](results/profile_v1_failure.md). No MetaGIN2D
validation result or full-scale claim exists.

The exact Kaggle2 profile v2 (`kaseichou/molgap-metagin-2d-runtime-profile:v2`)
completed and passed its [no-inference acceptance](results/profile_v2_acceptance.json).
The measured time and VRAM [decision](results/profile_v2_decision.md) releases
one new run identity of the *same* scientific screen. Its
[v2 V5 prelaunch/RML plan](attempt_v2/rml_plan/trajectory.json) passed
repository validation; the private source-v4 dataset was ready before the
single GPU submission. Kaggle returned `kaseichou/molgap-metagin-2d-s42:v2`
(numeric kernel 136263563), verified `RUNNING` at submission and bound to
the existing Luna B monitor. The [receipt](attempt_v2/submission_receipt.json)
fixes the physical version, source and sidecar. Kaggle subsequently reported
`COMPLETE`; the exact v2 outputs were retained under the ignored local
`platforms/_records/kaggle/training/metagin_2d_s42_v2/` directory. The
[no-inference acceptance](attempt_v2/results/acceptance.json) passed, the
[strict comparison](attempt_v2/results/comparison_readiness.json) passed, and
the [RML terminal](attempt_v2/rml_plan/rml_finalized/finalization.json) entered
the replay pool. The [dated decision](attempt_v2/decision.md) closes the
scientific question under this adaptation; no further compute is released.
