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
