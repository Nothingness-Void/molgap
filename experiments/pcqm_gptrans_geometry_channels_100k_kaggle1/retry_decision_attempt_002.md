# Geometry-channel pair attempt 002 release decision

Attempt 001 stopped during remote preflight because the complete geometry
initial state was reconstructed from runtime-dependent basis tensors. Neither
arm trained. The authoritative failure and its scientific limits are recorded
in [the infrastructure disposition](decision_attempt_001_infrastructure.md).

The user authorized direct resubmission. Release at most one repaired Kaggle1
T4x2 paired attempt on a new immutable source dataset and kernel. Use the
existing GPTrans V4 frozen-state loader to restore the complete geometry state
from artifact SHA256
`c3b49755c30a511a30720cf4845c7ac688f139535382b5bc79ce2500c14b5c97`.
The full initial state remains
`d471924cebde2e0382fe758021a066108acce3858e99c4838451c8c92db9ffba`
for both arms. Core initialization, scientific mechanisms, dataset, roles,
seed, FP32 recipe, exposure, and 3 meV paired gate remain unchanged.

Before push, require a new source commit, package and Spec identities; two
new prospective trajectories with same-run replay binding; exact frozen-state
bytes; source/data mount checks; and the existing real-shard loader preflight.
Both remote T4 workers must pass the runtime certificate and budget gate
before training. Preserve independent checkpoints and terminal evidence.
No protected evaluation role, additional seed, scale-up, or later retry is
authorized by this decision.
