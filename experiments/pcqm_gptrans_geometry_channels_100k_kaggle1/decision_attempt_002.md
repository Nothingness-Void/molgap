# Attempt 002 paired result

Kaggle1 reports `KernelWorkerStatus.COMPLETE` for the frozen T4x2 kernel
`nothingnessvoid/molgap-geometry-channels-100k-s42-v2` (ID 136152423).
The remote outputs bind source commit `e4be2b828e0022df7e057db2c0ba8f576cad31a8`,
source archive SHA256 `84866ec721d55d163442cb4ae678bda29396df5a51b781d7c8bf68365fc5a4e6`,
and the accepted fixed-100K graph manifest
`1b0e8fd579ab1cb86c02e833e7ad284b4af7582b059f912a77853fdccf3ede6d`.
The [attempt-002 release gate](release_gate_attempt_002.json) owns the frozen
package and Spec identities.

Both arms independently passed the existing GPTrans mechanical acceptance.
Each retained an accepted deterministic FP32 Tesla T4 runtime certificate,
60 ordered epochs, 46,860 optimizer steps, 5,998,080 sample presentations,
a selected model, aligned finite predictions on the same 50,000 internal
development rows, and a resumable epoch-59 checkpoint. The official
validation, test-dev and test-challenge roles remained sealed.

| Arm | Development Gap MAE | Measured training/evaluation cost |
| --- | ---: | ---: |
| `distance_only` | 0.1519791187 eV | 2.759468 T4-device-hours |
| `distance_angle` | 0.1567683146 eV | 2.866277 T4-device-hours |

The angle-minus-distance paired difference is **+0.0047891959 eV** (angle is
worse by 4.789 meV). A 10,000-replicate paired row bootstrap with seed 42
gives a 95% interval of **[+0.0037790547, +0.0057713758] eV**. The frozen
shortlist gate required at least 0.003 eV improvement and an interval upper
bound below zero. The angle increment therefore closes as
`NEGATIVE_UNDER_CONTRACT` at 100K. The row bootstrap does not estimate
training-seed variation. No second seed, scale-up, protected-role evaluation,
or production change follows from this result.

The measured T4 costs sum the remote trace's per-epoch training/evaluation
wall intervals for one assigned T4 per arm. Setup and queue time are excluded;
CPU and queue costs are unknown.

**RML qualification:** Mechanical acceptance and the paired endpoint are
recorded, but strict V5 comparison and dual replay readiness are unavailable.
The retained remote trace has EMA development and live training metrics but
no live-weight development metric for its 60 epochs. The frozen Spec also
records different `feature_sha256` values for the two arms. The present V5
mechanism-comparison rule requires the live development trace field and treats
feature identity as strict, with only architecture configuration declared as
an allowed intervention. Local post-hoc checkpoint evaluation cannot fill a
remote per-epoch T4 trace. The canonical RML terminal transaction has not
been executed, and no replay-pool entry is asserted here. These gaps must be
recorded honestly rather than filled with inferred metrics or a new training
run.
