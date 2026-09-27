# Attempt 002 terminal decision

Kaggle1 reports `KernelWorkerStatus.COMPLETE` for the frozen paired kernel
`nothingnessvoid/molgap-gptrans-rwse-local-bias-paired-100k-s42-v2`. The
remote source archive, source commit, package identity, Spec identity, and both
mounted dataset identities match the attempt-002 release record.

Both `rwse16` (A) and `rwse16_local_edge` (B) passed independent mechanical
acceptance. Each has accepted native Tesla T4 runtime qualification, the full
60-epoch trace (46,860 optimizer steps and 5,998,080 sample presentations),
selected and resumable checkpoints, finite aligned predictions for the same
50,000 development rows, and observed training/development role use. The
official validation, test-dev, and test-challenge roles remained sealed.

On the 50,000 aligned development rows, A's MAE was 0.1541538451 eV and B's
was 0.1530722509 eV, a gain of 0.0010815942 eV. With 10,000 paired row
bootstrap replicates and seed 42, the 95% interval for B−A was
[-0.0021281876, -0.0000292677] eV. The sign condition passed, but the gain was
below the frozen 0.003 eV minimum. B therefore fails the frozen shortlist gate
and the outcome is `NEGATIVE_UNDER_CONTRACT`. This closes the local-edge
increment under this contract. No follow-on attempt, scale-up, protected-role
evaluation, or production promotion is authorized by this result.

Observed training cost, computed as the sum of the 60 recorded per-epoch
training/evaluation wall intervals multiplied by one assigned Tesla T4, was
3.169152 T4-device-hours for A and 4.450462 T4-device-hours for B. These
measurements exclude setup and queue time; CPU and queue costs are unknown.

Mechanical acceptance is complete. The canonical RML terminal and replay-pool
transaction is not yet recorded, so this pair is not being reported as
dual replay-ready. A paired replay claim requires two distinct complete pool
entries with empty exclusion lists.
