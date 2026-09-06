# Authorized Audit and Matched Scratch Control (2026-09-06)

## P0: numerical audit

Reuse the already accepted full geometry cache without reconstruction or a
second exhaustive graph acceptance. Verify the immutable acceptance hash and
each consumed shard hash. Replay all 73,545 official validation rows with the
unchanged epoch-30 source and the weight-mapped candidate, both in FP32 and
FP16, with TF32 disabled. Compare identities and labels with the retained
source predictions; no validation fitting or hyperparameter search occurs.

Write independently resumable prediction parts. Compare source/candidate FP32
at maximum 2e-5 eV, and source FP16 versus retained predictions at mean 5e-4
eV. Differences in historical batch shape/TF32 are documented, not concealed.
Numerical probes use four fixed 192-row official-train batches, disposable
model copies, identical RNG and AdamW settings. Record loss, unscaled gradient
norm, finite state, parameter update, and FP16/FP32 gradient difference.
FP32 must be finite. FP16 is diagnostic, not a selected training mode.

Record actual checkpoint dropout and candidate module dropout. Eval-mode
initial equality alone cannot prove that the fine-tuning training function
preserved dropout or numerical precision. No scientific cause is established
by this audit alone.

## P1: matched scratch control

P0 must pass before a GPU training successor is submitted. Freeze two arms:
`ogb_sparse_triangle_edge_state_gps9` and
`ogb_distance_angle_triangle_edge_state_gps9`. Both retain the accepted 100K
factory configuration, including dropout 0.10, and start from random weights.
Copy the common randomly initialized tensors into the geometry arm so initial
functions match. No old checkpoint enters either training initialization.

Both use official train (3,378,606 rows), seed 42, FP32 with TF32 disabled,
batch 192, AdamW lr 4e-4, weight decay 1e-5, gradient clipping 1, cosine decay
over exactly 12 epochs to 1e-6, no early stopping, and the same epoch/shard
data-order seeds. The 12-epoch horizon is fixed before launch and not extended
by resetting the scheduler. A later longer horizon would be a new contract.
Each arm must pass a measured 12-hour projection on A100; a timing failure
does not authorize reducing epochs or changing precision automatically.

Report validation curves and best and final epoch for both arms. Historical
EdgeState 0.099638 eV is a secondary reference with a different training
horizon. Matched improvement alone does not establish superiority to that
reference or convergence. Retain epoch-zero state separately from selected
trained checkpoints. Select only on official validation; test roles remain
unopened. This control does not authorize leaderboard submission.

Persist atomic shard-boundary last checkpoints with model/optimizer/scheduler,
RNG, next-shard cursor, loss counters and exact configuration/hash identities.
Use separate per-arm directories and validation parts. Run no DataLoader
subprocesses to avoid the unresolved historical host-memory cgroup issue.
CPU acceptance is reused; heavy execution stays in scheduled jobs below the
authorized IMS root. P2 ring hierarchy remains owned by the server branch.
