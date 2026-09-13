# OGB-rich EdgeState Structural GPS9 — PCQM-100K V4 reference

## Question

Freeze a V4 100K reference for the model architecture already submitted to the
OGB PCQM4Mv2 leaderboard. This is a pure-2D Gap run; it does not use the
official validation or either test role.

## Architecture

- `ogb_edge_state_structural_gps9`, 4,771,073 parameters.
- OGB nine-category atom and three-category bond encoders; hidden width 192,
  nine GPS layers, four heads, dropout 0.1, RWSE16, persistent EdgeState64,
  mean pooling, direct scalar Gap output.
- Geometry attributes in the fixed graph payload are removed before batching.

## Frozen V4 contract

- Accepted fixed dataset manifest SHA-256
  `1b0e8fd579ab1cb86c02e833e7ad284b4af7582b059f912a77853fdccf3ede6d`;
  geometry aggregate SHA-256
  `bc83a4bd9a7fd7fd6fc6fd085d78ba3caab0e40f997d1932e0f344e701dd40c5`.
- Official-train-derived source rows 0:100,000 for training and 100,000:150,000
  for development. All sealed roles remain unread.
- Seed 42; strict deterministic FP32, TF32 disabled; one visible accelerator,
  physical batch 128, no accumulation, `drop_last`.
- 60 epochs; 781 steps and 99,968 presentations per epoch (5,998,080 total).
- AdamW (`foreach=False`), LR 1e-3, weight decay 0.05, four-epoch linear
  warmup, cosine decay to 1e-6, gradient clip 1.0, EMA 0.9999.
- Normalized Gap L1; training mean and sample standard deviation; minimum
  development MAE of the EMA model selects the checkpoint.
- Repeated real optimizer-step calibration must be bitwise deterministic and
  reserve at least 15% accelerator memory before training begins.
- Atomic checkpoint at least every 200 optimizer steps and at each epoch
  boundary. Checkpoint includes model, optimizer, scheduler, EMA, RNG, cursor,
  contract, source, data, and runtime identities.

The feature fingerprint records the actual GPS inputs (OGB atom9/bond3 and
RWSE16, no geometry). Results with a different feature fingerprint are not
claimed as direct V4 architecture comparisons, even though they share the same
dataset and role boundaries.

## Acceptance

The mechanical acceptance checks the exact model identity and parameter count,
60 completed epochs, optimizer exposure, fixed data/source/runtime hashes,
development prediction row order and recomputed MAE, finite model tensors, and
the sealed-role flags. It does not execute model inference or inspect any
official validation/test artifact.
