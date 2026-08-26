# Directed EdgeState Structural GPS R10 validation protocol

## Question

Can non-backtracking directed bond-to-bond communication improve accepted R3
without virtual edges, target-specific paths, or additional readout logic?

R10 follows the failed R9 disposition and closes the shortest-path branch. It
changes the persistent chemical-bond state update, not the training schedule.

## Frozen architecture

`directed_edge_state_structural_gps` retains the complete accepted R3 RWSE16,
nine-layer, 192-channel, 64-edge-channel EdgeState Structural GPS, dense GPS
attention, mean pooling, and direct three-target head.

At each depth, every directed bond receives the sum of edge states entering its
source atom, excluding the reverse bond. A bias-free 64-to-64 projection adds
this non-backtracking context to the existing endpoint-conditioned persistent
edge update. The projection is zero-initialized, so the first forward pass is
exactly R3. Separate projections are learned at all nine depths.

The candidate uses only real chemical bonds. It has no virtual edge, shortest-
path input, checkpoint, warm start, target residual, prediction fusion,
ensemble, coordinate, or conformer. The expected parameter count is 4,776,515,
below the 4,800,000 cap.

## Frozen validation contract

- QM9 roles: 30,000 train / 3,000 validation / 3,000 sealed test.
- Split seed 42 and encoder seed 42; split fingerprint `01656b1a538f89c8`.
- Accepted RWSE16 SHA-256:
  `09943454c2e2aae7cf8eeaa828cca79c14be4b920c4c348e0de13a4263331ca5`.
- FP32, batch 48, AdamW, learning rate `4e-4`, weight decay `1e-5`, 20 epochs,
  patience eight.
- Only train and validation may be constructed. The test role remains unread.

The frozen R3 comparator is validation average MAE
`0.10527653247117996 eV`, validation Gap MAE `0.1261376142501831 eV`, and model
SHA-256 `c99883ba5efb247121cce6d83f64d95f5d493e82862ec040eed4a5206c86e186`.

R10 passes only if both validation metrics are strictly lower and all input,
execution, and artifact checks pass. A failure closes R10 and permits only a
separately recorded new information-flow architecture, never a width, seed, or
training-schedule retry. Neither outcome reads or submits QM9 test.

## Resource and durability contract

- One Kaggle2 GPU validation task is permitted; no duplicate or automatic
  retry. No new graph cache is needed.
- Expected GPU time is below 35 minutes with a 90-minute stop bound.
- Remote preflight verifies R3 identity, reverse-edge coverage, parameter
  count, finite FP32 forward/backward, split/cache identity, and no test read.
- Atomic progress, checkpoint, model, metrics, and train/validation payloads are
  independently retrievable.
- Local checks and acceptance execute no model or inference.
