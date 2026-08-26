# Multihop EdgeState Structural GPS R8 validation protocol

## Question

Can persistent edge states use bounded shortest-path virtual edges to improve
the accepted R3 model without adding target-specific heads, prediction fusion,
or three-dimensional information?

R8 follows the failed R7 disposition. It is not a graph-token retry and does
not change the frozen optimizer or validation schedule.

## Frozen architecture

`multihop_edge_state_structural_gps` retains the complete accepted R3
RWSE16, nine-layer, 192-channel, 64-edge-channel EdgeState Structural GPS and
direct three-target head. Its only architectural change is the local edge set:

- every directed atom pair whose unweighted shortest-path distance is one to
  four becomes a local message edge;
- the original four bond channels are retained for bonded pairs and are zero
  for virtual pairs;
- a four-channel one-hot shortest-path code is appended to every edge;
- the resulting eight-channel feature initializes the same persistent edge
  state updated at every GPS depth.

The global GPS attention, RWSE input, pooling, and prediction head are
unchanged. The candidate uses no checkpoint, warm start, target residual,
prediction fusion, ensemble, coordinate, or conformer. PyTorch must measure
exactly 4,739,907 trainable parameters, below the 4,800,000 cap.

## Immutable CPU cache gate

The path expansion is a separate CPU-only job. It may construct only the
30,000 train and 3,000 validation roles. The 3,000-row test role is identified
by the frozen split but is not indexed into the path builder or copied into any
cache part. Each bounded part is written atomically and independently hashed.

GPU validation is forbidden until the downloaded CPU output independently
passes all of the following without importing or executing a model:

- source commit and split fingerprint `01656b1a538f89c8` match;
- exactly 33,000 source indices appear once in frozen train-then-validation
  order;
- every part hash and the aggregate manifest hash match;
- edge width is eight, distance cap is four, and all feature values are finite;
- `test_role_read=false` and no test role is present in the cache roles.

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

R8 passes only if both validation metrics are strictly lower and every cache,
execution, and artifact check passes. A failure closes R8 and permits only a
separately recorded new information-flow architecture, never an R8 retry.
Neither outcome reads or submits QM9 test.

## Resource and durability contract

- One Kaggle2 CPU cache job is permitted before any R8 GPU task.
- After cache acceptance, at most one Kaggle2 GPU validation task is permitted;
  no duplicate or automatic retry.
- Expected CPU prep and GPU validation time are each below 90 minutes.
- Cache progress, bounded parts, acceptance, training progress, checkpoints,
  model, metrics, and train/validation payloads are atomically durable and
  independently retrievable.
- Local checks and acceptance execute no model or inference.
