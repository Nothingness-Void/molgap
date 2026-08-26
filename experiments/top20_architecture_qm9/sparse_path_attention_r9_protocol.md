# Sparse path-attention EdgeState R9 validation protocol

## Question

Can shortest-path information improve accepted R3 when it biases a bounded
sparse attention branch instead of replacing real-bond local message edges?

R9 follows the failed R8 disposition. It reuses the independently accepted
path cache but is not a distance-cap or optimization retry.

## Frozen architecture

`sparse_path_attention_structural_gps` retains the complete accepted R3
RWSE16, nine-layer, 192-channel, 64-edge-channel EdgeState Structural GPS,
real-bond local edge updates, global GPS attention, mean pooling, and direct
three-target head.

Its only new mechanism is one shared low-rank sparse attention module applied
after each GPS block:

- queries attend to source atoms connected by cached shortest paths of length
  one to four;
- a learned scalar embedding of path length is added to each attention logit;
- query, key, and value width is 16, shared across all nine depths;
- the output projection is zero-initialized, preserving the exact R3 forward
  path at initialization;
- multihop pairs never enter the local EdgeState convolution, so real-bond
  semantics remain unchanged.

It uses no checkpoint, warm start, target residual, prediction fusion,
ensemble, coordinate, or conformer. The expected parameter count is 4,752,327,
below the 4,800,000 cap.

## Frozen input and validation contract

- QM9 roles: 30,000 train / 3,000 validation / 3,000 sealed test.
- Split seed 42 and encoder seed 42; split fingerprint `01656b1a538f89c8`.
- Accepted RWSE16 SHA-256:
  `09943454c2e2aae7cf8eeaa828cca79c14be4b920c4c348e0de13a4263331ca5`.
- Accepted path-cache aggregate SHA-256:
  `0ea8a0e27790b5bbdb038365d681b5f48974da959a1a8890e0ca1ef24a339dd3`.
- FP32, batch 48, AdamW, learning rate `4e-4`, weight decay `1e-5`, 20 epochs,
  patience eight.
- Only train and validation may be constructed. The test role remains unread.

The frozen R3 comparator is validation average MAE
`0.10527653247117996 eV`, validation Gap MAE `0.1261376142501831 eV`, and model
SHA-256 `c99883ba5efb247121cce6d83f64d95f5d493e82862ec040eed4a5206c86e186`.

R9 passes only if both validation metrics are strictly lower and all input,
execution, and artifact checks pass. A failure closes R9 and permits only a
separately recorded new information-flow architecture, never a rank, path-cap,
seed, or training-schedule retry. Neither outcome reads or submits QM9 test.

## Resource and durability contract

- One Kaggle2 GPU validation task is permitted; no duplicate or automatic
  retry. The accepted R8 CPU cache is reused read-only.
- Expected GPU time is below 45 minutes with a 90-minute stop bound.
- Remote preflight verifies R3, cache identities, parameter count, finite FP32
  forward/backward, real-bond local edges, and no test-role read.
- Atomic progress, checkpoint, model, metrics, and train/validation payloads are
  independently retrievable.
- Local checks and acceptance execute no model or inference.
