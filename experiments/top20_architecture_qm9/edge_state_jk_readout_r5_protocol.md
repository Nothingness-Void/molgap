# EdgeState-JK readout R5 validation protocol

## Question

Can the accepted sparse persistent-edge encoder improve its molecule summary by
exposing selected shallow, middle, and final node states together with the final
edge state, without changing its message-passing backbone or direct targets?

This question was authorized after the R3 validation decision. It is distinct
from the untriggered conditional R4 node-attention readout and does not reopen
any dense PairGPS branch.

## Frozen architecture

`edge_state_structural_jk_readout` retains the complete RWSE16, nine-layer,
192-channel, 64-edge-channel `edge_state_structural_gps` encoder and its direct
HOMO/LUMO/Gap head. It mean-pools node states after layers 3, 6, and 9, mean-pools
the final directed-edge state, and feeds their normalized concatenation through
a 32-channel bottleneck. The bottleneck's final projection is initialized to
zero and added to the original final-layer mean representation.

The first forward pass is therefore exactly the accepted R3 winner. The
candidate uses no checkpoint, warm start, target residual, prediction fusion,
ensemble, coordinate, or conformer. PyTorch must measure exactly 4,767,779
trainable parameters, below the 4,800,000 cap.

## Frozen validation contract

- QM9 roles: 30,000 train / 3,000 validation / 3,000 sealed test.
- Split seed 42 and encoder seed 42; split fingerprint `01656b1a538f89c8`.
- Accepted RWSE16 SHA-256:
  `09943454c2e2aae7cf8eeaa828cca79c14be4b920c4c348e0de13a4263331ca5`.
- FP32, batch 48, AdamW, learning rate `4e-4`, weight decay `1e-5`, 20 epochs,
  patience eight.
- Only train and validation may be constructed. The test role must remain
  unread before, during, and after selection.

The frozen comparator is the independently accepted R3 winner:

- validation average MAE `0.10527653247117996 eV`;
- validation Gap MAE `0.1261376142501831 eV`;
- model SHA-256
  `c99883ba5efb247121cce6d83f64d95f5d493e82862ec040eed4a5206c86e186`.

R5 passes only if both validation metrics are strictly lower and all acceptance
checks pass. A failure retains the R3 winner. A pass freezes R5 instead. Neither
outcome reads or automatically submits the one-time QM9 test gate.

## Resource and durability contract

- One Kaggle2 GPU task is permitted; no automatic retry or second candidate.
- Expected GPU time is below 30 minutes with a 90-minute stop bound.
- The accepted RWSE graph cache is mounted read-only; no GPU-side graph building.
- Remote preflight must verify the accepted R3 selection and CPU tensor replay,
  exact parameter count, finite FP32 forward/backward, split/cache identity, and
  no test-role read.
- Training writes atomic progress, checkpoint, selected model, metrics, and
  train/validation payloads to independently retrievable paths.
- Local checks are limited to syntax, AST, manifests, hashes, and policy tests;
  no local model execution is permitted.
