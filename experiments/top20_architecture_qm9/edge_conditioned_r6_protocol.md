# Edge-conditioned Structural GPS R6 validation protocol

## Question

Can persistent bond states improve the accepted EdgeState encoder when they
condition atom representations before every GPS block, rather than being
re-pooled after the complete graph representation as in failed R5?

This is a new architecture hypothesis authorized after the R5 disposition. It
does not rerun R5 or reopen PairGPS, target residuals, coordinates, or 3D.

## Frozen architecture

`edge_conditioned_structural_gps` retains the complete RWSE16, nine-layer,
192-channel, 64-edge-channel EdgeState Structural GPS and direct
HOMO/LUMO/Gap head. After each persistent edge update, directed edge states are
mean-aggregated onto their target atoms. One shared FiLM transform maps the
64-channel atom-local bond context to scale and shift vectors for the
corresponding 192-channel atom state before the GPS block.

The shared transform is zero-initialized, so the first forward pass exactly
matches accepted R3. Unlike R5, R6 preserves atom correspondence, omits the
duplicated final node representation, and exposes current bond context to both
the local convolution and global attention branches. It uses no checkpoint,
warm start, target residual, prediction fusion, ensemble, coordinate, or
conformer. PyTorch must measure exactly 4,765,123 trainable parameters, below
the 4,800,000 cap.

## Frozen validation contract

- QM9 roles: 30,000 train / 3,000 validation / 3,000 sealed test.
- Split seed 42 and encoder seed 42; split fingerprint `01656b1a538f89c8`.
- Accepted RWSE16 SHA-256:
  `09943454c2e2aae7cf8eeaa828cca79c14be4b920c4c348e0de13a4263331ca5`.
- FP32, batch 48, AdamW, learning rate `4e-4`, weight decay `1e-5`, 20 epochs,
  patience eight.
- Only train and validation may be constructed. The test role remains unread.

The frozen comparator is accepted R3:

- validation average MAE `0.10527653247117996 eV`;
- validation Gap MAE `0.1261376142501831 eV`;
- model SHA-256
  `c99883ba5efb247121cce6d83f64d95f5d493e82862ec040eed4a5206c86e186`.

R6 passes only if both validation metrics are strictly lower and every
acceptance check passes. A failure closes R6 and may open a separately recorded
new architecture hypothesis; it never authorizes a repeat of R6. A pass freezes
R6 as the validation winner. Neither outcome reads or submits QM9 test.

## Resource and durability contract

- One Kaggle2 GPU task is permitted for R6; no duplicate or automatic retry.
- Expected GPU time is below 30 minutes with a 90-minute stop bound.
- The accepted RWSE graph cache is mounted read-only; no GPU graph building.
- Remote preflight verifies the R3 anchor, exact parameter count, finite FP32
  forward/backward, split/cache identity, and no test-role read.
- Training writes atomic progress, checkpoint, selected model, metrics, and
  train/validation payloads to independently retrievable paths.
- Local checks execute syntax, AST, manifest, hash, and policy validation only;
  no local model execution is permitted.
