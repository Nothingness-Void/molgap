# Graph-token Structural GPS R7 validation protocol

## Question

Can a recurrent molecule token improve accepted EdgeState by carrying an
explicit graph-level memory through all nine GPS blocks?

R7 follows the accepted disposition of R6. It is not a retry of R5 readout or
R6 edge conditioning and does not change the frozen training schedule.

## Frozen architecture

`graph_token_structural_gps` retains the complete RWSE16, nine-layer,
192-channel, 64-edge-channel EdgeState Structural GPS and direct three-target
head. It adds one learned 192-channel token per graph. At every depth, a shared
16-channel bottleneck updates the token from the current token and mean node
state; a shared projection broadcasts the token into all nodes of that graph
before the following GPS block.

The two output projections are zero-initialized, so the first forward pass
exactly matches R3. The token then learns recurrent node-to-graph and
graph-to-node communication. It uses no checkpoint, warm start, target
residual, prediction fusion, ensemble, coordinate, or conformer. PyTorch must
measure exactly 4,787,475 trainable parameters, below the 4,800,000 cap.

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

R7 passes only if both validation metrics are strictly lower and all acceptance
checks pass. A failure closes R7 and permits only a separately recorded new
information-flow architecture, never an R7 repeat. Neither outcome reads or
submits QM9 test.

## Resource and durability contract

- One Kaggle2 GPU task is permitted for R7; no duplicate or automatic retry.
- Expected GPU time is below 30 minutes with a 90-minute stop bound.
- The accepted RWSE cache is mounted read-only; GPU graph building is forbidden.
- Remote preflight verifies the R3 anchor, parameter count, finite FP32
  forward/backward, split/cache identity, and no test-role read.
- Atomic progress, checkpoint, model, metrics, and train/validation payloads are
  independently retrievable.
- Local checks execute no model or inference.
