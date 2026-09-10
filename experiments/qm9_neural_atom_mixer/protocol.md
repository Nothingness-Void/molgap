# Frozen protocol — QM9 Neural-Atom global mixer

## Question

Can a low-rank, content-addressed four-slot global communication channel replace
dense GPS atom attention and improve Gap validation MAE beyond both full GPS
and a parameter-identical one-slot control?

## Architecture arms

- `full_gps`: unchanged 192-wide, 9-layer EdgeState GPS9 with RWSE16.
- `neural_atom_k1`: the same initialized local EdgeState path with atom global
  attention removed; one of four allocated latent slots is active after blocks
  3, 6, and 9.
- `neural_atom_k4`: parameter-identical to `neural_atom_k1`, with all four
  latent slots active.

Each mixer uses learned slot queries, node-key/value allocation normalized over
valid atoms, latent self-attention, and the same allocation transposed for
latent-to-node return. Its final return projection is zero-initialized. The
candidate is not the closed R7 graph token: it replaces GPS attention, pools by
content rather than mean, exchanges among multiple slots, and returns
atom-specific mixtures only at blocks 3/6/9.

## Frozen training contract

- task: direct QM9 Gap prediction using transferable OGB categorical atom/bond
  features and RWSE16;
- roles: exactly 30,000 train and 3,000 internal validation graphs from accepted
  cache aggregate
  `80a2d256ccbbde8de5862e7ad4fd61ea7c1255a19217e590033a94cabe1c6340`;
- seed 42, FP32, physical batch 128 per independent model/device;
- AdamW, learning rate `4e-4`, weight decay `1e-5`, gradient clip 1;
- cosine schedule to `1e-6`, exactly 40 epochs, identical row ordering and
  sample exposure;
- Kaggle2 T4x2: GPU0 trains `full_gps`; GPU1 sequentially trains the two mixer
  arms, each with independent model/RNG/optimizer/checkpoint/output state.

No held-out/test graph, official PCQM validation/test-dev role, checkpoint,
pretraining, teacher distillation, target residual, prediction fusion, ETKDG,
3D coordinate, desktop 304-wide result, SCNet result, IMS asset, or full-scale
result may enter the screen.

## Remote preflight

Before training, verify without comparing separate CUDA whole-model forwards:

- K1 and K4 have identical parameter counts and initial state hashes;
- inactive-slot masking leaves exactly one or four active slots;
- assignment mass sums to one over valid atoms and is zero on padding;
- every return projection and direct mixer update is exactly zero initially;
- a full forward/backward produces finite, nonzero gradients on return
  projections;
- candidate parameter count is at most 4.2M.

## Promotion and stopping gates

`neural_atom_k4` is nominated for exactly one paired PCQM-100K transfer only if
all conditions hold:

- validation MAE gain versus `full_gps` is at least `0.003 eV`;
- validation MAE gain versus `neural_atom_k1` is at least `0.001 eV`;
- mean epoch-time ratio versus `full_gps` is at most `1.15`;
- peak-memory reserve is at least 15%; and
- all mechanical/contract/hash checks pass.

Failure closes this exact slot count, exchange placement, latent width, seed,
optimizer, and schedule. Infrastructure-only failure may be repaired under the
unchanged contract and does not consume a scientific attempt.
