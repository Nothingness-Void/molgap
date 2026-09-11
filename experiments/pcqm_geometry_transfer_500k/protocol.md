# Fixed-500K geometry-transfer protocol

## Question

Do ETKDGv3+MMFF94s bond lengths and bond angles improve each of the two accepted
pure-2D information-flow cores at fixed 500K scale, and can their predictions
be combined without an Oracle router?

## Data

- Official PCQM4Mv2 training role only.
- Train source indices `0:500000`; development `500000:550000`.
- Accepted SCNet aggregate SHA-256:
  `676a506c808402bc16a4437cc02286168239a8dd5de4451e992131eddb4f1b20`.
- OGB categorical atoms/bonds, RWSE16, ETKDGv3+MMFF94s bond distances and wedge
  angles. The same geometry contract is required at inference.
- Official validation, test-dev, and test-challenge remain unread.

## Arms

`geometry_gptrans_t` retains the accepted 12-layer, 256-node/32-pair GPTrans-T
core. A zero-start distance projection enters real-bond pair states and a
zero-start angle projection enters each wedge center's node state.

`geometry_neural_atom_k1` retains nine local persistent-EdgeState layers and
one 64-channel Neural-Atom exchange at layers 3/6/9. The accepted sparse wedge
and distance/angle bottom-fusion path is inserted into its local backbone.

Both additions are zero initialized, so each candidate starts from its original
architecture's function under the same shared initialization.

## Training

- Seed 42, strict FP32, physical BS128, one optimizer step per batch.
- `drop_last=True` inside every immutable 50K shard: 3,900 steps and 499,200
  presented rows per epoch; no partial optimizer batch.
- GPTrans-T keeps its accepted AdamW `1e-3`, weight decay `0.05`, four-epoch
  warmup, cosine 60-epoch schedule, clipping 1.0, and EMA 0.9999.
- K1 keeps its accepted AdamW `4e-4`, weight decay `1e-5`, cosine 40-epoch
  schedule, and clipping 1.0.
- Every epoch atomically replaces the resumable last checkpoint, trace,
  progress record, and best model/prediction payload.

Historical pure-2D scores are contextual references because their older jobs
did not satisfy this exact no-tail v4 contract. This screen may nominate but
cannot by itself claim a causal geometry gain without a matching v4 reference
or a disjoint audit.

## Fusion

After both encoders complete, a CPU dependency validates identical targets and
source order, then reports:

- each component;
- equal prediction blend;
- five-fold out-of-fold scalar convex blend;
- per-row Oracle upper bound, labeled as non-deployable headroom only.

No model is selected from the Oracle and no sealed role is opened.
