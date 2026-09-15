# Frozen protocol — K1 GPS++ directional local adapter

## Question and arms

Can a compact GPS++-style edge-conditioned local residual improve K1 by
returning outgoing bond messages explicitly to sender atoms?

- `neural_atom_k1_gpspp_sender`: after every unchanged K1 local block, add a
  zero-initialized residual built only from messages aggregated to each bond's
  source atom.
- `neural_atom_k1_gpspp_bidirectional`: use the identical parameterization and
  initialization, but concatenate separately normalized receiver and sender
  aggregates before the same residual projection.

Each adapter computes a 64-channel real-bond message from normalized source
atom, target atom and persistent EdgeState. Per-node means are concatenated
with the normalized node state and passed through a 128-channel bottleneck.
The final projection starts at exactly zero, so both candidates initially
compute the frozen K1 function. The nine adapters add 857,088 parameters;
both candidates contain 4,515,905 parameters.

K1's nine local ResGatedGraphConv blocks, persistent 64-channel real-bond
EdgeState, RWSE16, one 64-channel molecular slot at layers 3/6/9, mean readout
and direct Gap head remain unchanged. No geometry, path cache, motif, extra
slot, target residual, fusion, teacher or pretraining is used.

## Immutable V4 contract

Use Kaggle2 `kaseichou/pcqm4mv2-ogb-fixed-100k-v1`, manifest SHA
`1b0e8fd579ab1cb86c02e833e7ad284b4af7582b059f912a77853fdccf3ede6d`.
Train `[0,100000)` and select only on `[100000,150000)`. Seed42, physical
BS128 per independent T4, strict deterministic FP32/noTF32, drop-last, 40
epochs, 31,240 optimizer steps, 3,998,720 presentations, AdamW `4e-4`, weight
decay `1e-5`, clip1, cosine `1e-6`, normalized direct-Gap L1 and minimum-dev
selection are frozen. Official validation, shadow and every test role remain
absent and unread.

Remote preflight must verify the exact K1 shared-state hash, exact initial
predictions, both directional equations, finite gradients after two real
optimizer steps, deterministic runtime calibration, at least 15% memory
reserve and an enforced ten-hour wall bound. Each T4 owns one model, RNG,
optimizer, checkpoint tree and log. Atomic per-epoch checkpoints and ten-epoch
recovery chunks are mandatory.

## Decision gate

Saved-artifact acceptance recomputes the ordered 50K development MAE without
constructing or executing a model. An arm passes only with gain at least
`0.003 eV`, paired-bootstrap upper95% below zero, complete artifact/source/RNG
identity and at least 15% memory reserve. The bidirectional arm must also beat
the sender-only capacity control before receiver-plus-sender interaction can be
credited. A pass creates mechanism-shortlist evidence only; it does not release
another seed, scale-up, shadow, official evaluation or desktop handoff.

This is the third and final round under the authorization recorded by
`../pcqm_k1_edge_memory_100k/results/authorization.json`. Any scientific loss
closes the adapter family without width, placement, seed or optimizer rescue.
