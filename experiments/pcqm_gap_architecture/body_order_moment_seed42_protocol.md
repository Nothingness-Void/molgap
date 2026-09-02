# Compact body-order moment seed-42 protocol

## Question

Can one cheap, rotationally invariant local-environment summary improve the
fresh GraphState9 control without explicit new atom pairs, equivariant tensor
propagation, or another recurrent state?

This is a CACE/CEIT-inspired attribution test, not a reproduction of either
model. It uses only accepted ETKDGv3+MMFF94s positions and directed real bonds.

## Matched candidates

1. Fresh baseline:
   `ogb_distance_angle_triangle_edge_state_graph_state9`, 3,665,809
   parameters.
2. Candidate:
   `ogb_distance_angle_body_order_triangle_edge_state_graph_state9`, exactly
   3,681,329 parameters.

Both retain OGB atom/bond categories, RWSE16, width 192, EdgeState64,
WedgeState16, distance/angle bottom fusion, nine local ResGatedGraphConv
blocks, GraphState64 exchanges after blocks 3/6/9, mean pooling, and direct
scalar Gap prediction. Neither contains atom self-attention.

The candidate adds only one pre-message node injection. For each directed real
bond ending at atom `i`, it expands the bond length with a fixed 16-channel
Gaussian basis and the normalized Cartesian direction `u_ji`. Per radial
channel it forms

- scalar density `sum_j R_k(r_ji)`;
- vector moment `sum_j R_k(r_ji) u_ji`;
- rank-2 moment `sum_j R_k(r_ji) u_ji u_ji^T`.

Only rotational invariants enter the network: scalar density, squared vector
norm, and squared Frobenius norm of the rank-2 moment. Their 48 channels pass
through LayerNorm, `48 -> 64 -> 192`, with SiLU and a zero-initialized final
bias-free projection. Invalid-geometry graphs contribute an all-zero
injection. There is no cutoff graph, explicit tuple enumeration, learned
coordinate frame, force target, auxiliary loss, or update after initialization.

Shared baseline parameters must be bit-identical at matched initialization.
The zero output must make the initial candidate function equal to the baseline
while preserving a finite nonzero gradient into the final projection.

## Training and execution

- one private Kaggle2 T4x2 task, one isolated candidate per GPU;
- accepted geometry cache SHA-256
  `3e4206fd239942ab79f9c4978cd4334f6025ba14e8f1cd1df78c20060a0d1d22`;
- seed 42, FP32, batch 48, AdamW, learning rate `1.6e-4`, weight decay
  `1e-6`, at most 40 epochs, patience 8, normalized L1 loss, cosine schedule;
- random initialization; no warm start, residual target, auxiliary target,
  pretraining, prediction fusion, or optimizer search;
- atomic epoch checkpoints and independently retrievable metrics, validation
  payload, traces, hashes, and cache lineage;
- 14,400-second search budget and 4.0M parameter ceiling.

No new CPU graph cache is required because all inputs already passed geometry
acceptance. Official PCQM validation/test-dev, extra seeds, full-data training,
and the molecular-research server remain out of scope.

## Decision

The candidate passes only if its matched seed-42 internal-validation Gap MAE is
strictly lower and all mechanical, identity, parameter, time, finite-value,
zero-injection, gradient, and artifact checks pass. A scientific failure closes
this exact moment basis without radial-channel, order, width, seed, optimizer,
or schedule variants.
