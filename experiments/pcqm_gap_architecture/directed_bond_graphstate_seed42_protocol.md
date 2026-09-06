# Directed-bond GraphState seed-42 protocol

## Question

Does explicitly directional, non-backtracking bond memory improve the accepted
GraphState9 anchor when the target is the official PCQM Gap?

## Isolated mechanism

Both paired models retain OGB atom/bond categories, RWSE16, one ETKDGv3+MMFF94s
conformer, bond lengths, bonded angles, persistent 64-channel EdgeState,
persistent 16-channel wedge state, nine local atom blocks, GraphState exchange
after blocks 3/6/9, mean pooling, and one direct Gap head. The challenger alone
adds one per-layer edge-to-edge path:

1. use the accepted directed wedge list `i->j->k`;
2. average the predecessor states `i->j` onto each outgoing edge `j->k`;
3. apply a separate LayerNorm/64-channel MLP;
4. add the result before the normal persistent EdgeState update.

The return projection is zero initialized, so the challenger and control have
the same initial function and all shared tensors have identical seed-42 values.
This is not sparse all-bond attention, a second encoder, or prediction fusion.

## Frozen comparison

- data: accepted official-train-derived PCQM 100,000 train / 10,000 internal
  validation geometry cache;
- roles: official validation and test-dev unread;
- execution: one private Kaggle1 T4x2 job, one isolated process and GPU per model;
- seed 42, FP32, batch 48, AdamW, learning rate `1.6e-4`, weight decay `1e-6`;
- 40 epochs, cosine decay to `1e-6`, patience 8;
- parameter ceiling: 4,000,000 per model;
- atomic epoch checkpoint, best model, trace, validation payload, metrics and
  SHA-256 manifest.

The challenger advances only if its internal-validation Gap MAE is strictly
lower than the fresh paired GraphState9 control. A seed-42 win records a
promising mechanism only; it does not authorize seeds 43/44, full training,
official-role reads, production changes, or molecular-research-server access.
