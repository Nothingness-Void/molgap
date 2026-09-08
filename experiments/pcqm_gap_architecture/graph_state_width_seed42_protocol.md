# GraphState Width Seed-42 Protocol

## Question

Does the accepted GraphState9 encoder lose useful molecule-level information
because its recurrent graph state is only 64 channels wide?

This is a paired 100K architecture screen. It does not authorize extra seeds,
the available full PCQM4Mv2 corpus, official validation, test-dev, production
changes, or molecular-research-server access.

## Frozen pair

Both candidates retain the same OGB categorical inputs, RWSE16, persistent
EdgeState64, sparse WedgeState16, ETKDGv3/MMFF94s distance-angle bottom
fusion, nine local ResGatedGraphConv blocks, direct scalar Gap head, and
GraphState exchanges after blocks 3/6/9:

1. `ogb_distance_angle_triangle_edge_state_graph_state9`: GraphState64.
2. `ogb_distance_angle_triangle_edge_state_graph_state9_w128`: GraphState128.

Only the recurrent graph-state width changes. The low-rank graph-to-atom
exchange remains rank 32. No atom-level attention, new geometry, pretraining,
fusion, routing, residual target, or data augmentation is introduced.

## Data and optimization

- official PCQM4Mv2 database;
- first 100,000 official-train rows and the frozen 10,000 train-derived
  internal-validation rows;
- accepted ETKDGv3/MMFF94s geometry cache with aggregate SHA-256
  `3e4206fd239942ab79f9c4978cd4334f6025ba14e8f1cd1df78c20060a0d1d22`;
- seed 42, FP32, batch 48 per candidate;
- AdamW, learning rate `1.6e-4`, weight decay `1e-6`;
- at most 40 epochs, cosine schedule, patience 8;
- preferred execution: one isolated fresh candidate per Kaggle T4 in a single
  T4x2 task;
- scheduler fallback after repeated single-P100 allocation: two concurrently
  submitted single-GPU kernels, one candidate per kernel, accepted only when
  both report the same GPU model and immutable source/cache identities;
- 4,000,000-parameter ceiling and 14,400-second search budget.

The two models receive the same seed and all equal-shaped parameters outside
`graph_context` must initialize bit-identically. Width-dependent
`graph_context` tensors are recorded explicitly rather than treated as shared
parameters.

## Durability and decision

Every epoch atomically writes a resumable checkpoint and trace. Each candidate
must emit a hashed best model, last checkpoint, aligned 10K prediction payload,
metrics, and finite CUDA forward/backward preflight.

The fallback changes only process placement. It does not change model code,
data order, seed, optimizer, schedule, early stopping, target, or acceptance
arithmetic. Results from mismatched GPU models are not scientifically accepted.

Promotion requires all of:

- GraphState128 improves the paired fresh GraphState64 control by at least
  `0.001 eV` validation Gap MAE;
- candidate throughput is at least 70% of the control;
- at least 15% T4 memory remains after preflight;
- source, cache, row, target, checkpoint, and sealed-role acceptance passes.

A failure closes this width increase. A pass records one seed-42 candidate;
seeds 43/44 and the 1M/full scale gates require separate authorization.
