# ContactState paired GPU seed-42 protocol

## Question

Does the accepted non-covalent relation improve the frozen GraphState9
architecture when it is represented by one narrow, separately normalized
persistent state?

The accepted CPU cache is immutable: aggregate SHA-256
`49725b92c2c0d33e17633abf8ffa7148ebc8bc9721d3e5b3635f1309891bc826`,
100,000 train graphs, 10,000 internal-validation graphs, 3,658,038/366,116
directed train/validation contacts, and zero failures. Official PCQM validation
and test-dev remain unread.

## Matched candidates

1. Fresh baseline:
   `ogb_distance_angle_triangle_edge_state_graph_state9`, 3,665,809
   parameters.
2. Candidate:
   `ogb_distance_angle_contact_state_triangle_edge_state_graph_state9`,
   exactly 3,700,321 parameters.

Both retain OGB atom/bond categories, RWSE16, width 192, persistent real-bond
EdgeState64, sparse wedge state16, ETKDG bond-distance and angle bottom fusion,
nine local ResGatedGraphConv blocks, GraphState64 exchanges after blocks 3/6/9,
mean pooling, and direct scalar Gap prediction. Neither model contains atom
self-attention.

The candidate adds only:

- a fixed 16-channel Gaussian basis over contact distance;
- a separately normalized persistent ContactState32 initialized from its two
  endpoint atom states and distance basis;
- one shared gated ContactState update after blocks 2/4/6/8;
- an incoming-contact mean returned through a rank-16 low-rank projection whose
  value path is zero initialized.

Shared baseline parameters must be bit-identical at matched initialization.
The zero return must make the initial candidate function equal to the baseline
while preserving a finite nonzero gradient into the return projection.

## Training and execution

- one private Kaggle2 T4x2 task, with one isolated candidate per GPU;
- seed 42, FP32, batch 48, AdamW, learning rate `1.6e-4`, weight decay
  `1e-6`, at most 40 epochs, patience 8, normalized L1 loss, cosine schedule;
- random initialization; no warm start, checkpoint transfer, residual target,
  auxiliary target, pretraining, prediction fusion, or optimizer search;
- atomic epoch checkpoints plus independently retrievable model, trace,
  validation payload, metrics, hashes, and cache lineage;
- 14,400-second search budget and 4.0M parameter ceiling.

## Decision

The candidate passes seed 42 only if it has strictly lower matched
internal-validation Gap MAE, every artifact passes no-model acceptance, and
the parameter/time contracts hold. A pass is only a promising result; seeds
43/44 require a later explicit compute-budget decision. A scientific failure
closes this exact ContactState without cutoff, width, exchange-depth, seed,
optimizer, or schedule variants.
