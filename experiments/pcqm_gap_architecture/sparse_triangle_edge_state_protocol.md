# Sparse Triangle EdgeState GPS9 Seed-42 Protocol

## Question

Does a sparse topology-wedge state improve the frozen persistent-real-bond
EdgeState GPS9 comparator for direct PCQM4Mv2 Gap prediction without the dense
all-pairs cost of PairGPS?

## Frozen architecture

- Candidate key: `ogb_sparse_triangle_edge_state_gps9`.
- Official OGB categorical atom and bond encoders with topology-only RWSE16.
- Nine GPS layers, width 192, four attention heads, dropout 0.1, mean pooling,
  and one direct scalar Gap head.
- Persistent 64-channel directed real-bond EdgeState is retained.
- A 16-channel persistent state is attached only to directed non-backtracking
  wedges `i -> j -> k` formed by adjacent bonds. Each wedge update reads the
  two current bond states and the center-node state.
- Wedge state is aggregated sparsely back to its two bond states and its center
  node before the matched GPS block. No dense atom-pair tensor is constructed.
- No coordinates, 3D features, target residual, prediction fusion, ensemble,
  checkpoint, pretraining, distillation, or HOMO/LUMO auxiliary target.

## Frozen data and optimization

- Parent cache: accepted official-train-derived PCQM4Mv2 Gap100K cache with
  aggregate SHA-256
  `eb7c843e33f430ac755bc575d80153aba87677cea1ad5bb0dcf73cca906e2c21`.
- Exactly 100,000 train graphs and 10,000 internal-validation graphs. Official
  validation and test-dev roles remain unread.
- The CPU stage derives and accepts a separate immutable wedge cache from that
  parent cache. The GPU stage may read only the accepted derived cache.
- Seed 42, FP32, batch 48, AdamW, learning rate `1.6e-4`, weight decay
  `1e-6`, at most 40 epochs, patience 8.
- Parameter ceiling: `5,200,000`.
- Frozen comparator: `ogb_edge_state_structural_gps9`, internal-validation Gap
  MAE `0.13798263211250306 eV`.

## Execution and gates

1. The CPU cache must pass no-model acceptance: parent identity, graph counts,
   wedge index validity, shard hashes, aggregate hash, and both sealed-role
   flags.
2. Only after CPU acceptance, one Kaggle GPU seed-42 task may run the candidate.
3. The candidate advances only if its downloaded artifacts pass no-inference
   acceptance and its internal-validation Gap MAE is strictly below the frozen
   comparator.
4. A seed-42 pass only opens a separate decision for seeds 43/44; it does not
   authorize full-data training, official validation, test-dev inference, or
   molecular-research-server access.
5. A failure closes this information-flow question. No seed, width, schedule,
   optimizer, or cache-threshold retry is allowed under this protocol.

## Timing boundary

The CPU cache is a graph-representation preflight. Before any full-data run,
the selected architecture must pass the existing single-A100 gate: at least
1,800 graphs/s, no epoch above 32 minutes, projected end-to-end time at most
10.5 hours, and at least 15% A100 memory reserve.
