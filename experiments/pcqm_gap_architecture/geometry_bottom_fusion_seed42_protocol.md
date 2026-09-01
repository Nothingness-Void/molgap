# Geometry Bottom-Fusion Seed-42 Screen

## Question

Can one deterministic ETKDGv3+MMFF94s conformer improve the accepted pure-2D
Sparse Triangle EdgeState GPS9 when geometry is injected into the persistent
bond/wedge states rather than fused at the prediction head?

## Immutable data contract

- The parent is the accepted official-train-derived PCQM 100K/10K sparse-wedge
  cache. Parent graph SHA-256:
  `eb7c843e33f430ac755bc575d80153aba87677cea1ad5bb0dcf73cca906e2c21`;
  wedge SHA-256:
  `dc62b8289b0d85bd71a2eca9a16b6223f53206dd9a901670bb799125eff77406`.
- Only the first 3,378,606 rows of `piero0/pcqm4mv2/data.csv` may be read to
  recover the SMILES for the already selected row indices. Official validation
  and test-dev remain unread.
- One heavy-atom conformer is generated with deterministic ETKDGv3. MMFF94s is
  run for at most 200 iterations when parameters exist. A failed geometry stays
  in its original role with zero geometry and an explicit invalid mask; no row
  is dropped or replaced.
- The CPU cache stores aligned heavy-atom positions, directed-bond distances,
  wedge-angle cosines, validity/convergence flags, reason-coded failures, shard
  hashes, and an aggregate hash. GPU work cannot start before no-inference
  acceptance of that immutable cache.

## Three candidates in one GPU notebook

All candidates retain OGB atom/bond categories, RWSE16, persistent 64-channel
real-bond EdgeState, persistent 16-channel topology-wedge state, nine 192-wide
GPS blocks, mean pooling, and direct scalar Gap prediction.

1. `ogb_distance_triangle_edge_state_gps9`: a fixed 16-channel Gaussian basis
   of ETKDG bond distance enters the bond state before every GPS block.
2. `ogb_angle_triangle_edge_state_gps9`: a fixed 16-channel Gaussian basis of
   wedge-angle cosine enters the wedge state before every GPS block.
3. `ogb_distance_angle_triangle_edge_state_gps9`: both bottom-fusion channels.

Geometry projections are zero-initialized so each candidate starts from the
accepted pure-2D function. There is no SchNet, independent 3D encoder, late
fusion, prediction residual, auxiliary target, pretrained checkpoint, ensemble,
or multiple conformer.

## Training and stop rule

- Seed 42 only; FP32, batch 48, AdamW, learning rate `1.6e-4`, weight decay
  `1e-6`, at most 40 epochs, patience 8, normalized L1 loss, cosine schedule.
- Parameter ceiling: 5,200,000 per candidate.
- The three candidates run sequentially in one bounded Kaggle GPU task with
  atomic per-candidate checkpoints and independently retrievable outputs.
- Frozen comparator: accepted pure-2D Sparse Triangle seed-42 validation Gap
  MAE `0.13790177369117737 eV`.
- A geometry mode is positive only when its accepted internal-validation MAE is
  strictly lower than the comparator. Only the lowest positive mode becomes
  eligible for later seeds 43/44; this notebook does not run additional seeds.
- If no mode is positive, bottom-fused ETKDG geometry closes without a retry.
- Completion does not authorize full training, official validation/test-dev,
  desktop submission, or molecular-research-server work.
