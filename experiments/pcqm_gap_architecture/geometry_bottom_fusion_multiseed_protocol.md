# Geometry Bottom-Fusion Paired Multiseed Protocol

## Question

Does the seed-42 distance-plus-angle bottom-fusion result reproduce against
freshly trained, same-seed pure-2D Sparse Triangle comparators at seeds 43 and
44?

## Frozen data and representation contract

- Parent official-train-derived PCQM graph cache SHA-256:
  `eb7c843e33f430ac755bc575d80153aba87677cea1ad5bb0dcf73cca906e2c21`.
- Accepted sparse-wedge cache SHA-256:
  `dc62b8289b0d85bd71a2eca9a16b6223f53206dd9a901670bb799125eff77406`.
- Accepted ETKDGv3+MMFF94s geometry-cache SHA-256:
  `3e4206fd239942ab79f9c4978cd4334f6025ba14e8f1cd1df78c20060a0d1d22`.
- Exactly 100,000 training and 10,000 internal-validation roles. The 315
  invalid geometries retain their rows and use the accepted explicit geometry
  mask; no molecule is dropped or replaced.
- Official PCQM validation and test-dev remain unread.

## Paired model contract

- Comparator: `ogb_sparse_triangle_edge_state_gps9` with no geometry input.
- Candidate: `ogb_distance_angle_triangle_edge_state_gps9` with the accepted
  single deterministic conformer fused into persistent bond-distance and
  wedge-angle states before every GPS block.
- Seeds 43 and 44. For each seed, a fresh comparator is trained first and the
  geometry candidate second in the same sequential Kaggle GPU task.
- The two models use the same explicit per-seed data-order generator. Both
  start from random initialization; there is no warm start, checkpoint
  transfer, prediction fusion, residual target, auxiliary target, or ensemble.
- FP32, batch 48, AdamW, learning rate `1.6e-4`, weight decay `1e-6`, at most
  40 epochs, patience 8, normalized L1 loss, cosine schedule, direct scalar Gap,
  and a 5.2M parameter ceiling.
- Distance-only and angle-only do not advance and are absent from this task.

The accepted seed-42 pure-2D and distance-plus-angle metrics are immutable
references; they are not retrained in this confirmation task.

## Acceptance and stop rule

1. Every one of the four new runs must emit an atomic best model, resumable
   checkpoint, trace, metrics, and aligned internal-validation payload.
2. No-inference acceptance must verify source/cache identities, exact parameter
   counts, seeds, training contracts, row/target identity, artifact hashes, and
   sealed-role flags.
3. The geometry candidate must have lower validation Gap MAE than its paired
   comparator at seeds 42, 43, and 44 and in their arithmetic mean.
4. Any non-improving seed closes this geometry architecture. No seed, width,
   geometry mode, conformer, optimizer, schedule, or threshold retry is allowed.
5. A pass freezes one candidate for a separate A100 timing gate. It does not
   authorize full-data training, official validation/test-dev, desktop
   submission, or molecular-research-server access.

The sequential GPU task is bounded to 11 hours. Only one Kaggle GPU kernel may
run at a time. Infrastructure-only failures may be repaired and resubmitted
under the repository-wide unchanged-contract retry policy.
