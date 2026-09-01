# Sparse Triangle EdgeState Paired Multiseed Protocol

## Question

Does the seed-42 Sparse Triangle result reproduce against freshly trained,
same-seed real-bond EdgeState comparators at seeds 43 and 44?

## Frozen data and model contract

- Parent official-train-derived PCQM cache SHA-256:
  `eb7c843e33f430ac755bc575d80153aba87677cea1ad5bb0dcf73cca906e2c21`.
- Accepted sparse-wedge cache SHA-256:
  `dc62b8289b0d85bd71a2eca9a16b6223f53206dd9a901670bb799125eff77406`.
- Exactly 100,000 training and 10,000 internal-validation graphs.
- Complete OGB categorical atoms and real bonds, RWSE16, direct scalar Gap.
- Comparator: `ogb_edge_state_structural_gps9`.
- Candidate: `ogb_sparse_triangle_edge_state_gps9`.
- Seeds 43 and 44. Each seed trains a fresh comparator and candidate from
  random initialization in the same sequential Kaggle GPU task.
- FP32, batch 48, AdamW, learning rate `1.6e-4`, weight decay `1e-6`, at most
  40 epochs, patience 8, cosine schedule, and a 5.2M parameter ceiling.
- No warm start, checkpoint transfer, prediction fusion, residual target,
  auxiliary HOMO/LUMO target, coordinates, 3D input, or test evaluation.

The accepted seed-42 comparator and candidate metrics are immutable references;
they are not retrained in the confirmation task.

## Acceptance and advancement

1. Every one of the four new runs must emit an atomic best model, resumable
   checkpoint, trace, metrics, and aligned internal-validation payload.
2. A no-inference acceptance must verify source, cache, seed, contract,
   parameter count, row-identity hash, target hash, artifact hashes, and sealed
   role flags.
3. The candidate must have lower validation Gap MAE than the paired comparator
   at seeds 42, 43, and 44 and in the three-seed arithmetic mean.
4. Any non-improving seed closes Sparse Triangle. No seed, width, wedge rank,
   optimizer, schedule, or threshold retry is permitted.
5. A pass freezes the architecture definition for a separate cost and handoff
   decision. It does not itself authorize full training, official validation,
   test-dev, or molecular-research-server work.

The GPU task is bounded to 11 hours so one Kaggle session cannot silently turn
into an unbounded search. Only one GPU kernel may run at a time.
