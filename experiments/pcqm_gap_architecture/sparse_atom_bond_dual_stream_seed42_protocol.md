# Sparse Atom--Bond Dual Stream Seed-42 Protocol

## Question

Does a separately normalized sparse real-bond attention stream with symmetric
atom--bond exchange improve the accepted distance-plus-angle Sparse Triangle
EdgeState GPS9 on the frozen official-train-derived PCQM 100K/10K roles?

## Frozen comparator and data

- Comparator: `ogb_distance_angle_triangle_edge_state_gps9` from the accepted
  sparse-torsion paired run.
- Comparator seed: 42; validation Gap MAE: `0.1353926807641983 eV`;
  parameter count: 4,891,057.
- The comparator's complete metrics, 40-epoch trace, model, checkpoint, and
  validation payload are taken from the hash-pinned private dataset
  `nothingnessvoid/molgap-pcqm-sparse-torsion-s42-resume-v3`; no comparator
  prediction is recomputed and no label-dependent fusion is performed.
- Graph roles come from the accepted account1 torsion-cache mirror. Torsion
  tensors are removed after load; the experiment reuses only the frozen
  100,000 train and 10,000 internal-validation graphs, OGB categories, RWSE16,
  sparse wedge ids, ETKDG bond distances, angle cosines, and geometry masks.
- The accepted parent graph, wedge, geometry, and cache aggregate hashes remain
  fixed. Official validation and test-dev remain unread.

Using the accepted comparator is conservative: it is lower than the earlier
seed-42 geometry-screen result under the same architecture. A fresh comparator
would consume roughly half the task without adding a new scientific question.

## Candidate architecture

- Candidate name:
  `ogb_distance_angle_dual_stream_triangle_edge_state_gps9`.
- Preserve node width 192, EdgeState width 64, wedge width 16, RWSE16, nine
  four-head GPS blocks, distance-plus-angle bottom fusion, mean pooling, and a
  direct scalar Gap head.
- Add four sparse bond-stream updates after atom-GPS blocks 2, 4, 6, and 8.
- Each update uses a bond-local LayerNorm, four heads of width 16, and segmented
  attention only over directed real bonds connected by an accepted
  non-backtracking wedge. It has no dense bond-pair matrix, global bond
  attention, learned graph query, or bond-level prediction head.
- Each bond block has a separately normalized gated two-layer FFN with
  expansion factor 2 and dropout 0.1.
- One shared rank-32 exchange maps the mean of the two endpoint atom states
  into each directed bond and maps the mean of incoming directed bond states
  back to each atom. Atom-to-bond and bond-to-atom value and gate projections
  are separate.
- Attention output, FFN output, and exchange value projections are
  zero-initialized. Exact shared-backbone parameter equality and zero injection
  are checked before the GPU forward preflight.
- Parameter ceiling: 5,200,000. The candidate must fit without reducing the
  accepted backbone width or depth.

## Training and resource contract

- Seed 42 only; fresh random initialization; FP32; batch 48.
- AdamW, learning rate `1.6e-4`, weight decay `1e-6`, cosine schedule, at most
  40 epochs, patience 8, normalized direct-Gap L1 loss.
- One Kaggle1 GPU task and a hard 14,400-second search budget. The expected
  range is 3--3.6 P100 hours because only the candidate is trained.
- Atomic epoch checkpoints include model, optimizer, scheduler, DataLoader
  generator, CPU RNG, CUDA RNG, best state, and trace. All outputs are
  independently retrievable.
- No width/head/dropout/LR grid, checkpoint transfer, auxiliary loss,
  HOMO/LUMO target, prediction fusion, second encoder, new conformer,
  pretraining, official-role access, or molecular-research-server use.

## Acceptance and stop rule

1. Preflight must verify the exact source/cache/comparator hashes, parameter
   count, initial shared-backbone equality, zero new injections, finite forward,
   finite loss, finite gradients, and one-card memory.
2. The candidate trace and all model/checkpoint/payload/metrics hashes must pass
   no-inference acceptance. Validation row and target hashes must match the
   frozen comparator.
3. The route advances only if candidate validation Gap MAE is strictly below
   `0.1353926807641983 eV`.
4. A non-improvement closes this mechanism without seed, width, depth,
   optimizer, or epoch retry. A strict seed-42 improvement records eligibility
   for a separately authorized paired seeds 43/44 confirmation and stops.
5. Full-data training, official validation/test-dev, desktop submission, and
   molecular-research-server work remain unauthorized by this screen.
