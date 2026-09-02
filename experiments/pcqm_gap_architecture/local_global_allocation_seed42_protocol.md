# Local/global allocation seed-42 protocol

## Question

Does the accepted distance-plus-angle Sparse Triangle EdgeState encoder benefit
from global atom attention in every block, or can a local-heavy allocation give
lower internal-validation Gap MAE with fewer parameters and higher throughput?

This is one information-flow question. It does not authorize a new dataset,
optimizer search, width/depth search, target residual, prediction fusion,
pretraining, extra conformer, extra seed, official validation/test-dev access,
full-data training, or molecular-research-server work.

## Evidence and attribution

GPS++ reports that its edge-aware local MPNN path contributes more than global
attention on PCQM and that its 2D MPNN-only variant can outperform the 2D
hybrid. A controlled OGB-PCQM study likewise reports a smaller local encoder
ahead of its GPS-heavy alternative. Neither result proves that the accepted
MolGap encoder should remove all attention, because its persistent bond, wedge,
and geometry paths differ. A matched three-way screen is required.

The experiment changes only the allocation of graph-global communication:

```text
shared input and local path in every candidate
Atom + RWSE16
       <-> persistent real-bond EdgeState64
       <-> sparse wedge state16
       <-> ETKDG distance/angle bottom fusion
       <-> nine local ResGatedGraphConv updates

candidate A: global atom MHA in blocks 1--9
candidate B: global atom MHA only in blocks 3, 6, 9
candidate C: no atom MHA; one shared 64d graph state updates/broadcasts
             only after blocks 3, 6, 9
```

Candidate C pools mean, sum, and max node summaries into one persistent graph
state. A shared gated update and rank-32 graph-to-atom projection broadcast the
state. It is representation-level communication, not prediction fusion.

## Frozen data and training contract

- official-train-derived roles: 100,000 train and 10,000 internal validation;
- parent graph SHA-256:
  `eb7c843e33f430ac755bc575d80153aba87677cea1ad5bb0dcf73cca906e2c21`;
- wedge SHA-256:
  `dc62b8289b0d85bd71a2eca9a16b6223f53206dd9a901670bb799125eff77406`;
- ETKDG distance/angle cache SHA-256:
  `3e4206fd239942ab79f9c4978cd4334f6025ba14e8f1cd1df78c20060a0d1d22`;
- seed 42, FP32, batch 48, AdamW, learning rate `1.6e-4`, weight decay
  `1e-6`, at most 40 epochs, patience 8, identical target normalization,
  scheduler, shuffle order, loaders, direct scalar Gap head, and mean pooling;
- every candidate must remain at or below 5.2M trainable parameters;
- one Kaggle T4x2 task with candidate-level process isolation: GPU 0 trains
  fresh full-GPS, while GPU 1 trains sparse-GPS then GraphState; the shared
  search wall budget remains 39,600 seconds;
- official validation and test-dev remain unread.

The full-GPS comparator is trained fresh inside the same task. The previously
accepted seed-42 value `0.1353926807641983 eV` is retained only as a reproducibility
anchor; advancement is decided against the fresh same-task comparator.

## Remote preflight and durability

Before training, the remote GPU preflight must verify:

- global-attention blocks are exactly `1..9`, `3/6/9`, and none;
- only the no-attention candidate owns the graph-state module;
- all common state-dict tensors are bit-identical after matched seeded
  initialization;
- parameter counts are finite and within the ceiling;
- one batch gives finite predictions, loss, and gradients;
- sufficient GPU memory remains.

Every epoch writes an atomic checkpoint and complete trace. Each completed
candidate writes a best model, validation payload, metrics, and hashes so a
terminal task can be accepted without local model inference.

The two workers inherit one read-only graph cache through Linux copy-on-write,
set disjoint `CUDA_VISIBLE_DEVICES` values before CUDA initialization, use
independent RNG state, model, optimizer and checkpoint directories, and never
combine gradients. Parallelism changes wall-clock allocation only; it does not
change the scientific contract.

## Decision rule

Sparse-GPS or GraphState advances only if it is strictly lower than the fresh
full-GPS validation Gap MAE and all artifacts pass no-inference acceptance.
Parameter count and throughput are reported but cannot rescue a worse MAE.

A passing seed 42 grants only eligibility to plan seeds 43/44. Failure closes
the exact global schedule without changing seed, width, block positions,
optimizer, learning rate, graph-state width, or training horizon. The completed
ring-hierarchy cache remains accepted and deferred; its GPU candidate is not
submitted concurrently.
