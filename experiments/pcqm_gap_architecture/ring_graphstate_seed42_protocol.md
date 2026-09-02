# Ring-GraphState seed-42 protocol

## Question

Does deterministic smallest-ring communication add useful mesoscale chemistry
to the frozen GraphState9 winner on the official-train-derived PCQM 100K/10K
screen?

This is one material architecture change. It does not authorize an attention
retry, ring-definition search, width/depth search, extra seed, full-data
training, official validation/test-dev access, or molecular-research-server
work.

## Frozen baseline and input

- Baseline: `ogb_distance_angle_triangle_edge_state_graph_state9`.
- Candidate:
  `ogb_distance_angle_ring_hierarchy_triangle_edge_state_graph_state9`.
- Both use the same OGB atom/bond encoders, RWSE16, node width 192, persistent
  real-bond EdgeState64, sparse wedge state16, ETKDG distance/angle bottom
  fusion, nine local ResGatedGraphConv blocks, shared GraphState64 exchanges
  after blocks 3/6/9, atom-mean readout, and direct scalar Gap head.
- Parent geometry SHA-256:
  `3e4206fd239942ab79f9c4978cd4334f6025ba14e8f1cd1df78c20060a0d1d22`.
- Accepted ring cache SHA-256:
  `3f8b271571b8d1026e96fc1dae51d9479489ddd13b73df95740288e6f630779f`.
- Roles remain exactly 100,000 train and 10,000 internal validation graphs.
  Official validation and test-dev remain unread.

## One added mechanism

The candidate reuses the accepted deterministic `GetSymmSSSR` hierarchy:
12-channel ring descriptors, atom--ring memberships, and four-channel
spiro/fused/direct/conjugated ring relations. One shared recurrent RingState64
cell updates after atom blocks 2, 4, 6, and 8. A rank-32 ring-to-atom return is
zero initialized. At block 6 the ring update precedes the existing GraphState
exchange, so newly aggregated ring information can enter the molecule state.

```text
Atom/Edge/Wedge/Geometry local state
          | blocks 2/4/6/8
          v
      RingState64
          |
          +---- rank-32 return ----> atoms

Atoms ---- blocks 3/6/9 ----> GraphState64 ----> atoms
```

The baseline has 3,665,809 parameters and the candidate must have exactly
3,723,849. Their shared parameters must be bit-identical under matched seed-42
initialization; the zero ring return must make the candidate initially
function-equivalent while retaining a finite nonzero return-path gradient.

## Training and execution

- one private Kaggle1 T4x2 task;
- GPU 0 trains a fresh GraphState9 baseline and GPU 1 trains a fresh
  Ring-GraphState candidate in isolated processes;
- seed 42, FP32, batch 48, AdamW, learning rate `1.6e-4`, weight decay `1e-6`,
  at most 40 epochs, patience 8, normalized L1 loss, and cosine schedule;
- no warm start, checkpoint transfer, residual target, auxiliary target,
  pretraining, prediction fusion, or optimizer search;
- atomic epoch checkpoints and independently retrievable metrics, trace,
  best model, validation payload, and hashes;
- 14,400-second GPU search budget and 4.0M parameter ceiling.

## Decision

The candidate passes seed 42 only if it has strictly lower internal-validation
Gap MAE than the fresh same-task GraphState9 baseline, every artifact passes
no-model acceptance, and the parameter/time contracts hold. A pass grants only
eligibility to plan seeds 43/44. A scientific failure closes this exact ring
addition without parameter or seed variants.
