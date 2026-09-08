# Sparse Relative-Value GraphState Seed-42 Protocol

## Question

Can path-conditioned sparse attention turn the weak positive exact-shortest
2/3-hop signal into a material PCQM Gap improvement without restoring dense
atom attention or exceeding the frozen resource envelope?

This is a one-seed architecture-discovery screen. It does not authorize
official validation, test-dev, full-data training, IMS access, or additional
seeds.

## Attribution basis

The frozen GraphState9 gain came from removing all nine dense global-attention
blocks and replacing them with three shared molecule-state exchanges. Later
state additions mostly moved validation MAE by less than `0.001 eV` while
making the model slower. Across five accepted fresh seed-42 GraphState controls,
validation MAE spanned about `0.000702 eV`, so unmatched sub-millielectronvolt
changes cannot support promotion.

The exact-shortest hop-path candidate was the strongest remaining weak positive:
`-0.0009066 eV` against its paired control. Per-molecule residual analysis
showed improvement concentrated in middle-size and middle-ring-density graphs,
while the smallest graphs and both target extremes regressed. Its uniform mean
aggregation cannot select a useful path or suppress weakly supported returns.

GRPE's primary ablation gives a matching mechanism: structural context in the
attention value path was more useful than a scalar topology bias alone. The
screen therefore changes the relation mixer, not the cache, depth, width, seed,
optimizer, target, or geometry.

## Frozen pair

Both models train from scratch in isolated T4 workers:

1. `ogb_distance_angle_triangle_edge_state_graph_state9`
2. `ogb_distance_angle_relative_value_triangle_edge_state_graph_state9`

The candidate retains RWSE16, persistent true-bond EdgeState64, sparse
WedgeState16, ETKDGv3/MMFF94s bond-distance and wedge-angle bottom fusion, nine
local ResGatedGraphConv blocks, and GraphState exchanges after blocks 3/6/9.
At the same three blocks it adds one shared 32-channel, four-head attention over
the accepted ordered exact-shortest 2/3-hop relations:

- target atom state forms the query;
- source atom state plus eight path features forms the key;
- source atom state plus the same path features forms the value;
- incoming relations receive a target-wise sparse softmax;
- log relation support enters the atom return gate;
- the final value projection is zero-initialized, preserving the baseline
  function at initialization while retaining a nonzero return gradient.

This is not dense global attention, prediction fusion, residual-target
learning, routing, pretraining, or a second geometry source.

## Frozen data and training contract

- database: official PCQM4Mv2;
- roles: first 100,000 official-train molecules plus the frozen 10,000
  official-train-derived internal validation rows;
- input: accepted geometry/hop-path cache only;
- official validation and test-dev: unread;
- seed: 42;
- precision: FP32;
- batch: 48 per candidate;
- optimizer: AdamW;
- learning rate: `1.6e-4`;
- weight decay: `1e-6`;
- scheduler: cosine over exactly 40 epochs;
- patience: 8;
- target: direct scalar Gap;
- execution: one fresh control and one candidate in parallel on T4x2;
- parameter ceiling: 4,000,000;
- wall-clock search budget: 14,400 seconds.

## Acceptance and decision

Mechanical acceptance must verify both T4 identities, isolated assignments,
source/cache identities, parameter counts, zero-initialized candidate return,
finite preflight gradients, role locks, atomic checkpoints, best-model and
payload hashes, unique aligned validation rows, checkpoint metadata, and a
fresh recomputation of both MAEs from stored per-row predictions.

Scientific promotion requires all of:

- candidate MAE strictly below the paired fresh control;
- paired gain at least `0.001 eV`;
- candidate/control mean epoch-time ratio at most `1.25`;
- at least 15% T4 memory reserve;
- no role, cache, source, or checkpoint acceptance failure.

A seed-42 pass records one promising candidate only. Seeds 43/44 require a
separate explicit compute-budget decision. A scientific loss closes this exact
relative-value mechanism without width, cutoff, seed, optimizer, or schedule
retry.

