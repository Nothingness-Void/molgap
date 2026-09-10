# GraphState Paired Multiseed Protocol

## Question

Does the seed-42 no-attention shared GraphState result reproduce against a
freshly trained full-GPS comparator at seeds 43 and 44?

## Frozen representation and data

- Official-train-derived roles remain exactly 100,000 train and 10,000
  internal-validation rows.
- Parent graph, sparse wedge, and ETKDG distance/angle caches are unchanged.
  The accepted geometry aggregate SHA-256 is
  `3e4206fd239942ab79f9c4978cd4334f6025ba14e8f1cd1df78c20060a0d1d22`.
- Comparator: `ogb_distance_angle_triangle_edge_state_gps9`.
- Candidate: `ogb_distance_angle_triangle_edge_state_graph_state9`.
- Both retain RWSE16, persistent real-bond EdgeState64, sparse wedge state16,
  distance/angle bottom fusion, nine local ResGatedGraphConv updates, and the
  direct scalar Gap head. The candidate replaces all atom MHA with one shared
  64-dimensional graph state updated and broadcast after blocks 3, 6, and 9.
- Official validation and test-dev remain unread. The molecular-research
  server is not accessed.

## Training and execution contract

- Seeds 43 and 44 are independent confirmation stages. Seed 43 runs first;
  seed 44 is submitted only if seed 43 strictly passes its paired gate.
- Each stage is one Kaggle T4x2 task. GPU 0 trains the fresh full-GPS model and
  GPU 1 trains the fresh GraphState model in isolated Python processes.
- FP32, batch 48, AdamW, learning rate `1.6e-4`, weight decay `1e-6`, at most
  40 epochs, patience 8, normalized L1 loss, cosine schedule, identical
  per-seed shuffle generators, direct Gap prediction, and a 5.2M ceiling.
- No warm start, checkpoint transfer, prediction fusion, residual target,
  pretraining, auxiliary target, width/depth search, or optimizer search.
- Every epoch writes an atomic checkpoint. Each model emits metrics, trace,
  best model, aligned validation payload, and hashes.

## Decision rule

For each seed, GraphState must have strictly lower internal-validation Gap MAE
than the fresh same-seed full-GPS model. Final advancement additionally
requires strict improvement at seeds 42, 43, and 44 and on their arithmetic
mean. Any non-improving confirmation seed closes this exact GraphState route;
it is not retried with another seed or training setting.

A three-seed pass reaches the desktop handoff gate for a separate timing and
full-data decision. It does not authorize official validation/test-dev access,
full-data training, production changes, or molecular-research-server access.
Infrastructure-only failures may be repaired and resubmitted under the
unchanged scientific contract.
