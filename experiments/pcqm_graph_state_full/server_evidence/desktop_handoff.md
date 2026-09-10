# GraphState desktop handoff

## Frozen candidate

- Identity: `ogb_distance_angle_triangle_edge_state_graph_state9`
- Confirmation source commit: `9068ddb82e6bdf16b841570abbff023b90c07f07`
- Geometry cache aggregate SHA-256:
  `3e4206fd239942ab79f9c4978cd4334f6025ba14e8f1cd1df78c20060a0d1d22`
- Parameter count: 3,665,809
- Passed seeds: 42, 43, and 44 against fresh same-seed full-GPS controls
- Three-seed decision: `decision.md`

## Authorized desktop action

Run one frozen A100 benchmark on official-train graphs to establish throughput,
projected full epoch time, peak memory, and at least 15% memory reserve. Keep the
architecture, direct Gap target, ETKDG construction, precision decision audit,
and immutable data roles explicit. Do not read official validation or test-dev
during this timing gate.

If and only if the benchmark projects completion within 12 A100 hours, desktop
may prepare exactly one resumable full-data training run for this frozen
candidate. Official validation remains a later one-time gate; test-dev requires
explicit user authorization after that gate. Server-side architecture discovery
must not submit a competing full-data model.
