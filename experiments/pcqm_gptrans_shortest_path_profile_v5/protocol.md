# GPTrans exact shortest-path profiling protocol

## Question

How much of frozen GPTrans-T BS128 runtime is spent reconstructing its fixed
shortest-path tensor, and can a cached tensor reproduce the current optimizer
trajectory within the accepted `1e-7` DCU tolerance?

## Scope

- profiling only; no architecture ranking or scientific result;
- accepted fixed 500K train cache only; no development or protected role;
- unchanged 5,246,817-parameter GPTrans-T reference, seed 42, deterministic
  FP32 and physical BS128;
- representative and graph-size-tail cohorts;
- three order-randomized repeats per cohort, each with two warm-up and eight
  measured optimizer steps;
- baseline reconstructs shortest paths online with the existing cap-20 code;
- cached mode transfers the exact padded integer distance tensor and changes no
  learned operation, target, optimizer, loss, schedule or sample order.

The profile records loader, offline cache construction, dynamic path stage,
cached H2D, complete optimizer-step throughput, memory, loss/model/EMA deltas,
source/runtime/cache identities and allocation delay. A median representative
speedup of at least `1.05x` plus all equivalence checks only authorizes an exact
implementation follow-up. It does not change the V5 scientific contract or any
accepted model result.
