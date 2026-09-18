# GPTrans full training-step profiling protocol

## Question

Which parts of the frozen GPTrans-T FP32 BS128 execution dominate wall time,
and can any execution-only change reduce end-to-end time while preserving the
accepted DCU numerical tolerance?

## Scope

- profiling only; no architecture ranking or scientific result;
- accepted fixed 500K **train** cache only; no development or protected role;
- unchanged 5,246,817-parameter GPTrans-T reference, seed 42, deterministic
  FP32, physical batch 128 and the frozen unfused AdamW mathematical contract;
- representative and graph-size-tail cohorts;
- three order-randomized repeats, each with two warm-up and eight measured
  optimizer steps;
- a synchronized baseline decomposition of loader/collation, H2D, zero-grad,
  forward, target/loss, loss-finite synchronization, backward, gradient clip,
  gradient-finite synchronization, AdamW and EMA;
- train-role proxies for no-grad evaluation and atomic checkpoint persistence.

The execution variants isolate three implementation questions:

1. perform host-synchronizing finite checks every 50 steps rather than every
   step while retaining finite checking and the same clipping operation;
2. use the tensor-list AdamW implementation without changing AdamW fields;
3. update floating EMA tensors with tensor-list operations.

A combined variant is measured only as profiling evidence. No change is
eligible unless its individual variant stays within `1e-7` for loss, model,
optimizer and EMA state, improves median representative end-to-end throughput
by at least `1.05x`, and does not reduce graph-size-tail throughput below
`0.98x`. Passing only authorizes a separately reviewed implementation repair;
it never changes the scientific model, score, or accepted training record.

