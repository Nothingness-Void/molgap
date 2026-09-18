# Protocol

## Question

Which contract-preserving Kaggle3 T4 input-pipeline configuration gives the
highest stable optimizer-step throughput for GPTrans-T `pair_update_norm`?

## Frozen identity

- Model: GPTrans-T 12x256, pair width 32, `pair_update_norm`.
- Data: first 50K train shard from the accepted fixed 500K V5 cache.
- Batch: 128 graphs on one T4.
- Precision: FP32 with TF32 disabled.
- Optimizer: AdamW, `lr=4e-4`, `weight_decay=1e-5`, unfused,
  `foreach=False`.
- Each case: 10 warm-up steps and 80 measured steps.
- Sealed roles: development, official validation, test-dev, and challenge are
  not read.

## Runtime-only interventions

1. `workers=0`, no persistent workers.
2. `workers=2`, `prefetch_factor=2`, persistent workers.
3. `workers=4`, `prefetch_factor=2`, persistent workers.
4. `workers=4`, `prefetch_factor=4`, persistent workers.

Cases run sequentially on one T4 so they do not contend for Kaggle CPU or disk.
The second allocated T4 remains unused because multi-GPU execution would alter
the frozen physical/global batch contract.

## Selection

Choose the fastest case only if all steps are finite, physical batch remains
128, peak reserved memory is at most 85% of device memory, and measured
throughput exceeds the zero-worker case. Report the measured epoch and 60-epoch
projections as planning estimates, not scientific evidence.

