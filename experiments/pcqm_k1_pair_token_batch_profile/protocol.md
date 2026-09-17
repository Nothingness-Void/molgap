# PairToken Kunshan profiling-only batch sweep

## Question

Where does the frozen PairToken 500K implementation spend wall time on the
Kunshan DCU, and which physical batch sizes are technically efficient and leave
at least 15% device-memory reserve?

This is infrastructure evidence, not a model experiment. It cannot change the
V5 scientific contract, rank a model, produce a checkpoint, or authorize a
later training run.

## Frozen inputs

- Architecture: unchanged 3,681,665-parameter PairToken 500K implementation.
- Data: accepted fixed cross-platform 500K cache; train role only.
- Precision: deterministic FP32, TF32 disabled.
- Optimizer path: unfused AdamW, weight decay `1e-5`, clipping at `1.0`.
- Candidate physical batches: `64, 128, 256, 512, 1024, 2048`.
- Cohorts: one seeded representative cohort and one largest-graph tail cohort.

Every batch starts from the same seed and a fresh model/optimizer. Warm-up uses
4,096 graphs. Timed work uses 16,384 representative graphs and 4,096 tail
graphs. The same cohort indices are reused for every batch.

## Measurements

For every batch and cohort, record load/collate, H2D, forward/loss,
backward/optimizer, end-to-end throughput, peak allocated/reserved memory, and
OOM status. Also record queue allocation duration, graph-size quantiles,
runtime/source/cache fingerprints, and profiling-output hash time.

Development, official validation, test-dev, and test-challenge are not read.
No validation, checkpoint, model bundle, or scientific MAE is produced.

## Interpretation

The fastest configuration with at least 15% memory reserve is only a candidate
for a separately frozen future contract study. BS128 remains authoritative for
all current V4/V5 scientific comparisons. Differences caused by batch size are
not model improvements.

