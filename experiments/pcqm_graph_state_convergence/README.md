# GraphState Full-Data Convergence Continuation

This experiment continues the accepted 12-epoch GraphState full-data model
without modifying its immutable source directory. The source model, optimizer
moments and RNG state are restored; only the exhausted cosine schedule is
replaced with a lower learning rate and validation-driven plateau schedule.

The continuation validates once per completed epoch, retains the lowest
official-valid MAE, and declares convergence after eight epochs without an
improvement of at least 0.00005 eV. Total epoch 80 is a technical safety
ceiling, not a runtime gate. Training remains FP32 with TF32 disabled and does
not read official test roles. Every train shard produces an atomic checkpoint,
so four-hour IMS segments resume the same run.

This changes the model-selection protocol and must be reported separately from
the frozen 12-epoch result. It is not evidence from a scratch architecture
comparison and cannot modify the production registry automatically.
