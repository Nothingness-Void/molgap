# Host-copy attribution

The same immutable-scalar-rich metadata shows much lower recursive copy cost
with a scalar fast path, while preserving decoded checkpoint state. Original
per-item torch import/type checking/deepcopy overhead is avoidable in this CPU
fixture. This is not a change to the K1 mathematics or training recipe.

Local publication/hash cost remains roughly0.94s at65536 rows per arm in the
middle repetition; scalar copying alone cannot remove full-history serialization
and writes. Growth of rewritten history is a source-supported concern, not a
measured native training bottleneck. Fixture timing cannot attribute A100/T4
speed parity or predict allocation savings. Real sender history length, complete
two-arm snapshot contents and native storage latency remain missing discriminators.

The shared helper is locally exact but has not replaced an active sender or
qualified native CUDA-to-CPU snapshot/resume. Do not delete traces, reduce save
frequency or change checkpoint schema based on this diagnostic. No training
replay-ready or scientific benefit is established; the outcome is NO_TRAIN.
