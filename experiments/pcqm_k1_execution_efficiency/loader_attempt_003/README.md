# CPU loader isolation retry

Same TRAIN-only question as [attempt001](../loader_attempt_001/README.md).
[Attempt002 attribution](../loader_attempt_002/attribution.md) owns the retained
mechanical rejection and contradictory old runtime header; do not promote it.

Use CUDA visibility `-1` before Torch import, as existing CPU diagnostic owners
do, and verify unavailable/count0/uninitialized before and after collation.
Store observed runtime properties, not declarations. This is an implementation
fix to enforce the original CPU-only boundary, not relaxed acceptance.

Four balanced windows, CPU4, physical BS128,4096 fixed TRAIN members,300s limit,
original loader requested pinning but no effective CUDA pinning or GPU consumer.
Exact ordered row/batch hashes, owned worker cleanup, retained results before
rejection. No models, checkpoint decoding, gradients, optimizer, fitting,
inference/development/protected roles or remote work. Native T4/A100 remains separate.
