# CPU loader instrumentation retry

Same question and TRAIN-only CPU scope as [attempt001](../loader_attempt_001/README.md),
not a new model experiment. Read its [failure attribution](../loader_attempt_001/attribution.md).
The first attempt lost per-check diagnostics at final rejection; no timing or
cause is inferred from that failure.

Freeze a new source/prospective before decoding. Four balanced windows, CPU4,
BS128,4096 fixed TRAIN members,300s ceiling, no models/gradients/checkpoints,
inference/development/protected roles or CUDA. No native pinning/GPU consumer.
Now retain raw reports/checks before any final rejection; join only owned CPU
worker handles before checking shutdown. Defaults and active packages unchanged.
