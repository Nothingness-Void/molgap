# Retained mechanical rejection

2026-10-10; [raw report](results/result.json) from source0e4297d1. Member order,
all batch tensor bytes and owned-worker exit checks passed. The CUDA-uninitialized
check failed, so this is not an accepted CPU-only loader benchmark. Do not retry
this immutable attempt or silently treat its95.107s run as valid native GPU timing.

The old `runtime.cuda_initialized=false` header was hardcoded, not observed.
It contradicts the actual `checks.cuda_uninitialized=false`; the rejection guard
prevented false acceptance. Future source must write observed runtime properties.

Fresh synthetic controls, without molecular data, reproduced: original family
loader workers0 leaves Python CUDA uninitialized; workers2 initializes it.
On this Windows runtime empty visibility gives `is_available=True`, count0,
initializedFalse before loader use, whereas `-1` gives availableFalse/count0/
initializedFalse. Therefore an empty string alone did not enforce CPU isolation.
The accepted prior models and frozen experiment recipes are not reinterpreted.

Source2 timings remain descriptive rejected observations only. A separate child
prospective must use reliable CPU visibility and verify availability/count/init,
without weakening the no-CUDA guard. No model/optimizer/gradient/inference or
development role was invoked. Full CPU cost and accelerator cost stay unknown.
