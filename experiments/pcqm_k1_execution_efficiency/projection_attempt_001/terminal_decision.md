# Projection diagnostic decision

2026-10-10; desktop-owned, NO_TRAIN, no model promotion or payload change.
Source commit: `94ceaeb0`; exact source bytes are retained in `executed_source/`.
Authority: [result](results/result.json), [runtime](results/runtime.json),
[cost](results/cost_observation.json), [prospective](rml/trajectory.json).

RTX 5060, torch2.7.1+cu128, PyG2.7.0, FP32/no-TF32, physical BS128,
unfused AdamW. Three scratch-update equivalence checks and four balanced timing
windows per case; each timing uses three warmup and twelve measured updates.
Decoded4096 accepted training rows; executed first1920, no development/test roles.
Focused synthetic/regression suite:76 passed, one upstream deprecation warning.

| Case | Matched comparator | Median ms/step | Time reduction | Full-state exact |
|---|---|---:|---:|---|
| Project before edge gather | gather_control80.092 |79.086|1.256%|No|
| Project before dense pack | cpu_layout72.610 |74.760|-2.961%|No|
| Existing CPU layout control | baseline78.034 |72.610|6.951%|Yes|

Gather paired reductions: -0.154%,0.269%,3.244%,-2.869%; not a stable benefit.
Full-state maximum absolute error8.874e-6, with tolerance failures in parameter
state. Pack paired reductions: -7.893%,-0.289%,0.668%,-1.728%; three windows slower.
Maximum absolute error1.192e-7, within the declared tolerance but not exact.
These errors aggregate gradients/parameters/optimizer state; they are not eV MAEs.
All unchanged controls were exact. Existing layout paired savings were positive
in all four windows (6.784%,6.488%,4.137%,7.117%).

Reject both new hooks for direct strict-FP32 replacement; pack also lacks a cost
benefit. This is execution diagnostic rejection, not a learned-model conclusion.
Default branches remain unchanged and the hooks remain opt-in on this owner.
No checkpoint continuation or retrained/generalization quality was qualified for
these new hooks. Do not combine separate attempt medians into a cumulative saving.

Worker wall61.037s; process CPU210.984s across four threads. GPU busy time is
unknown. Timing includes context/transfer/guards/RNG, but excludes model setup,
data loading and host-side state comparison. No full-epoch extrapolation or
T4/A100 qualification. The entire diagnostic is not a training-replay-ready run.
No successor training is released by this result.
