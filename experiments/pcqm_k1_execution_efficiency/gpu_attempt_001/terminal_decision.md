# RTX 5060 terminal decision - 2026-10-10

Outcome NO_TRAIN. The prospective CUDA scratch diagnostic and atomic checkpoint
continuation completed under the 600s cooperative worker bound. Worker wall
67.066s, process CPU207.406s; probe body54.513s. Busy-device time was not measured.
No development/protected role, model adoption, production/default change or remote
submission. Three equivalence updates, four balanced timing windows of twelve
steps after three warmup steps; same FP32/no-TF32/BS128 ordered input.

| Case | Median sec/step | Exact full state/RNG | Median time reduction |
|---|---:|---|---:|
| Baseline |0.089524|Yes|0%|
| Nonblocking transfer |0.084973|Yes|5.08%|
| Remove redundant barrier |0.086675|Yes|3.19%|
| Transfer + barrier |0.087432|Yes|2.34%|
| CPU topology layout |0.083235|Yes|7.03%|
| Foreach AdamW |0.069118|No|22.79%|
| Fused AdamW |0.067037|No|25.12%|

CPU layout has four positive paired savings:7.16%,10.93%,7.41%,2.14%.
Nonblocking transfer also has four positive savings:3.22%,5.15%,10.38%,1.34%.
These are local nominations, not a calibrated variance/promotion threshold.
Barrier removal and combined transfer/barrier have mixed signs; do not treat
their descriptive medians as reliable gains or add independent gains together.

CPU layout preserves loss, pre/post-clip gradients, parameters, BN buffers,
optimizer state and RNG exactly in the short CUDA test. Uninterrupted versus
resumed original-to-layout continuation also matches exactly, with313 CPU AdamW
step states verified. Checkpoint keys remain unchanged.

Foreach and fused fail exactness (maximum tensor difference2.384e-7/2.617e-7)
and are optimizer-contract changes. Their speed merits separate numerical/quality
qualification, not silent adoption or a prediction of MAE harm/benefit.

This uses RTX5060/torch2.7.1/PyG2.7.0 and pre-collated training batches, not the
native T4/A100 stack or full500K data pipeline. Nominate opt-in CPU layout for
reviewed native qualification; keep defaults and current remote source unchanged.
Accepted diagnostic evidence/reusable code are eligible for the explicit desktop
non-promotion integration route; no native training successor is released.
