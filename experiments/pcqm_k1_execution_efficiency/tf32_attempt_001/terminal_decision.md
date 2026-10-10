# TF32 terminal decision - 2026-10-10

Outcome NO_TRAIN. The worker completed in69.504s under600s; all seven scratch
cases contain three numerical-update comparisons and four timing windows.
Unchanged FP32 controls are exact. Frozen selected500K weights/BN state stayed
unchanged; repeated strict-FP32 prediction on4096 train members is exact.
No development/protected roles, scientific training, production change or remote
submission. Original executed bytes are retained separately from closure repairs.

| Execution | Median step seconds | Exact state/RNG vs strict control |
|---|---:|---|
| Original strict FP32 |0.077901|Yes|
| CPU layout, strict FP32 |0.074598|Yes|
| TF32 permission, original layout |0.074818|No|
| TF32 permission + CPU layout |0.069951|No|

The precision-isolated comparison is TF32+layout versus strict-FP32+layout:
6.23% median time reduction; all four paired windows positive
(6.00%,5.28%,6.79%,11.15%). Both use the same layout context and no explicit
redundant barrier. Original-layout TF32 also changes the barrier relative to
baseline; its total baseline difference must not be attributed solely to TF32.
Combined layout+TF32 is10.20% faster in step-time terms than this attempt's
baseline, not an extrapolated full training saving. Do not combine medians from
the earlier GPU attempt: its baseline was slower under a different execution window.

Frozen trained-state numerical perturbation,4096 sampled TRAIN molecules:

| Metric | eV |
|---|---:|
| Mean absolute prediction difference |0.000116172|
| P95 absolute difference |0.000365257|
| Maximum absolute difference |0.000891685|
| Strict-FP32 training-sample MAE |0.06217623994|
| TF32 training-sample MAE |0.06217625682|
| Descriptive MAE change |+0.00000001688|

Near-zero sample MAE change does not mean predictions are unchanged; signed
error perturbations cancel in aggregate. No independent-domain or retrained
endpoint accuracy inference. Scratch updates differ, with maximum observed
full-comparison tensor error0.0010178; TF32 cannot enter strict-FP32 admission.

Synchronized phase attribution (instrumented, not throughput) supports substantial
forward/backward and unfused optimizer cost. Layout median phases were24.84ms
forward+loss,24.27ms backward,18.69ms optimizer,3.58ms clip,0.75ms layout/H2D;
TF32+layout forward+loss was20.63ms. Three phase samples are not robust native cost
calibration; summing medians or nested profiler totals into precise percentages
is inappropriate.

CUPTI initialization reported INVALID_DEVICE in this Windows CUDA12.8 runtime.
GPU kernel arrays are empty and profiler device self-times are zero (unavailable,
not zero GPU cost). CPU operator counts and synchronized wall phases remain
retained, but Tensor Core kernel dispatch is not directly qualified. No claim that
Tensor Cores accelerate K1 by a particular factor; only TF32-permission timings.

Nominate CPU layout as the conservative exact execution addon. TF32 is a separate
precision candidate with modest incremental local speed and small frozen-output
perturbation, not an adopted training recipe. Final retrained MAE/native hardware
qualification remains missing. No automatic100K/500K/full successor.
