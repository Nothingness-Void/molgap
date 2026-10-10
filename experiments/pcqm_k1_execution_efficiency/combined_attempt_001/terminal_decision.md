# Joint execution decision

2026-10-10; desktop NO_TRAIN. Executed source7fde33ff. Authority:
[raw report](results/result.json), [contrasts](results/contrasts.json),
[runtime](results/runtime.json), [cost](results/cost_observation.json).
Focused regression125 passed; closure/contrast tests50 passed (12 overlap), one
upstream PyG deprecation warning. No model or default adoption.

Same accepted4096 TRAIN probe and frozen initial tensors; first1920 members,
BS128, normalized Gap L1, FP32 storage, same owner forward/backward/clip guards.
Three scratch comparisons, four balanced windows, warmup3/measured12. The three
layout cases use identical precision-context machinery and omit the same explicit
barrier. Only declared AdamW flags and then matmul TF32 permission differ.
No development/official/test consumption, full fitting, remote job or payload edit.

| Case | Median ms/step | Full-state/RNG exact vs original |
|---|---:|---|
|Original FP32/unfused|87.167|Yes|
|Matched FP32 layout|85.035|Yes|
|FP32 layout + fused AdamW|62.468|No|
|Layout + fused AdamW + TF32|58.973|No|

Fused increment versus matched layout26.54%; all four reductions positive
(26.15%,24.35%,29.92%,21.05%). TF32 increment versus matched fused layout5.60%;
all four positive (5.34%,6.24%,7.83%,4.94%). Combined versus original32.34%, with
four positive windows (32.03%,30.56%,32.96%,36.66%). Original comparison includes
layout/barrier/context changes; it cannot attribute the whole saving to Tensor
Cores or fused AdamW. Percentages are measured contrasts, not added isolated gains.

Fused full-state maximum tensor difference2.6170e-7, within declared tolerance
but nonexact. Combined TF32 difference0.00101948, outside that tolerance. Neither
is an eV MAE. Every unchanged control is exact. Optimizer variants remain explicit
contract changes and never strict-FP32 execution-only admissible, even if exact.
Frozen-output TF32 drift remains owned by the prior precision diagnostic; no
new inference/quality result was obtained here. Do not assume rounding is harmless.

Layout-only median reduction2.45% in this window, with mixed paired signs
(2.78%,2.10%,-3.79%,15.61%). Earlier about7% nominations are not a universal
speed guarantee. Nonblocking transfer also does not show a stable new benefit.

Worker wall59.168s; process CPU197.797s; GPU busy unknown. Test/preparation/closure
costs excluded. Timing excludes setup/collation/equivalence comparisons and is not
native T4/A100/full-epoch/allocation calibration. No combined resume certificate.

Nominate FP32 fused+layout for the first separately authorized quality check;
TF32 is a modest extra speed candidate, not a prerequisite. Do not enable either
in a frozen/default trainer. Original recipes remain valid and unchanged.
Before prospective publication, strict custody rejected nine CRLF/Git mismatches;
the recorded commit retained existing disk bytes without changing code arithmetic.
