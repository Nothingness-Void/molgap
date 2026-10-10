# TF32 and remaining K1 execution costs

Desktop execution-only prospective sub-question, explicitly requested2026-10-10.
Prior local CUDA probe qualifies CPU layout over three scratch steps, with a
descriptive7.03% median time saving. Foreach/fused are faster but not exact; no
native T4/MAE qualification. RML contains historical FP16 investigations but no
equivalent K1 matmul-only TF32 result in the queried trajectory index. The new
discriminator is precision impact and operator cost, not repeating the old probe.

Reuse strict family K1 factory/step, frozen4096 training members and initialization,
accepted500K selected epoch48 checkpoint loader (previous profile custody), shared
execution probe, prediction helper, hashing/atomic IO and RML plan/finalize.
FP32 tensors and accumulation, physicalBS128, seed42 and unfused AdamW remain.
TF32 permits reduced-mantissa CUDA matmuls only, with no AMP, cuDNN TF32, BF16,
optimizer or scientific schedule change. The backend flags are restored on exit.
This separate precision contract cannot qualify as strict-FP32 replacement.

Allowed idle-only RTX5060 GPU worker600s cooperative ceiling, CPU4threads.
Three controlled numerical updates; four balanced timing repeats with3 warmup and
12 measured steps; original/layout/TF32/TF32+layout are compared. Predicted Gap
perturbations and descriptive training-sample MAE on4096 members use frozen trained
weights. No retraining, development/protected roles, remote submission or takeover.
Train-sample MAE is not independent generalization or retrained endpoint quality.
Weights remain immutable; repeated inference restores/rechecks state identity.
Three synchronized phase samples and two profiler steps in each of baseline,
layout and TF32+layout conditions. Instrumented attribution is not throughput.
Retain kernel names: permission to use TF32 alone is not proof of Tensor Core use.

NO_TRAIN closure: nominate only supported execution/precision questions. Defaults,
active remote payloads and promotion gates remain unchanged. No extrapolation of
RTX5060 speed to native T4/A100 or of perturbation to final training MAE.
