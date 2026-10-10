# Frozen native T4 speed-quality protocol

Authority: user requested a Kaggle1 original/optimized parallel pair, selected
complete100K/40ep with accuracy, and authorized running to completion rather
than a4h cost stop. Endpoint is40 epochs per arm, not unbounded training.
Native session/quota/hardware limits still apply; retain complete-epoch atomic
checkpoints on interruption, no automatic retry or continuation.

One private T4x2 Notebook, reference on device0, fused_layout on device1.
Both reuse neural_atom_k1/2, the frozen full initial tensor artifact (seed42),
fixed100K official-train-derived rows and fixed next50K internal development.
No official-valid/test/common/OOD/P8-hard access. Development is repeatedly
consumed for selection and cannot be called an independent holdout.
Source/package, initial file/tensor, manifest, row order, targets, roles and
prospective trajectories must be pinned before real-input diagnostics/training.

Original reference: unchanged single-forward normalized Gap L1, native AdamW
defaults (including foreach=None), LR4e-4, WD1e-5, gradient clip1.0, original
40ep cosine eta_min1e-6, physicalBS128/drop_last, seed42 epoch Python order.
Candidate: same model, objective, schedule/exposure and initialization, only
AdamW fused=True/foreach=False plus opt-in CPU batch layout reuse. Both keep
original epoch loaders and common clean development inference. No EMA,
two-forward objective, worker reuse, AMP, TF32, width/depth or batch changes.
FP32 deterministic execution, TF32 disabled. Fused rounding is intentionally
different; this is not a bitwise-equivalent substitution or architecture gain.

Original comparison is native-default AdamW, unlike the local speed control's
explicit foreach=False/fused=False. Local24.13% cannot predict this pair's gain.
Both assigned arms must pass their existing native real-batch repeatability,
resume-next-step, selected-state and memory/step-overhead checks before either
formally trains. Qualification failure is infrastructure, not model rejection.

Both run40ep:31240 optimizer steps and3998720 presentations per arm.
Retain best/last state, clean selected predictions with source_idx/target pins,
complete per-epoch trace and runtime/source/prospective evidence. Timing:
separate training pipeline, development and epoch pipeline; common entry
allocation includes both devices and idle/setup/qualification/cleanup. Per-arm
invocation windows are lower bounds, not total quota deduction or GPU busy time.
No automatic early stop, reference replacement, retry or threshold tuning.

Frozen acceptance: complete finite aligned endpoints, qualified runtime and
source/role/cost/exposure checks first. Report raw best and final development
MAE, paired errors and95% row-bootstrap interval (10000 resamples, seed42).
Conditional runtime nomination only if training-pipeline time decreases by
at least10% and selected candidate-minus-reference MAE's paired upper95% bound
is at most+0.001eV. Report full invocation/entry costs separately; if unavailable,
do not claim end-to-end allocation savings. A row interval is not training-seed
variance. Neither one-seed pass nor3meV research material gate authorizes
default replacement,500K/full release or production promotion.
