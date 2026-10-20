# Frozen Research Scope

Authorization: user requested preparation and submission to Kaggle1 of the
two approved 100K pairs; K1 40 passes and GPTrans 60 passes, no user-imposed
wall deadline. No other account, scale, seed or protected role is released.

## Controlled Comparisons

Seed42; the accepted fixed100K training range [0,100000), internal development
range [100000,150000), physical batch128, FP32, no TF32 or accumulation, frozen
per-family sampler, target transform, LR/WD/schedule and EMA selection. Each
pair loads the same transported full base initialization, not separately
constructed remote random weights. Same physical job, one arm per T4.

K1: single-forward supervised L1, AdamW LR0.0004/WD0.00001, cosine40, original
post-residual placement, Live selection and matched step-wise EMA0.999 recorded
in both arms. Primary outcome is best internal-development Live MAE, so BN
calibration and EMA selection do not confound the norm comparison. Only
`local_blocks.*.norm1/norm3` change BN to LN. Convolution, slot, dropout,
readout, residual order and exposure stay unchanged. No Clean-BN pass.

GPTrans: retained degree-scaled BondLocal base, AdamW LR0.001/WD0.05,
four-pass warmup then cosine60, step EMA0.999 and best-development EMA
selection. Candidate adds an initially zero learned 4-to-32 projection to
pair state once at the first attention slot. No second dense-input builder,
coordinate construction or change to data cache. Resistance uses an
unweighted undirected real-atom Laplacian; virtual, padded and disconnected
pairs are masked. Added channels: effective resistance, pseudoinverse diagonal
sum, absolute diagonal difference, and `(1-R).clamp_min(0)*adjacency`.
These are topology descriptors, not conjugation or electron-density labels.

## Cost, Safety and Evidence

No local wall watchdog is requested (`allocation_wall_limit_seconds=null`).
Kaggle's documented GPU session ceiling is 12 hours per notebook; the positive
submission quota reservation is 12 allocated-notebook hours per notebook,
not a new user deadline or a claim of 12 T4-device hours. Up to 24 T4-device
hours per notebook is only a physical upper-bound estimate. Platform quota
receipts remain the authority for actual debits. Session limit reference:
https://www.kaggle.com/docs/notebooks#technical-specifications

Native all-arm preflight must pass before formal training. Retain immutable
source/config/init/manifest identity, atomic best/last and complete-state
chunks, actual job/version, role access, LR/step/sample trace and measured cost.
Use the existing preflight/runtime guards; no synthetic result grants native
qualification. Stop on non-finite predictions, identity mismatch or native
qualification failure. No automatic retry, continuation or early-screen rule.

Shared operator profiling: one 16-step window after 120 warmup steps in every
arm; same instrumentation. Record extra pinv cost and end-to-end time. Reject
an uneconomic module even if its parameter count is small; do not infer cost
from parameters or extrapolate tiny CPU batches to T4 throughput.

## Terminal Decision

Retrieve finite row-aligned predictions and full curves; verify exact exposure,
roles, target transform, initial/source hashes, native runtime and checkpoint
custody. Compare each candidate only to its same-notebook control. Report
paired bootstrap interval and per-size/ring/connectivity cohorts where existing
analysis supports them. Row bootstrap is not seed uncertainty. Internal dev
has been repeatedly used for model selection, not an independent holdout.

Retain the current material gate of 0.003eV; a smaller gain is a weak signal,
not an automatic advance. No threshold recalibration from this single seed.
Even a positive result grants no automatic 500K/full training, READY label,
production/registry change or Track A role consumption. Finish mechanical,
scientific, cost and replay-readiness decisions separately, then attribute
failure modes before proposing another mechanism.
