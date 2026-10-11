# Native K1 initialized-layout100K pair

2026-10-12 JST. User requested an original/optimized K1 double-arm submission,
then desktop shutdown after verified submission. Same desktop execution-cost
question/owner; fresh variant identities, not a retry or rewrite of fused AdamW.

One private Kaggle1 T4x2 kernel. GPU0 reference, GPU1 initialized-layout;
fixed100K official-train-derived rows0..99999, fixed50K internal development
rows100000..149999. These development rows are repeatedly used for selection,
not independent held-out evidence. Seed42 full frozen initial state, FP32/noTF32,
BS128/drop_last, original epoch loaders, single normalized Gap L1 forward,
native AdamW defaults, clipping1, LR4e-4/weight-decay1e-5,40epoch cosine/eta1e-6,
best-development-live selection,40epochs/31240steps/3998720presentations per arm.
No EMA, changed model, extra view, loader reuse, official validation/test,
Common/OOD/P8-hard, early stop, automatic retry/continuation or adoption.

Candidate only: CPU layout reused by the existing one-slot K1 mixer; safety
fill disabled in the scoped TRAIN step. Deterministic algorithms and actual
zero/parameter initialization remain intact. Clean evaluation keeps safety fill.
Both use Kaggle native Torch2.11.0; no dependency replacement. Local Torch2.7.1
scratch equivalence is supporting engineering evidence, not2.11 qualification.
The observed runtime must match the candidate recipe. Existing all-arm barrier
requires exact candidate/reference native step fingerprints (gradients, AdamW
moments, model/BN, metrics and RNG), repeatability and exact checkpoint resume.
Any qualification failure prevents BOTH formal training arms; retain failure
and actual cost, do not relax a check or submit an automatic successor.

One authorized run to its40epoch endpoint or the native platform/session limit,
whichever occurs first; no newly invented hourly cutoff. Allocation/quota cost
is unknown until observed. Count wall, allocated T4-device time, setup/evaluation
and queue separately. Epoch train-pipeline timing is not GPU-busy time.
Both GPUs share host IO/CPU; allocation saving has no reference-only comparator.

Frozen nomination gates: >=10% complete TRAIN-pipeline reduction and paired
selected MAE candidate-minus-reference95% upper bound <=+0.001eV (10000 row
bootstraps, seed42). Report epoch/final MAE and exact-state differences, including
numerical drift even if noninferiority passes. One seed/row bootstrap cannot
establish training stochasticity. No gate changes or default/500K/full release.

Reuse registered prepare-workflow, bootstrap, per-arm family trainers, atomic
best/last checkpoints, trace/timing/predictions and family-output manifests.
Kaggle session outputs retain retrievable source-bound artifacts after desktop
shutdown. Prospective RML must precede submission. Actual remote identity is
bound from API response; no local slot trigger, monitoring or server takeover.

See [evidence review](evidence_review.md), [user release](user_release.json),
and [engineering report](../execution_audit/deep_profile_20261012.md).
