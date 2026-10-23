# K1 message-amplitude learning, fixed100K

Prospective desktop question, 2026-10-23. User authorizes a formal100K two-arm
Kaggle1 experiment using the existing infrastructure and strict replay closure.
Owner: codex/exp/k1-message-norm-100k, based on desktop904f2aea.
The [evidence review](evidence_review.md) owns route selection and prior results.

## Question and intervention

Does learning relative neighbor-message amplitude improve K1 over fixed-amplitude
MsgNorm under an otherwise identical recipe? Both arms are new interventions;
neither retrains unmodified K1. This pair does not establish absolute superiority
over original K1 or reproduce the full DeeperGCN architecture.

- msgnorm_fixed: each of nine local neighbor aggregates becomes
  m' = ||x_i||_2 * m / max(||m||_2, 1e-7), with fixed scale1.
- msgnorm_learned: identical operation with one independently learned scalar
  s per layer, initialized1. Nine extra trainable scalars; the same AdamW group
  and weight decay apply. The unconstrained scalar follows the original MsgNorm
  parameterization; negative values must be reported rather than hidden.

The original receiver input x_i is captured at the convolution input. Normalize
only the aggregated neighbor message before root/skip addition. Preserve local
BN, global mixer LN, residuals, architecture width, dropout and all other weights.
Zero aggregate remains zero; no 3D geometry, graph sampling or protected roles.
Both variants have identical initial forward values. One seed cannot estimate
training stochasticity. No existing evidence establishes pathological message
scale; this is a bounded mechanism test of a plausible hypothesis.

## Frozen training and roles

Reuse shared K1/3 trainer and the retained post-BN40-pass recipe. Fixed accepted
OGB PCQM4Mv2 graph mirror: train rows[0,100000), internal development[100000,150000).
Seed42, transported original initialization tensors, single forward, physical
batch128/drop_last, FP32/noTF32, native AdamW4e-4, weight_decay1e-5, betas.9/.999,
epsilon1e-8, gradient clip1.0. Existing cosine schedule40units/minimum1e-6,
no warmup; normalized Gap L1, fixed train target mean/sample standard deviation.
EMA.999 is tracked; best internal-development Live MAE selects the endpoint.
40 completed passes,31240 optimizer updates,3998720 gradient presentations per
arm;50K aligned finite development predictions. Standard epoch-only logging.
Official validation, test-dev/test-challenge and common/OOD/P8-hard remain sealed.

## Decision and attribution

Primary contrast: learned minus fixed selected Live MAE on exactly aligned rows.
Freeze2000 paired-row bootstrap resamples, seed42; negative delta means improvement.
A point improvement>=3meV and upper95% interval<0 nominates further review only.
Otherwise close negative under this material gate; retain subthreshold effects.
The fixed arm is a new scientific comparator, not a previously accepted baseline.
Both require independent native qualification and replay entries regardless of
the scientific sign. No automatic adoption,500K/full, retry or continuation.

Use retained full development/LR/exposure traces, matched10/20/30/40 endpoints,
selected predictions, final and selected scale parameters, and native costs for
attribution. Report learned-scale deviation and sign at selected/final checkpoints.
Scale movement is descriptive, not proof of a causal layer-specific effect.
No extra diagnostic forward is introduced. Missing matched training-cohort
metrics or per-layer raw message norms remain insufficient_evidence; do not
infer underfitting, exposure shortage or scale pathology from one endpoint.

## Resource and closure

One private Kaggle1 T4x2 job, one arm per device. Ceiling14400 entry-wall seconds,
180seconds cleanup reserve,8 allocated native T4-device hours total. Expected
1.8hours per arm is an estimate, not measured cost. Queue/CPU unknown remain unknown.
Existing native repeated/serialized-resume preflight precedes formal training.
Keep shared atomic selected/last checkpoints and bounded resumable chunks.

Use the existing scientific-input submit_workflow entry, exact returned job/version,
source roundtrip and receipts. Retain complete runtime proof dependencies, trace
context, selected weights/predictions, final resume state, roles and allocation
records through existing owners. On terminal state use strict exact-file retrieval
and accept_run --strict. No hand-built output/terminal schemas or bulk downloads.
The [closure runbook](closure.md) owns the supported operations and acceptance exit.
Both replay entries must be complete with no exclusions before reporting replay-ready.
Missing evidence blocks that claim; a scheduler COMPLETE or RML frozen check alone
does not establish it. Desktop owns offline reconciliation; no automatic monitor.
