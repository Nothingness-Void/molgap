# K1 Batch Renorm mean-correction contrast

Prospective desktop question, 2026-10-24. Owner:
`codex/exp/k1-batch-renorm-100k`, created from verified desktop283dc4db.
The user authorizes a new100K Kaggle1 pair using the repaired shared workflow.
[Evidence review](evidence_review.md) owns justification; [closure](closure.md)
owns acceptance and attribution. No original-baseline retraining is requested.

## Intervention

Replace only local_blocks.0..8.norm1/norm3 (18 BatchNorm1d modules).
Preserve position, feature axis, affine initialization, eps, momentum, original
running buffers, root/residual branches, global mixer LayerNorm and dropout.
For channel-wise batch mean b and biased variance v, and PRE-update running
mean M and running variance V, define detached corrections:
`r = clamp(sqrt(v+eps)/sqrt(V+eps), 1/r_cap, r_cap)` and
`d = clamp((b-M)/sqrt(V+eps), -d_cap, d_cap)`.
Apply corrections to batch-standardized activations before learned gamma/beta.

- brn_scale: r correction, d=0.
- brn_affine: the same r correction plus d correction.

No new trainable parameters. This is a variance-EMA Batch Renorm adaptation:
running_var retains PyTorch's unbiased update convention, rather than claiming
the paper's exact standard-deviation EMA implementation.
Existing integer num_batches_tracked controls correction progress. Training
forward t is one-based; p=clamp((t-3905)/3905,0,1), r_cap=1+2p,
d_cap=5p. First5passes have identity corrections; caps reach full values after10.
Separate affine arithmetic may differ in last bits from a fused original BN.
Both arms initially compute the same function. One training forward per update;
no mean2, training-mode validation, extra training probes or counter-free schedule.
Evaluation uses running buffers and never advances the correction counter.
Explicit cumulative BN calibration (momentum=None) uses ordinary BN statistics
updates without corrections. Its independent model copy cannot alter training.

## Frozen recipe and roles

Accepted fixed100K PCQM4Mv2 pure2D mirror, training rows[0,100000), internal
development[100000,150000). Shared K1/3 accelerated family trainer, seed42,
transported full original initial state, BS128/drop_last, FP32/noTF32,
AdamW4e-4, WD1e-5, betas0.9/0.999, eps1e-8, gradient clip1,
cosine40passes/minLR1e-6, no optimizer warmup. BRN correction warmup is a
separate frozen mechanism. Normalized Gap L1, fixed training mean/sample std.
EMA0.999 tracked; best-development LIVE selected. Epoch-only standard logs.
Per arm:40passes,31240updates,3998720gradient presentations;50K predictions.
Official validation/test/common/OOD/P8-hard are sealed. No new data acquisition.

## Decision

Primary: selected Live MAE(brn_affine)-MAE(brn_scale), on aligned retained rows.
2000 paired row-bootstrap draws, seed42. At least3meV improvement AND upper95%
interval below zero nominates further review only. Otherwise close this mean-
correction contrast negative under contract. This is not a test of BRN versus
unchanged BN; historical endpoints cannot establish that causal claim.
No adoption,500K/full, extra seed, retry or continuation is released automatically.
Row uncertainty does not measure training stochasticity.

## Attribution discriminator after training

Before another module, analyze full train/development/LR/exposure curves,
selected and final endpoints, tail/signed row errors, complete native costs,
and correction-counter consistency. Train online normalized loss is not a
matched fixed-cohort eV generalization measurement.

After strict training acceptance, perform a separately recorded NO_TRAIN
diagnostic using the retained selected models: original versus one cumulative
Clean-BN pass on the SAME first16384 training members, BS128, dropout off,
then predict the same50K selection-used development rows. Use existing
FamilyTrainingState/CalibrationPolicy and recipe-bound family reconstruction.
Keep weights, non-BN buffers, source order, target encoding and original
predictions fixed; preserve the original selected states. No labels are used
for calibration. Predeclare all four resulting MAEs and the paired contrast
before/after calibration; never select the best posthoc state for promotion.
A disappearing advantage is compatible with buffer mediation; persistence is
compatible with learned-function differences, not proof of a unique weight cause.
Missing diagnostics remain explicitly incomplete, not a fabricated explanation.
Diagnostic authority: local CPU or existing desktop GPU, one pass/model,
30minutes wall ceiling total, measured separately; no Colab/cloud allocation.

## Execution and strict closure

One private Kaggle1 T4x2 slot, one arm/device.14400s allocation ceiling,
180s cleanup reserve,8nativeT4device-hours ceiling total. Expected per-arm
2.0device/wall hours is an estimate based on the MsgNorm pair (~1.6h training),
not measured BRN performance. Queue/CPU unknown remain unknown.
Shared native repeat/serialized-resume preflight precedes training. Its two-step
initial probe covers the BN warmup; correction-active behavior additionally
requires the targeted local serialized-resume check and the complete finite
training trace. Any actual remote continuation additionally requires its own
retained resume proof. Do not imply the startup probe executed3906updates or
that a continuously running job exercised a correction-active remote restart.

Reuse scientific input -> submit_workflow -> native workers -> exact retrieval
-> accept_run --strict -> canonical terminal/RML. Freeze a committed source
including the completed storage repair before packaging. Download only planned
manifest-bound required objects, reusing hash-verified local objects; no full
output download or history-chunk retrieval. All bulk paths are on D:.
Each arm must independently reach complete replay with no exclusions. Scheduler
COMPLETE, passing RML freeze, or a positive endpoint alone is insufficient.
User-directed local Luna-max monitoring may check every30minutes while desktop
is online, silently for healthy unchanged states. No server takeover.
