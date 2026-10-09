# Frozen clean-fit discriminator - 2026-10-09

User-authorized local CPU NO_TRAIN diagnostic. Owner: desktop branch
codex/exp/k1-clean-fit-generalization-500k from bb9b4ed. No submission,
optimizer, gradients, reselection, protected roles or automatic successor.

Question: how do matched clean prediction train-sample/development errors and
their gap change across mean2/consistency, selected epoch49 and last epoch60,
and original/clean-BN inference buffers? Training objective traces are not
clean train MAE and are not used to infer underfitting or overfitting.

Both retained states use frozen package b2b7539d7fbd0fbff6e674ed632ab90be00f86e8484eb6f2e8cb3ddafe06a3fd.
Selected zero-based epoch48 and last zero-based epoch59 are fixed, not selected
by this diagnostic. Use the accepted factory/inference/BN owners, FP32/noTF32,
CUDA_VISIBLE_DEVICES=-1, four CPU threads, batch128, sequential deterministic
loaders. Sample16384 train rows with numpy.default_rng(20261008).choice(500000,
16384,replace=False), preserving draw order. Full consumed development is
[500000,550000). Load accepted packed graphs; never construct geometry/cache.

Reuse accepted epoch49 full-development original/calibrated predictions and
calibrated buffers after all file/state/row/source hashes and software identity
checks. Compute clean train predictions in each matching state. For epoch60,
use its own retained parameters and original BN; independently recalibrate on
the same train sample, then evaluate train and full development. Never apply
epoch49 buffers to epoch60. Exact state restoration and frozen parameters are
required. Retain all newly computed predictions and epoch60 calibrated buffers.

The numerical worker has a total1200-second wall ceiling, including loading,
calibration, inference and saving; retain partial outputs, no retry. Record
actual process CPU and wall timestamps separately. Historical inference costs
are not charged again; preparation/testing/analysis costs are reported separately.

Report fixed-cohort train MAE, development MAE and dev-minus-train gap for all
eight endpoint/state combinations. Pair epoch60-minus49, consistency-minusmean2,
and calibrated-minusoriginal errors within each role. Paired-row95% intervals
use1000 draws/seed42; they do not measure training-seed variability. Gap changes
are descriptive, not paired across disjoint train/development rows.

The sampled train cohort was used for BN calibration: calibrated train error is
in-sample, not independent. Development was already repeatedly selection-used.
No independent holdout, causal underfit/overfit proof, capacity diagnosis,
screening/early-stop policy, model promotion or full release follows. Large or
small clean gaps alone cannot identify mechanism. The cheapest falsifier is
this bounded retained-state clean-metric measurement, not further training.

Prior evidence: paired selected-state BN improves both arms but neither material
raw/clean coefficient gain qualifies; its optimizer-scaled train trace leaves
clean train fit unmeasured. Accepted BN row attribution and endpoint-average
records describe late development change, not matched clean train fit. See
evidence_review.md and reuse.md. RML remains CONTEXT_ONLY; no fabricated strict
reference, retrospective launch rationale or training replay readiness.
