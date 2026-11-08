# Batched-Muon K1 and GPTrans: retained 500K fusion

2026-11-08 JST. Desktop-owned Track B, CPU saved-prediction diagnostic.
The user authorizes existing 500K predictions only, followed by a recommendation
about full training. No model, checkpoint inference or new training is executed.

## Frozen question and references

Does original K1 with batched Muon add useful complementary errors to the
accepted GPTrans BondLocal+PairNorm LR1e-3 predictor? K1 selects LIVE pass54
(0.101863694448 eV); GPTrans selects EMA pass52 (0.099987377815 eV).
These are deliberately heterogeneous predictors, not an optimizer comparison.
Pin their accepted prediction bytes and exact source decisions in inputs.json.
Require unique source_idx [500000,550000), identical float32 Gap targets,
finite predictions, and the original manifest hashes before arithmetic.

Primary candidate is exactly 0.5*K1+0.5*GPTrans, in float64. Secondary fixed
weights give K1 0.25 and 0.75. They describe sensitivity and cannot replace the
primary after observing results. No weight fitting, oracle selection, routing,
calibration, model search or cohort filtering is performed.

Opportunity-cost reference is the already available equal blend of GPTrans
LR1e-3 and LR5e-4: the same two-model serving count. Previous accepted diagnostic
reports 0.097525 eV for this blend and 0.097580 eV for the older Clean-BN K1
plus GPTrans. The new K1 optimizer/execution/selected weights make this a new
question; the old result is not a reason to repeat training.

## Analysis and prioritization

Use existing load_aligned, paired-error/bootstrap, correlation and strata
helpers. Compare the fixed primary against each constituent and the two-GPTrans
blend. Use 2000 paired row resamples, seed42, 95% intervals. Report all fixed
weights, residual/error correlations, row wins and fixed physical-Gap strata
(<4,4-6,6-8,>=8 eV). These descriptions do not estimate training-seed variation.

A 3 meV point gain over the best constituent plus paired upper bound below0
is the prospective material-benefit check. Incremental value beyond the
existing two-GPTrans blend must be reported separately, including its interval.
The shared paired helper's stricter upper-bound-versus-threshold flag is
reported with its own semantics, not silently substituted for this point gate.
A result meeting only the single-model check is not sufficient to justify
training another full expert when an existing two-model alternative dominates.
Secondary weights never rescue a failed primary gate.

Outcome is NO_TRAIN with a supported recommendation, not model promotion or a
full-training authorization. A decision to launch full training remains separate.
All predictions were selected using this development role. The analysis is
descriptive on a consumed cohort, not independent held-out performance.

## Resources, roles and reuse

Only three retained development prediction files and their metadata are read.
No train data, official validation, tests, shadow roles or features are opened.
CPU only, four native threads, ten-minute arithmetic ceiling. Record actual wall
and process CPU time; GPU/queue costs are not applicable. Historical training
costs stay separate; no full training or serving-time estimate is fabricated.

Reuse molgap.research_memory.saved_analysis.load_aligned,
molgap.experiment_attribution._compare_paired_errors/_strata_table/_pearson,
atomic IO/hash owners and research_memory plan/finalize/rebuild/check. The thin
experiment wrapper contains only fixed arithmetic and output translation.
Publish the prospective trajectory before any fusion metrics. Retain compact
results and source/role/cost provenance; do not copy checkpoints or native proofs.
