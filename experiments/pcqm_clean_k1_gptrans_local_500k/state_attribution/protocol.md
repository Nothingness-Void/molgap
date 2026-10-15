# Same-checkpoint K1 state attribution

2026-10-15 JST. Desktop-owned follow-up to the completed candidate run, not a
new training question. User authorized local attribution. Original training
decisions and replay exclusions remain unchanged.

## Question and evidence

The [terminal attribution](../terminal_acceptance/attribution.md) leaves Live,
Raw EMA and Clean EMA weights unresolved. The selected Clean endpoint is
0.109049475eV, not a strict comparison with historical clean0.103658948eV.
The existing100K [state review](../../../docs/operations/DESKTOP_RML_CURVE_REVIEW_20261013.md)
also separates EMA averaging from BN buffers. No evidence currently establishes
slot capacity exhaustion, training-dropout harm or an optimal EMA decay.

## Frozen diagnostic

CPU only, four intra-op threads, FP32, no optimizer, gradients or training.
Maximum worker wall1200s, including data verification and analysis. Do not
automatically repeat or switch hardware after a failed/partial attempt.

Read only the accepted500K training and previously selection-used50K internal
development roles. Never read official validation/test, Common/OOD or Track A.
Reuse the immutable manifest, exact source indices, target transformation and
producer checkpoint/source hashes. No graph construction or geometry use.

For selected pass32/step124992 and final pass60/step234360, evaluate retained
Live, Raw EMA and Clean EMA on the same full50K development and fixed4096
training members. The training sample is disjoint from both calibration sets.
First reconstruct selected Clean predictions within1e-4eV of retained T4
predictions, or stop without interpreting new scores.

At the selected endpoint only, recompute BN for Live on the original16384-row
prefix, and separately for Live/EMA on a fixed uniformly sampled16384-row set.
All use128 equal batches, cumulative BN moments, dropout off and fixed learned
weights; buffers are restored after each intervention. This separates output
weight choice from buffer mismatch and the declared prefix estimator. It does
not search calibration seeds, fit predictions, tune a blend or select a model.

Rows and additional calibrations are frozen in inputs.json before execution.
Report all states, not just the most favorable. Paired row bootstrap1000/seed42
describes these consumed members, not training stochasticity or promotion.

## Interpretation boundaries

- Same-weight Raw versus Clean isolates the declared BN-buffer intervention.
- Live versus EMA after identical calibration compares retained weight states;
  it does not isolate the causal effect of training with one/two forwards.
- Prefix versus uniform estimates calibration-cohort sensitivity conditional on
  this fixed endpoint; it does not establish a universally optimal estimator.
- Same-state selected/final clean train and development metrics constrain tail
  fit, but do not uniquely diagnose overfitting or capacity limitations.
- Historical5.391meV is context across jobs/recipes. Do not force an additive
  causal decomposition against an unmatched historical reference.

Close NO_TRAIN using existing RML finalization, retain measured CPU/wall and
actual roles, no training trace or replay-ready/promotion/full-release claim.

## Reuse

FamilyTrainingState owns state validation/view restoration; k1_frozen_inference
owns factory/forward/inference; family_training_data owns fixed cache admission;
k1_bn_calibration owns buffer-only calibration; router owns row bootstrap;
research_memory.plan/finalize and frozen_diagnostic_closure own evidence closure.
Only the shared-session checkpoint translator and thin diagnostic orchestration
are new. Original training source, checkpoints and canonical records are read-only.
