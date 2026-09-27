# Frozen relation-dependency diagnostic

## Authority and question — September 27, 2026

The user authorized diagnosis of the three accepted relation-resolution models
and identification of genuinely useful modules. This releases one bounded
Kaggle2 NO_TRAIN job, not a retry, new architecture, schedule extension, seed,
500K training, full run or promotion. The preceding
[portability decision](../audit/decision.md) remains unchanged.

All three selected 100K checkpoints helped the reused original development
role but harmed fixed500K internal development. Matched training curves showed
shrinking relative gains, not a sustained late advantage. A post-hoc saved-row
analysis found regression concentrated in low-conjugation molecules despite
benefits in the original role's corresponding coarse group. More training
repetitions, insufficient unique-structure coverage, branch interference and
selection optimism are different hypotheses. This diagnostic tests branch
dependence/interference only; it cannot prove a different data-scale asymptote.

## Frozen interventions

Every mode uses its own accepted checkpoint, without tensor updates:

| Mode | Conditions | Targeted question |
|---|---|---|
| Receiver-pair | full, half, off, common_return | Does receiver-specific redistribution help relative to its graph-mean update? |
| Triplet-aggregate | full, half, off, triplet_off | Does the aggregate contribution help after preserving its final normalization? |
| RRWP-pair | full, half, off, rrwp_off | Does the learned walk-relative contribution help? |

`half` applies 0.5 to the complete added layer-6 update; `off` returns the
unchanged hidden state at that point. `common_return` broadcasts the original
update's within-molecule mean, preserving that mean but deleting receiver
differences. `triplet_off` removes the learned aggregate projection including
its bias, retains `triplet_norm(pair)` and all receiver processing.
`rrwp_off` zeros the entire RRWP projection including bias, without changing
the pair normalization or downstream receiver network. No coefficient search,
router, per-molecule gate or fitted combination is authorized.

An off candidate is NOT the original K1: its remaining weights co-adapted during
training. A harmful knockout proves dependence, not net training benefit.
Between-checkpoint Receiver/Triplet/RRWP comparisons reuse the previous matched
training evidence; frozen knockouts cannot replace from-scratch controls.

## Data and execution

Use the original factory/source archive and accepted best-model/target-transform
hashes. Consume only accepted fixed100K internal dev `[100000,150000)` and
fixed500K internal dev `[500000,550000)`, exactly 50,000 rows each. No training
labels, official validation, test-dev, test-challenge, geometry or teacher.
Reproduce the full mode on each role against its accepted payload before any
intervention there; maximum absolute discrepancy must be <=0.0001 eV. Reuse
accepted K1 predictions; do not infer or retrain K1. FP32, no TF32, physical
BS128 and deterministic evaluation remain unchanged.

Request T4x2; independently isolate frozen model workers on one visible device
each. Three models share at most two workers; fewer allocated GPUs imply a
sequential diagnostic, never a larger batch or precision change. Hard cap is
5,400 summed allocated-device seconds including setup and idle devices, and
5,400 wall seconds. Expected allocation is <=1 device-hour, not guaranteed.
Atomic 5,000-row prediction chunks and progress must survive failures.
Record actual GPU/runtime, wall and process CPU times, peak reserved memory,
all role events and checkpoint immutability; unknown queue/billing stays unknown.

## Analysis frozen before submission

Report every condition, not just a selected favorable result. For both roles,
compute aligned MAE, condition-minus-full and condition-minus-K1 deltas.
Use paired row bootstrap with 2,000 replicates; intervals are descriptive,
not seed uncertainty or multiplicity-adjusted promotion tests.
Retain per-row update/hidden RMS, receiver dispersion and assignment entropy.

Use fixed conjugated-bond-fraction groups: low <0.27272728085517883,
high >=0.7333333492279053, middle otherwise. These boundaries came from an
already examined role; they are diagnostic, not untouched confirmation.
Also report existing atom/bond/ring/RWSE descriptors without fitting thresholds.

Interpretation:

- A knockout improving low-conjugation rows but hurting high-conjugation rows
  supports heterogeneous dependence/interference, not a deployable router.
- Half improving both roles while full loses supports excess update strength
  in this trained checkpoint, not a new optimized model claim.
- A targeted knockout hurting both roles supports an active helpful computation
  inside that network. Net portable architecture improvement additionally
  requires matched parent/baseline benefit, absent from the prior audit.
- Mixed/null effects stay inconclusive. No module is declared a new winner just
  because deleting it hurts, or because a post-hoc coefficient looks better.

## Evidence and stop

Freeze a separate prospective NO_TRAIN trajectory, source/input release,
budget and roles before push. Independently accept all saved chunk hashes,
row/target identities, full reproduction, state immutability and native cost.
Finalize distinct NO_TRAIN RML evidence, then validate/rebuild/check. This is
not an optimizer-trace/training-prefix replay entry and does not alter the
three existing replay-ready training records. Terminal analysis returns to A;
B remains silent while healthy and cannot retry or submit successors.
No automatic training follows any result.
