# K1 dropout pair result check — 2026-10-04

This is saved-artifact analysis, not completed scientific acceptance.
Exact Kaggle3 kernel ID136942701/version3 is COMPLETE; the recovery adapter
records TRAINING_COMPLETE with mechanical acceptance BLOCKED. Version1 supplied
the candidate and the control's first39epochs; version3 supplied only control
epoch40. Version2 failed before training. No successor or scale-up was submitted.

## Retained results

| Arm | Best 50K development MAE (eV) | Best epoch | Epoch40 MAE (eV) |
|---|---:|---:|---:|
| Matched two-pass dropout_mean2 | 0.1402572855657339 | 37 | 0.14028020203113556 |
| dropout_consistency2, coefficient0.1 | 0.13872850305497647 | 37 | 0.13884525001049042 |

Both traces now contain40epoch observations,31240optimizer steps and3998720
sample presentations. Recovery's first39observations exactly equal the original
control trace. The selected control model and predictions are byte-identical to
version1: completing epoch40 did not improve the selected checkpoint.
Original and resumed runtime fingerprints, row-order fingerprints and initial
state identities match; physical attempts remain separately identified.

The existing `analyze_pair.py:paired_metrics` callable verified ordered50000
rows100000–149999, finite predictions/targets and exactly aligned targets.
It gives a primary consistency gain of1.5287825meV, paired row-bootstrap95%
interval[0.6161362,2.3489377]meV;50.648%of rows improve. This numerical result
does not meet the frozen3meV material-gain gate, despite a positive lower bound.
Historical clean K1 MAE0.1412944608205557 gives contextual gain2.5659578meV,
also below3meV. That historical comparison is not the matched causal control.
Input hashes and exact metrics are in [result_check.json](result_check.json).

## Attribution and limits

The observed consistency contribution is modestly positive on this fixed cohort,
not observed module harm. From epochs35–40 both development curves improve
slightly while both minima remain at37. The missing-epoch exposure gap is resolved;
it does not explain away the subthreshold mechanism gain. These traces cannot
establish whether more training would help. Online two-dropout training MAE and
clean development MAE have different semantics; do not infer overfitting or
underfitting from their absolute gap. One seed and row bootstrap do not measure
training stochasticity. No further coefficient tuning or training is authorized
by this result check.

## Acceptance and replay blockers

1. The frozen expected development target hash is the pinned historical
   little-endian float32 digest. The generic inspector computes float64 bytes
   and returns `Development target identity mismatch`. Retained float32 targets
   reproduce the frozen historical hash; no target alteration is observed.
   Preserve the original failure and frozen expectation. An explicit, pinned
   encoding compatibility repair is still needed; diagnostic probes are not
   accepted evidence.
2. Resumed control cumulative allocated-T4 cost is missing because the first
   segment lacks its native allocation ledger. Epoch40 process wall is measured
   at187.8175seconds; recovery invocation assigned-T4 time is203.7717seconds.
   Neither substitutes for total40epoch native cost. Candidate measured training
   allocation remains9180.5245T4seconds, excluding bootstrap/queue.

Scientific acceptance is not finalized; two complete replay_pool entries have
not been established. Keep the owning branch pending, without adoption/archive
routing or a dual replay-ready claim. Analysis consumed only already-used
development predictions, with no model execution or official/test-role access.
