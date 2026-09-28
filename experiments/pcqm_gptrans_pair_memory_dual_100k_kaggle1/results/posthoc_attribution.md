# Pair-memory saved-artifact attribution — 2026-09-28

This is a descriptive analysis of the accepted Kaggle1 pair, using the already
consumed 50,000-row internal-development role. The frozen decision remains in
`../decision.md`; this analysis does not change its gate or authorize training.
The machine-readable calculations and SHA256 of every input are in
`posthoc_attribution.json`. Both arms' accepted best-model, resume-checkpoint,
prediction and trace hashes were rechecked against mechanical acceptance.
The existing paired-analysis function checked ordered source indices, equal
targets and finite predictions. No checkpoint was deserialized, model was run,
remote job was launched, or protected role was accessed for this analysis.

## Observed effect

| Same-job arm | Selected internal-development MAE | Measured T4 device hours |
| --- | ---: | ---: |
| `memory_value` | 0.155583367 eV | 2.693759 |
| `memory_message` | 0.159906685 eV | 2.694155 |

The candidate is **4.323316 meV worse** on aligned rows. The frozen 10,000
resample row-bootstrap interval for candidate-minus-reference absolute error
is **+3.266 to +5.337 meV**. Only 48.64% of rows favor the candidate. This
rules out a material gain under the frozen same-job gate; the bootstrap does
not measure training-seed variability.

The candidate predicts 17.324 meV higher on average. Its mean signed error
is -10.532 meV versus -27.857 meV for the reference, so a simple global
underprediction shift does not explain the candidate's worse MAE. Fitting a
separate median offset to each arm on these *same development labels* leaves
the candidate 6.522 meV worse. That is only a diagnostic; the fitted scores
are not held-out evaluation or a proposed calibration. In descriptive target
strata, candidate-minus-reference MAE is -0.981 meV below 4 eV (3,725 rows),
+4.758 meV from 4 to 6 eV (34,837), +3.419 meV from 6 to 8 eV (10,451),
and +18.581 meV at or above 8 eV (987). The sparse tail does not justify a
router or subgroup rule.

## Training behavior and causal limit

At epoch 60, the candidate's online training MAE is 1.363 meV worse than the
reference, while its selected EMA development MAE is 4.323 meV worse. The
candidate's early EMA development lead narrowed and reversed: candidate-minus-
reference was -21.674 meV at epoch 30, -1.210 meV at epoch 40, +3.205 meV
at epoch 50, and +4.323 meV at epoch 60. The last ten epoch deltas average
+3.913 meV. Both arms still improved in absolute EMA development MAE from
epoch 50 to 60 (reference 16.910 meV; candidate 15.792 meV). Thus the data
show relative erosion, not a demonstrated absolute plateau or a validated
early-stop rule. Online training and EMA development metrics use different
weights and roles; their difference is not a train–validation generalization
gap.

These observations are consistent with a weaker terminal fit or less useful
pair readback from `memory_message`; they do not isolate representation harm,
optimization difficulty, excess capacity, exposure insufficiency, or seed
noise. There is no fixed-cohort train evaluation, second seed, observed live
development series, or per-epoch checkpoint identity to distinguish them.
The per-epoch canonical traces also lack observed cumulative optimizer-step
and sample-presentation coordinates. RML therefore excludes **both** arms
from the replay pool despite terminal mechanical acceptance. Do not synthesize
these observations from the known terminal totals.

## Disposition

Close `memory_message` as `NEGATIVE_UNDER_CONTRACT` relative to the same-job
`memory_value` reference; keep the latter as a contextual closed arm. The
smallest future discriminator, if a new decision ever depends on this family,
would be a prospectively frozen fixed-cohort training/development comparison
with observed exposure and per-epoch checkpoint identities. The present
negative gate does not justify that additional GPU spend or a successor.
