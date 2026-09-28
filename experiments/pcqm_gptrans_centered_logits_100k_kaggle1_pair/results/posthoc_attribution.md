# Centered-logits saved-artifact attribution

This post-hoc analysis reads only the two accepted development-prediction
payloads and canonical traces. The prediction SHA256 values match each arm's
mechanical acceptance; ordered source indices are exactly 100000–149999,
targets are identical, predictions are finite, and protected-role flags are
false. Exact input hashes and calculations are in
`posthoc_attribution.json`. The [terminal decision](../decision.md) remains
the sole scientific disposition.

## Observed pattern

The frozen endpoint gain is 0.598 meV, below the 3 meV gate, and the accepted
paired interval crosses zero. Centering improves 49.702% of rows; the median
per-row absolute-error gain is **−0.479 meV**. Its mean prediction is
4.240 meV higher than the same-job reference. Both models underpredict on
average (reference −21.434 meV; centered −17.194 meV). If each prediction
vector is given its own MAE-optimal constant offset *fitted on these same
development labels*, the descriptive gain falls from 0.598 to 0.186 meV.
That offset exercise is not an evaluation score or an authorized calibration
step; it suggests that a global prediction shift accounts for much of the
small observed advantage.

The EMA-development lead averaged 193.362 meV over epochs 0–9 and only
0.236 meV over epochs 50–59. It was −0.011 meV at epoch 49 and +0.598 meV
at epoch 59. Both arms improved by roughly 14 meV from epoch 49 to 59 and
selected epoch 59. The candidate's final online training MAE was 3.896 meV
lower than the reference's, while its EMA-development gain was only
0.598 meV. Online training and EMA development are different measurements;
this is compatible with a training-fit advantage that transfers weakly, not
proof of overfitting.

Post-hoc target strata show −2.378 meV gain in the common 4–6 eV region
(34,837 rows), versus +7.415 meV at 6–8 eV (10,451 rows). Target Gap is not
available at inference, and these selected-role strata cannot define a router
or new promotion gate.

## Failure-mode disposition

- **Underfitting:** the candidate's online training metric is better than the
  reference's, so the module did not worsen that relative measurement. This
  does not establish whether either arm underfit absolutely, because there
  is no matched fixed-cohort training evaluation.
- **Insufficient exposure:** both curves were still improving at epoch 59,
  so a late reversal cannot be logically excluded. The relative advantage
  had contracted to near zero by the frozen endpoint; existing evidence does
  not justify more epochs or a new seed.
- **Module harm:** no clear aggregate harm was observed, but the gain is
  weak, row-sign evidence is mixed, and the common 4–6 eV stratum worsens.
- **Next cheapest falsifier if this question is ever reopened:** predeclare a
  train-only calibration/fixed-cohort diagnostic that separates output-offset
  changes from per-row ranking changes. Reusing these development labels to
  fit an offset cannot establish generalization.

No new training, checkpoint inference, protected-role access, or scientific
promotion occurred in this analysis.
