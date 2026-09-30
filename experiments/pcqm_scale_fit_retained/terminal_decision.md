# Retained scale-fit diagnostic - 2026-10-01

## Disposition

NO_TRAIN after a completed frozen-checkpoint diagnostic. Production and the
accepted full EdgeState reference are unchanged. The broad causal explanation
of 100K-to-500K contraction remains insufficient_evidence.
Machine results and acceptance are in `results/analysis.json` and `acceptance.json`.

## Matched subset observations

All states used identical ascending 5K training rows [0,5000) and 5K consumed
internal-development rows [500000,505000), at terminal epoch 59, without updates.
Values below are reference-minus-joint MAE gains, not cross-scale MAE subtractions.

| Retained states | Training gain | Development gain |
|---|---:|---:|
| 100K LIVE | 0.006489 eV | 0.004203 eV |
| 100K EMA | 0.005446 eV | 0.004685 eV |
| 500K LIVE | 0.008568 eV | 0.005251 eV |

The joint mechanism improves fixed-cohort training fit at both scales; its
development advantage does not vanish on this prefix. The 500K training gain
exceeds its development gain by 0.003317 eV, versus 0.002286 eV at 100K LIVE.
This measures unequal transfer of fit benefit, not proof of overfitting.
Both 500K models also have much lower prefix-development MAE than their 100K
counterparts; no uniform representation collapse is demonstrated.

The accepted whole-50K endpoint and earlier post-hoc Gap slices remain with
`../pcqm_gptrans_500k_frozen_readout/decision.md`. This bounded prefix is not a
representative random sample and cannot replace that whole-role decision or
show a calibrated subgroup effect. Terminal versus best selection also differs
in those historical summaries. Development heterogeneity remains relevant.

## Missing discriminators

500K EMA was not retained. The joint 100K transform uses the authenticated same
producer/cache/statistics derivation, not independently serialized joint values.
Cross-scale schedules, exposure, source provenance and hardware are not a
matched factorial control; one training seed does not identify stochasticity.
The existing raw/EMA probe already excluded EMA alone as a material explanation;
this diagnostic adds fixed-cohort training fit, not a repeated selection claim.

Only a separately justified matched-step, matched-LR, same-role data-size/exposure
control could isolate those causal alternatives. No such training or automatic
early-stop policy is released by this diagnostic. Its row-bootstrap intervals
do not measure training-seed variance.

Observed execution: RTX 5060, 60,000 forwards, 22.971 script wall seconds,
15.062 synchronized GPU-resident seconds and 11.524 CUDA-event forward seconds.
These scopes are not interchangeable; CPU hours and full allocation time are unknown.
