# Learning-rate pair: terminal attribution

This analysis uses the retained aligned predictions and complete traces for
the two 500K arms. The controlled change is the peak learning rate. The
candidate `5e-4` selected at pass 44 reaches 0.100502567 eV; the reference
`1e-3` selected at pass 52 reaches 0.099987378 eV. Candidate minus reference
is +0.515189 meV (paired-row 95% interval [+0.056049, +1.034131] meV).
The frozen material improvement criterion is not met.

| Pass | Optimizer steps per arm | `1e-3` EMA MAE/eV | `5e-4` EMA MAE/eV |
| ---: | ---: | ---: | ---: |
| 10 | 39060 | 0.113161 | 0.110034 |
| 20 | 78120 | 0.107775 | 0.104990 |
| 30 | 117180 | 0.104339 | 0.102762 |
| 40 | 156240 | 0.101404 | 0.100736 |
| 50 | 195300 | 0.100308 | 0.100939 |
| 60 | 234360 | 0.100148 | 0.101135 |

The lower-rate arm is ahead at 40 of the 60 paired evaluations but fades
late: its selected pass 44 to pass 60 worsens by 0.632 meV. The reference
worsens by 0.160 meV from its selected pass 52 to pass 60. The lower-rate
arm's online training metric falls from 0.081765 to 0.068641 across its
selected-to-final interval, while its development EMA worsens. That training
metric is an in-pass average under changing live weights, not a fixed-weight
training evaluation of the EMA model. It does not establish overfitting,
underfitting, a capacity ceiling, or the reason for late degradation.

The experiment supports a rate-specific conclusion for this one seed and
recipe. It cannot establish the incremental effect of PairNorm on BondLocal:
both arms include both mechanisms and the older 500K run lacks a qualified
strict reference. Row bootstrap does not measure training-seed variability.
An exploratory equal 50/50 prediction blend gives 0.097525 eV on the same
selection-used development rows; it requires two trained models, is post-hoc,
and does not establish held-out or full-scale performance.

The machine-readable diagnosis is retained at
`D:/molgap-artifacts/reconciliation/gptrans-bondlocal-pairnorm-lr-500k/v1/diagnosis.json`.
