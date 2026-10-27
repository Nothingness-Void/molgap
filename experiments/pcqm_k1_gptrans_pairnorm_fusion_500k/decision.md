# Decision

NO_TRAIN: exploratory saved-prediction diagnostic completed on 50,000 previously
selection-used internal development rows. No model adopted or official role used.

| Fixed prediction | Gap MAE (eV) | Model passes |
| --- | ---: | ---: |
| K1 clean BN | 0.103659 | 1 |
| GPTrans PairNorm, LR 1e-3 | 0.099987 | 1 |
| K1 + GPTrans 1e-3, equal | 0.097580 | 2 |
| GPTrans 1e-3 + 5e-4, equal | 0.097525 | 2 |
| K1 + both GPTrans, equal | 0.096301 | 3 |
| K1 + GPTrans 1e-3, oracle | 0.078183 | not deployable |
| All three, oracle | 0.069418 | not deployable |

The three-way mean improves on the two-GPTrans mean by 1.224 meV on this
consumed cohort. A paired row-bootstrap 95% interval for the three-way minus
two-GPTrans MAE is [-1.418, -1.030] meV; this does not estimate seed variance.
The two-model K1 blend is 0.055 meV worse than the two-GPTrans mean, with a
row interval [-0.266, +0.387] meV. Both are selection-set descriptions.

Oracle chooses the lowest error using the actual target for each molecule.
It is an unattainable upper bound, not a validated or usable router. No claim
of achieving this bound or official leaderboard performance is justified.
