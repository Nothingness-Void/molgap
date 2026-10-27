# Decision

NO_TRAIN for any backbone or remote job. The prediction-only cross-fitted
gate does not meet its prospectively frozen 1 meV nomination rule.

| Method | K1 + GPTrans 1e-3 | K1 + mean(two GPTrans) |
| --- | ---: | ---: |
| Fixed equal-model mean | 0.097580 | 0.096301 |
| Cross-fitted global weight | 0.097351 | 0.096278 |
| Cross-fitted soft gate | 0.097097 | 0.096118 |
| Cross-fitted hard gate | 0.099987 | 0.097525 |
| Label-informed pair Oracle | 0.078183 | 0.078668 |

Relative to cross-fitted global weighting, the soft gate gains 0.254 and
0.160 meV respectively; five out of five folds improve in both cases.
The gains fall below 1 meV. The hard gate worsens both comparisons.
The second Oracle selects K1 or the *average* of both GPTrans, not one
of three separately. All models were already selected on this development
role; no independent score, training replay, full release or model adoption.
