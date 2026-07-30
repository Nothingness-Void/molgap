# Phase 8 Routed Dual-GPS Architecture Decision

Rule: use the dual-GPS hybrid only when the base v3 predicted Gap is `< 4 eV`.
Training data and SchNet are unchanged.

| evaluation | routed n / n | avg MAE delta | Gap MAE delta | Gap 95% CI |
|---|---:|---:|---:|---:|
| internal held-out test | 12693 / 49758 | -0.002042 | -0.002665 | [-0.002904, -0.002434] |
| common all | 540 / 1977 | -0.002212 | -0.003388 | [-0.004616, -0.002187] |
| common ood1000 | 191 / 999 | -0.001143 | -0.001811 | [-0.003116, -0.000536] |
| common p8_targeted_hard | 349 / 978 | -0.003305 | -0.004998 | [-0.007073, -0.002994] |
| PCQM proxy | 365 / 2988 | n/a | -0.000223 | [-0.001358, +0.000851] |


**Decision: positive. Promote as the v4 B3LYP accuracy predictor; keep the v3 single hybrid as the component/compatibility loader.**
The next gate is re-running Phase 9/10 Delta and UQ against routed v4 outputs.
