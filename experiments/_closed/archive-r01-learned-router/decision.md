# archive-r01 Learned Router External Decision

Selected features: `R3` (38 features).
Candidate below is budget-matched to the fixed Gap<4 route on each full evaluation set.

| evaluation | fixed route | learned route | fixed precision | learned precision | weighted delta | Gap delta | Gap 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|
| common_all | 27.3% | 27.3% | 58.0% | 58.9% | +0.000048 | -0.000207 | [-0.001781, +0.001143] |
| common_ood1000 | 19.1% | 29.1% | 59.2% | 52.8% | +0.000681 | +0.000527 | [-0.000937, +0.002045] |
| common_p8_targeted_hard | 35.7% | 25.6% | 57.3% | 66.0% | -0.000598 | -0.000957 | [-0.003732, +0.001278] |
| pcqm_proxy | 12.1% | 12.1% | 49.0% | 44.1% | +0.001898 | +0.001898 | [+0.000735, +0.003093] |

**Decision: do not promote. Keep fixed routed-v4 as the default.**
At least one external promotion gate failed; keep fixed v4.
