# Phase 8 Model Comparison

Generated from accepted result artifacts. Blank cells indicate that a model was not evaluated on that domain.

| Model | Role | Common | OOD | P8-hard | PCQM Gap | PC100K avg | PC100K Gap | Encoder passes |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| routed_v4_500k | production | 0.103654 | 0.112783 | 0.094329 | 0.291691 |  |  | 3.00 |
| repaired_2m_d_gps7_seed42 | accepted general candidate | 0.100074 | 0.109555 | 0.090390 | 0.309216 |  |  | 1.00 |
| repaired_2m_d_gps9_seed42 | hard expert candidate | 0.099599 | 0.110666 | 0.088293 | 0.310251 |  |  | 1.00 |
| retention_d_three_seed_equal_ensemble | accuracy mode candidate | 0.099251 | 0.108520 | 0.089783 | 0.308859 |  |  | 3.00 |
| route_b_minimal | architecture control |  |  |  |  | 0.141998 | 0.171588 | 3.00 |
| route_b_cost | architecture control |  |  |  |  | 0.138825 | 0.167276 | 4.00 |
| route_b_precision | accepted scale-up candidate |  |  |  |  | 0.138046 | 0.165819 | 4.00 |
| route_b_precision_bounded_residual | accepted scale-up protocol |  |  |  |  | 0.134463 | 0.160809 | 4.00 |
