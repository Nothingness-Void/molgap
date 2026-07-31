# Phase 8 Model Comparison

Generated from accepted result artifacts. Blank cells indicate that a model was not evaluated on that domain.

Common/OOD/P8-hard columns are only comparable within the same `Rows` value: the paired repaired-2M comparison dropped the 4 rows where ETKDG failed so every method shared identical molecules, while the single-model runs kept all 1,977.

| Model | Role | Rows | Common | OOD | P8-hard | PCQM Gap | PC100K avg | PC100K Gap | Encoder passes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| routed_v4_500k | previous production baseline | 1,977 | 0.103654 | 0.112783 | 0.094329 | 0.291691 |  |  | 3.00 |
| repaired_2m_d_gps7_seed42 | accepted general candidate | 1,977 | 0.100074 | 0.109555 | 0.090390 | 0.309216 |  |  | 1.00 |
| repaired_2m_d_gps9_seed42 | hard expert candidate | 1,977 | 0.099599 | 0.110666 | 0.088293 | 0.310251 |  |  | 1.00 |
| retention_d_three_seed_equal_ensemble | accuracy mode candidate | 1,977 | 0.099251 | 0.108520 | 0.089783 | 0.308859 |  |  | 3.00 |
| route_b_minimal | architecture control |  |  |  |  |  | 0.141998 | 0.171588 | 3.00 |
| route_b_cost | architecture control |  |  |  |  |  | 0.138825 | 0.167276 | 4.00 |
| route_b_precision | accepted scale-up candidate |  |  |  |  |  | 0.138046 | 0.165819 | 4.00 |
| route_b_precision_bounded_residual | accepted scale-up protocol |  |  |  |  |  | 0.134463 | 0.160809 | 4.00 |
| repaired_2m_three_gps_dense_2d | production | 1,973 | 0.097638 | 0.106655 | 0.088407 | 0.302120 |  |  | 3.00 |
| repaired_2m_gps7_gps9_equal_2d | production lower-cost preset | 1,973 | 0.098467 | 0.108798 | 0.087892 | 0.307979 |  |  | 2.00 |
