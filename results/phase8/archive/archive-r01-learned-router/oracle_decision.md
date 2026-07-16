# archive-r01 Learned Router Oracle Decision

Frozen models: v3 base and dual-GPS expert. Objective weights: HOMO/LUMO/Gap = 0.25/0.25/0.50.
Fixed control: route when base Gap is `< 4 eV`.

| evaluation | n | fixed route | expert wins | fixed precision | fixed recall | fixed Gap MAE | budget Oracle Gap MAE | Oracle Gap MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| internal | 49758 | 25.5% | 60.0% | 60.6% | 25.8% | 0.097766 | 0.086980 | 0.083544 |
| common_all | 1977 | 27.3% | 53.9% | 60.9% | 30.9% | 0.121896 | 0.111528 | 0.108872 |
| common_ood1000 | 999 | 19.1% | 48.5% | 59.7% | 23.5% | 0.132178 | 0.124396 | 0.120520 |
| common_p8_targeted_hard | 978 | 35.7% | 59.3% | 61.6% | 37.1% | 0.111394 | 0.098608 | 0.096974 |
| pcqm_proxy | 2988 | 12.2% | 46.1% | 49.9% | 13.2% | 0.252838 | 0.241384 | 0.233834 |

**Decision: GO.** Oracle leaves more than 0.0015 eV additional internal Gap improvement.
Internal unrestricted Oracle adds `0.014222 eV` Gap improvement over v4; the same-budget Oracle adds `0.010786 eV`.

Build the leakage-controlled router development table and train feature ablations.
