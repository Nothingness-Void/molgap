# Retention-D Three-Seed Equal Ensemble

## Decision

Retain the equal seed42/43/44 GPS7 ensemble as an **accuracy-mode candidate**.
It requires three GPS7 encoder passes per molecule and does not replace the
production model or the one-pass Retention-D seed42 candidate.

## Fixed External Metrics

| Scope | Average MAE (eV) | Gap MAE (eV) |
|---|---:|---:|
| Common | 0.099251 | 0.115401 |
| OOD-1000 | 0.108520 | 0.126700 |
| P8-hard | 0.089783 | 0.103858 |
| PCQM valid 4,981 | n/a | 0.308859 |

The machine-readable record contains per-target metrics, aligned input hashes,
and the explicit `3.0` GPS7-pass cost:
`retention_d_three_seed_equal_ensemble.json`.

No sealed-20K rows were used and the production registry was not changed.
