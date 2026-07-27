# Exact-2M pure-2D coverage expert decision

## Decision

**Do not replace the current `anchor + repair` incumbent and do not open the
future sealed 20K.** The exact-2M coverage expert is strongly positive on OOD,
the prior scaffold-novel 10K, and PCQM, but significantly regresses on the
P8-targeted-hard scope. It is a specialist-positive result, not a global model
upgrade.

All deltas below are candidate MAE minus incumbent MAE; negative is better.

| Scope | Candidate | Average delta (eV) | Gap delta (eV) |
|---|---|---:|---:|
| Common all, n=1,980 | coverage2m | +0.003362 | +0.003901 |
| Common all | anchor + coverage2m | +0.000125 | -0.000026 |
| Common all | anchor + repair + coverage2m | -0.000497 | -0.000468 |
| OOD1000, n=999 | coverage2m | -0.002284 | -0.003449 |
| OOD1000 | anchor + coverage2m | -0.001775 | -0.002349 |
| OOD1000 | anchor + repair + coverage2m | -0.002396 | -0.002916 |
| P8-targeted-hard, n=981 | coverage2m | +0.009112 | +0.011385 |
| P8-targeted-hard | anchor + coverage2m | +0.002061 | +0.002341 |
| P8-targeted-hard | anchor + repair + coverage2m | +0.001436 | +0.002025 |
| Prior scaffold-novel 10K | coverage2m | -0.011068 | -0.015533 |
| Prior scaffold-novel 10K | anchor + coverage2m | -0.006981 | -0.010042 |
| Prior scaffold-novel 10K | anchor + repair + coverage2m | -0.006423 | -0.008865 |
| PCQM valid 5K | coverage2m | n/a | -0.003560 |
| PCQM valid 5K | anchor + coverage2m | n/a | -0.004984 |
| PCQM valid 5K | anchor + repair + coverage2m | n/a | -0.003430 |

The OOD improvements for the two fixed ensembles are statistically positive.
The P8-targeted-hard regressions are also statistically significant. On common
all, the tri-expert average is only a statistical tie. The prior 10K and PCQM
improvements are significant, but they do not override the hard-scope failure.

## Training outcome

| Model | Test average MAE (eV) | Test Gap MAE (eV) |
|---|---:|---:|
| GPS7 | 0.102431 | 0.121676 |
| GPS9 | 0.101905 | 0.121112 |
| dual-GPS head | 0.100467 | 0.119418 |

GPS7 and GPS9 both strictly warm-started from their accepted 1.5M checkpoints.
The final evaluation retry was required because the evaluator initially
rejected an ensemble baseline before inference. The validation was corrected
to accept any generated prediction name; retry job `702259` completed without
retraining and evaluated common, OOD, P8-hard, prior 10K, and PCQM.

Machine-readable metrics are in `../multi2d_2m_dev_eval/`.
