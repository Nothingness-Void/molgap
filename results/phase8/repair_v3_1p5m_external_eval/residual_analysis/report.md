# Original 1M Residual Acquisition Analysis

This analysis uses the fixed common/OOD/P8-hard predictions to design
future acquisition. These 1,977 rows are therefore development evidence
from this point onward; promotion requires a new scaffold-disjoint sealed set.

## Highest-error structural strata (minimum 40 molecules)

| family | bucket | n | original 1M Gap MAE | 1.5M minus 1M | candidate win rate |
|---|---|---:|---:|---:|---:|
| aromatic_rings | (-inf, 0.0] | 169 | 0.18329 | +0.00127 | 0.485 |
| fraction_csp3 | (0.7, inf] | 186 | 0.15935 | +0.00874 | 0.468 |
| mw | (700.0, inf] | 216 | 0.15072 | +0.01046 | 0.440 |
| true_gap | (2.5, 3.2] | 124 | 0.14617 | +0.00750 | 0.476 |
| eval_set | ood1000 | 999 | 0.13491 | -0.00456 | 0.531 |
| true_gap | (5.5, inf] | 295 | 0.13337 | +0.00038 | 0.529 |
| n_N | no_n_N | 165 | 0.13081 | -0.00284 | 0.558 |
| rotatable_bonds | (7.0, inf] | 832 | 0.12906 | +0.00618 | 0.472 |
| amide_bonds | (0.0, 2.0] | 742 | 0.12744 | +0.00495 | 0.496 |
| aromatic_rings | (0.0, 2.0] | 791 | 0.12711 | +0.00051 | 0.520 |
| amide_bonds | (2.0, 5.0] | 216 | 0.12648 | +0.00300 | 0.468 |
| fraction_csp3 | (0.4, 0.7] | 529 | 0.12495 | -0.00090 | 0.537 |
| n_O | has_n_O | 1777 | 0.12346 | +0.00263 | 0.493 |
| n_F | no_n_F | 1535 | 0.12255 | +0.00159 | 0.498 |
| mw | (-inf, 300.0] | 465 | 0.12231 | +0.00010 | 0.514 |
| fraction_csp3 | (0.1, 0.4] | 963 | 0.12077 | +0.00269 | 0.481 |
| n_S | has_n_S | 684 | 0.12060 | +0.00519 | 0.481 |
| n_Cl | no_n_Cl | 1570 | 0.11980 | +0.00319 | 0.499 |
| true_gap | (3.2, 4.0] | 396 | 0.11919 | +0.00800 | 0.460 |
| mw | (300.0, 500.0] | 841 | 0.11858 | -0.00010 | 0.505 |

## Outputs

- `worst_200_molecules.csv`: molecule-level acquisition seeds.
- `worst_scaffolds.csv`: repeated high-error scaffolds.
- `residual_strata.csv`: all interpretable descriptor strata.
- `residual_descriptors.csv`: complete descriptor-enriched residual table.
