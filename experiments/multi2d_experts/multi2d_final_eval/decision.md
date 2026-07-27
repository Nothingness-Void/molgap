# Pure-2D Multi-Expert Decision

## Decision

Retain `mean_control_repair` as a **positive pure-2D candidate**. It is the
fixed equal average of two independently trained dual-GPS predictors:

- `control_a`: the equal-cost original-1M control from the residual40k run;
- `repair_v2`: the controlled repair-v2 1M model.

This candidate is not a new production version and does not replace routed-v4.
It requires both dual-GPS experts at inference (four GPS encoder passes total),
so the next gate is compression/distillation, not another Router search.

## Independent gate

The four ensemble formulas were fixed before the final sealed set was opened.
The final set contains 10,000 rows from hard-acquisition rounds 04-05, one row
per Bemis-Murcko scaffold. CID, canonical-SMILES, and scaffold overlaps were
removed against the original 1M training data, the repair candidate union, and
the broad/residual development pool. It is permanently forbidden for training
or weight selection. Construction details and SHA256 are in
`../multi2d_final_sealed/selection_report.json`.

## Result

All deltas are paired MAE differences versus `control_a`; negative is better.

| Evaluation | N | Average MAE control -> ensemble | Delta average (eV) | Gap MAE control -> ensemble | Delta Gap (eV) | Gap 95% CI |
|---|---:|---:|---:|---:|---:|---:|
| Common all | 1,980 | 0.10237 -> 0.10029 | -0.00208 | 0.11907 -> 0.11691 | -0.00216 | [-0.00357, -0.00076] |
| OOD-1000 | 999 | 0.11586 -> 0.11236 | -0.00350 | 0.13536 -> 0.13116 | -0.00420 | [-0.00627, -0.00211] |
| P8 targeted hard | 981 | 0.08863 -> 0.08800 | -0.00063 | 0.10248 -> 0.10240 | -0.00008 | [-0.00198, 0.00180] |
| New scaffold-sealed hard | 10,000 | 0.20046 -> 0.19486 | -0.00560 | 0.25414 -> 0.24734 | -0.00681 | [-0.00788, -0.00571] |
| PCQM4Mv2 valid proxy | 5,000 | n/a | n/a | 0.31301 -> 0.31246 | -0.00054 | [-0.00234, 0.00122] |

The sealed average-MAE delta 95% CI is `[-0.00627, -0.00495]` eV. All seven
sealed difficulty buckets improve both average and Gap MAE. The P8-hard and
PCQM changes are favorable but statistically tied, so they are not claimed as
confirmed gains.

## Interpretation

The prior negative conclusion applied to replacing the base with one repaired
dataset model. The new result shows a different fact: the original-data and
repair-data models retain complementary residuals, and prediction averaging
reduces them without a learned Router. This is genuine multi-expert benefit,
but its current compute cost is too high for default deployment.

Raw paired predictions and 10,000-draw bootstrap metrics are stored beside this
file. SCNet job `699912` completed successfully in 2m16s.
