# V1 / V2 / V3 PPT Metrics

Date: 2026-06-30

Inputs:

- common eval predictions: `results\phase8\full_expansion500k_common_eval_predictions.csv`
- PCQM4Mv2 proxy predictions: `results\phase8\pcqm4mv2_proxy_p7_v2_v3_predictions.csv`

## Recommended PPT Table

Same 1,977 common-eval molecules for all three models. This is the fairest
cross-version comparison.

| model | n | HOMO_mae | HOMO_r2 | LUMO_mae | LUMO_r2 | Gap_mae | Gap_r2 | average_mae | average_r2 |
|---|---|---|---|---|---|---|---|---|---|
| v1_phase7 | 1977 | 0.1274 | 0.8626 | 0.1292 | 0.9469 | 0.1793 | 0.9187 | 0.1453 | 0.9094 |
| v2_replacement300k | 1977 | 0.1150 | 0.8860 | 0.1141 | 0.9581 | 0.1561 | 0.9389 | 0.1284 | 0.9277 |
| v3_expansion500k | 1977 | 0.0943 | 0.9200 | 0.0972 | 0.9679 | 0.1253 | 0.9569 | 0.1056 | 0.9483 |

## Common Eval By Slice

### OOD-1000

| model | n | Gap_mae | Gap_r2 | average_mae | average_r2 |
|---|---|---|---|---|---|
| v1_phase7 | 999 | 0.1488 | 0.9566 | 0.1243 | 0.9413 |
| v2_replacement300k | 999 | 0.1448 | 0.9609 | 0.1214 | 0.9443 |
| v3_expansion500k | 999 | 0.1340 | 0.9647 | 0.1137 | 0.9500 |

### P8 Targeted Hard

| model | n | Gap_mae | Gap_r2 | average_mae | average_r2 |
|---|---|---|---|---|---|
| v1_phase7 | 978 | 0.2104 | 0.7934 | 0.1667 | 0.8291 |
| v2_replacement300k | 978 | 0.1676 | 0.8583 | 0.1355 | 0.8775 |
| v3_expansion500k | 978 | 0.1164 | 0.9163 | 0.0973 | 0.9274 |

## Component-Level Common Eval

All common-eval molecules. Useful if a slide needs to show 2D, 3D, and fusion.

| model | component | Gap_mae | Gap_r2 | average_mae | average_r2 |
|---|---|---|---|---|---|
| v1_phase7 | gps_2d | 0.1890 | 0.9107 | 0.1525 | 0.9005 |
| v1_phase7 | schnet_3d | 0.2028 | 0.9025 | 0.1650 | 0.8890 |
| v1_phase7 | hybrid | 0.1793 | 0.9187 | 0.1453 | 0.9094 |
| v2_replacement300k | gps_2d | 0.1811 | 0.9230 | 0.1467 | 0.9110 |
| v2_replacement300k | schnet_3d | 0.1784 | 0.9217 | 0.1466 | 0.9094 |
| v2_replacement300k | hybrid | 0.1561 | 0.9389 | 0.1284 | 0.9277 |
| v3_expansion500k | gps_2d | 0.1263 | 0.9560 | 0.1076 | 0.9461 |
| v3_expansion500k | schnet_3d | 0.1616 | 0.9331 | 0.1323 | 0.9236 |
| v3_expansion500k | hybrid | 0.1253 | 0.9569 | 0.1056 | 0.9483 |

## PCQM4Mv2 Valid Proxy

This is a leakage-filtered PCQM4Mv2 valid proxy, **not** an OGB submission.

| model | n | Gap_mae | Gap_rmse | Gap_r2 | Gap_bias | Gap_median_abs_err |
|---|---|---|---|---|---|---|
| v1_phase7 | 2988 | 0.2588 | 0.4298 | 0.8422 | 0.0205 | 0.1696 |
| v2_replacement300k | 2988 | 0.2519 | 0.4090 | 0.8570 | 0.0180 | 0.1658 |
| v3_expansion500k | 2988 | 0.2531 | 0.4198 | 0.8494 | 0.0042 | 0.1620 |

## Internal Test Metrics

These are training-run records and are **not** a fair cross-phase comparison,
because the datasets/splits differ. Use them only as provenance.

| model | HOMO_mae | HOMO_r2 | LUMO_mae | LUMO_r2 | Gap_mae | Gap_r2 | average_mae | average_r2 | best_val_mae |
|---|---|---|---|---|---|---|---|---|---|
| v1_phase7 | 0.0640 | 0.9578 | 0.0617 | 0.9853 | 0.0756 | 0.9764 | 0.0671 | 0.9732 | 0.0671 |
| v2_replacement300k | 0.0892 | 0.9256 | 0.0881 | 0.9742 | 0.1150 | 0.9597 | 0.0974 | 0.9532 | 0.0966 |
| v3_expansion500k | 0.0788 | 0.9377 | 0.0798 | 0.9765 | 0.1004 | 0.9672 | 0.0863 | 0.9605 | 0.0853 |

## Files

- JSON: `results\phase8\v1_v2_v3_ppt_metrics.json`
- common eval CSV: `results\phase8\v1_v2_v3_common_eval_metrics.csv`
- PCQM proxy CSV: `results\phase8\v1_v2_v3_pcqm_proxy_metrics.csv`
- internal CSV: `results\phase8\v1_v2_v3_internal_test_metrics.csv`
