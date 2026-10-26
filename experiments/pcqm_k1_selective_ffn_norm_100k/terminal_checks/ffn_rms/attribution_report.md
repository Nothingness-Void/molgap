# Post-Training Attribution Report (`bn_reference` vs `ffn_rms`)

## 1. Executive Summary & Signals
- **Direction**: `NEGATIVE_DEGRADATION`
- **Selection Metric**: `live_dev_metric` (`50000` aligned development molecules)
- **Control MAE**: `0.139876 eV`
- **Candidate MAE**: `0.145607 eV`
- **Paired Delta (Candidate - Control)**: `+0.005731 eV` (95% CI `[+0.004737, +0.006805] eV`)
- **Material Gate (`<= -0.0030 eV`)**: `False` | **Noninferiority Gate (`<= +0.0005 eV`)**: `False`
- **Row Win / Loss / Tie**: `23860` / `26140` / `0` (Candidate win rate `47.72%`, median row gain `-0.003389 eV`)
- **Global Bias Shift Dominant**: `False` | **Late-Horizon Fade**: `True` | **Early-Lag Recovery**: `False`

## 2. Global Signed Bias & Post-Hoc Scalar Shift Diagnostic
| Arm | Raw MAE (eV) | Mean Signed Error (eV) | Post-Hoc Optimal Shift (eV) | Shift-Corrected MAE (eV) |
| --- | ---: | ---: | ---: | ---: |
| `bn_reference` | `0.139876` | `+0.009168` | `-0.002937` | `0.139846` |
| `ffn_rms` | `0.145607` | `+0.004028` | `+0.005049` | `0.145531` |
- **Shift-Corrected Delta (Candidate - Control)**: `+0.005685 eV` (Raw Delta `+0.005731 eV`)

## 3. Tail Absolute Error Quantiles (eV)
| Quantile | Control (eV) | Candidate (eV) | Delta (Candidate - Control) |
| --- | ---: | ---: | ---: |
| `p50_eV` | `0.092821` | `0.096426` | `+0.003605` |
| `p75_eV` | `0.181642` | `0.188403` | `+0.006760` |
| `p90_eV` | `0.314342` | `0.325497` | `+0.011155` |
| `p95_eV` | `0.425206` | `0.436873` | `+0.011667` |
| `p99_eV` | `0.746821` | `0.772290` | `+0.025468` |
| `max_eV` | `5.699914` | `5.433044` | `-0.266870` |

## 4. Target HOMO-LUMO Gap Regime Stratification
| Stratum | Rows | Target Mean (eV) | Control MAE (eV) | Candidate MAE (eV) | Delta (eV) | Candidate Win % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `<4.0_eV` | `3725` | `3.5308` | `0.201171` | `0.224001` | `+0.022830` | `44.43%` |
| `4.0_to_6.0_eV` | `34837` | `5.0830` | `0.125550` | `0.130931` | `+0.005381` | `47.70%` |
| `6.0_to_8.0_eV` | `10451` | `6.6951` | `0.157999` | `0.159445` | `+0.001446` | `48.79%` |
| `>=8.0_eV` | `987` | `8.5116` | `0.222331` | `0.221243` | `-0.001088` | `49.54%` |

## 5. Complementarity & Ensemble Check
- **Mean |Pred_Cand - Pred_Ctrl|**: `0.090164 eV`
- **Signed Residual Pearson r**: `0.780173` | **Absolute Error Pearson r**: `0.741901`
- **50/50 Equal Blend MAE**: `0.134586 eV`
- **Grid-Oracle Blend MAE**: `0.134265 eV` (optimal candidate weight `w = 0.41`, gain over best single `+0.005611 eV`)

## 6. Optimization & Trace Dynamics
- **Control Selected Pass / Final Pass**: `40.0` (`0.139876 eV`) vs `40.0` (`0.139876 eV`)
- **Candidate Selected Pass / Final Pass**: `39.0` (`0.145607 eV`) vs `40.0` (`0.145637 eV`)
- **Early Window (Passes 1..10) Mean Delta**: `-0.006120 eV`
- **Late Window (Last 10 Passes) Mean Delta**: `+0.005926 eV` | **Last 5 Passes**: `+0.005660 eV`
- **Candidate Ahead in Evaluations**: `11 / 40`

## 7. Runtime Cost
- **Process Wall**: control `2041.59s`, candidate `2055.92s`
- **Wall Ratio**: `1.0070x`
- Post-hoc descriptive analysis only; row bootstrap does not measure training stochasticity, qualify a strict reference or authorize promotion.
