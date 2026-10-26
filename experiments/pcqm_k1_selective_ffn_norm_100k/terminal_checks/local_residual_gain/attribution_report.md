# Post-Training Attribution Report (`bn_reference` vs `local_residual_gain`)

## 1. Executive Summary & Signals
- **Direction**: `NEUTRAL_EQUIVALENT`
- **Selection Metric**: `live_dev_metric` (`50000` aligned development molecules)
- **Control MAE**: `0.139876 eV`
- **Candidate MAE**: `0.139387 eV`
- **Paired Delta (Candidate - Control)**: `-0.000490 eV` (95% CI `[-0.001328, +0.000313] eV`)
- **Material Gate (`<= -0.0030 eV`)**: `False` | **Noninferiority Gate (`<= +0.0005 eV`)**: `True`
- **Row Win / Loss / Tie**: `25269` / `24731` / `0` (Candidate win rate `50.54%`, median row gain `+0.000721 eV`)
- **Global Bias Shift Dominant**: `False` | **Late-Horizon Fade**: `False` | **Early-Lag Recovery**: `True`

## 2. Global Signed Bias & Post-Hoc Scalar Shift Diagnostic
| Arm | Raw MAE (eV) | Mean Signed Error (eV) | Post-Hoc Optimal Shift (eV) | Shift-Corrected MAE (eV) |
| --- | ---: | ---: | ---: | ---: |
| `bn_reference` | `0.139876` | `+0.009168` | `-0.002937` | `0.139846` |
| `local_residual_gain` | `0.139387` | `+0.008086` | `-0.001725` | `0.139377` |
- **Shift-Corrected Delta (Candidate - Control)**: `-0.000469 eV` (Raw Delta `-0.000490 eV`)

## 3. Tail Absolute Error Quantiles (eV)
| Quantile | Control (eV) | Candidate (eV) | Delta (Candidate - Control) |
| --- | ---: | ---: | ---: |
| `p50_eV` | `0.092821` | `0.091884` | `-0.000937` |
| `p75_eV` | `0.181642` | `0.181879` | `+0.000237` |
| `p90_eV` | `0.314342` | `0.312805` | `-0.001537` |
| `p95_eV` | `0.425206` | `0.420778` | `-0.004429` |
| `p99_eV` | `0.746821` | `0.736229` | `-0.010593` |
| `max_eV` | `5.699914` | `5.335600` | `-0.364314` |

## 4. Target HOMO-LUMO Gap Regime Stratification
| Stratum | Rows | Target Mean (eV) | Control MAE (eV) | Candidate MAE (eV) | Delta (eV) | Candidate Win % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `<4.0_eV` | `3725` | `3.5308` | `0.201171` | `0.202473` | `+0.001302` | `48.24%` |
| `4.0_to_6.0_eV` | `34837` | `5.0830` | `0.125550` | `0.124371` | `-0.001178` | `51.01%` |
| `6.0_to_8.0_eV` | `10451` | `6.6951` | `0.157999` | `0.157803` | `-0.000195` | `50.16%` |
| `>=8.0_eV` | `987` | `8.5116` | `0.222331` | `0.236263` | `+0.013933` | `46.40%` |

## 5. Complementarity & Ensemble Check
- **Mean |Pred_Cand - Pred_Ctrl|**: `0.069924 eV`
- **Signed Residual Pearson r**: `0.867630` | **Absolute Error Pearson r**: `0.832981`
- **50/50 Equal Blend MAE**: `0.134647 eV`
- **Grid-Oracle Blend MAE**: `0.134644 eV` (optimal candidate weight `w = 0.51`, gain over best single `+0.004743 eV`)

## 6. Optimization & Trace Dynamics
- **Control Selected Pass / Final Pass**: `40.0` (`0.139876 eV`) vs `40.0` (`0.139876 eV`)
- **Candidate Selected Pass / Final Pass**: `40.0` (`0.139387 eV`) vs `40.0` (`0.139387 eV`)
- **Early Window (Passes 1..10) Mean Delta**: `+0.003531 eV`
- **Late Window (Last 10 Passes) Mean Delta**: `-0.000190 eV` | **Last 5 Passes**: `-0.000478 eV`
- **Candidate Ahead in Evaluations**: `24 / 40`

## 7. Runtime Cost
- **Process Wall**: control `2041.59s`, candidate `1994.42s`
- **Wall Ratio**: `0.9769x`
- Post-hoc descriptive analysis only; row bootstrap does not measure training stochasticity, qualify a strict reference or authorize promotion.
