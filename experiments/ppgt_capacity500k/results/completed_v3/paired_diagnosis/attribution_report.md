# Post-Training Attribution Report (`ppgt_w192` vs `ppgt_w384`)

## 1. Executive Summary & Signals
- **Direction**: `NEGATIVE_DEGRADATION`
- **Selection Metric**: `live_dev_metric` (`50000` aligned development molecules)
- **Control MAE**: `0.171471 eV`
- **Candidate MAE**: `0.368031 eV`
- **Paired Delta (Candidate - Control)**: `+0.196560 eV` (95% CI `[+0.193905, +0.199364] eV`)
- **Material Gate (`<= -0.0050 eV`)**: `False` | **Noninferiority Gate (`<= +0.0005 eV`)**: `False`
- **Row Win / Loss / Tie**: `12662` / `37338` / `0` (Candidate win rate `25.32%`, median row gain `-0.134596 eV`)
- **Global Bias Shift Dominant**: `False` | **Late-Horizon Fade**: `False` | **Early-Lag Recovery**: `True`

## 2. Global Signed Bias & Post-Hoc Scalar Shift Diagnostic
| Arm | Raw MAE (eV) | Mean Signed Error (eV) | Post-Hoc Optimal Shift (eV) | Shift-Corrected MAE (eV) |
| --- | ---: | ---: | ---: | ---: |
| `ppgt_w192` | `0.171471` | `+0.007519` | `+0.000643` | `0.171470` |
| `ppgt_w384` | `0.368031` | `+0.017624` | `+0.006695` | `0.367981` |
- **Shift-Corrected Delta (Candidate - Control)**: `+0.196512 eV` (Raw Delta `+0.196560 eV`)

## 3. Tail Absolute Error Quantiles (eV)
| Quantile | Control (eV) | Candidate (eV) | Delta (Candidate - Control) |
| --- | ---: | ---: | ---: |
| `p50_eV` | `0.113577` | `0.271982` | `+0.158406` |
| `p75_eV` | `0.224988` | `0.501427` | `+0.276439` |
| `p90_eV` | `0.381264` | `0.796035` | `+0.414770` |
| `p95_eV` | `0.505931` | `1.028317` | `+0.522386` |
| `p99_eV` | `0.897336` | `1.622739` | `+0.725403` |
| `max_eV` | `4.628912` | `5.295690` | `+0.666778` |

## 4. Target HOMO-LUMO Gap Regime Stratification
| Stratum | Rows | Target Mean (eV) | Control MAE (eV) | Candidate MAE (eV) | Delta (eV) | Candidate Win % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `<4.0_eV` | `2708` | `3.5374` | `0.309895` | `0.759550` | `+0.449655` | `19.09%` |
| `4.0_to_6.0_eV` | `30313` | `5.1515` | `0.151477` | `0.331644` | `+0.180167` | `24.56%` |
| `6.0_to_8.0_eV` | `15168` | `6.7560` | `0.179659` | `0.358811` | `+0.179153` | `27.74%` |
| `>=8.0_eV` | `1811` | `8.6637` | `0.230566` | `0.468856` | `+0.238289` | `27.17%` |

## 5. Complementarity & Ensemble Check
- **Mean |Pred_Cand - Pred_Ctrl|**: `0.314503 eV`
- **Signed Residual Pearson r**: `0.528214` | **Absolute Error Pearson r**: `0.446288`
- **50/50 Equal Blend MAE**: `0.241388 eV`
- **Grid-Oracle Blend MAE**: `0.171471 eV` (optimal candidate weight `w = 0.00`, gain over best single `+0.000000 eV`)

## 6. Optimization & Trace Dynamics
- **Control Selected Pass / Final Pass**: `3.9989759344598053` (`0.171471 eV`) vs `3.9989759344598053` (`0.171471 eV`)
- **Candidate Selected Pass / Final Pass**: `3.9989759344598053` (`0.368031 eV`) vs `3.9989759344598053` (`0.368031 eV`)
- **Early Window (Passes 1..10) Mean Delta**: `+0.219072 eV`
- **Late Window (Last 10 Passes) Mean Delta**: `+0.202860 eV` | **Last 5 Passes**: `+0.197854 eV`
- **Candidate Ahead in Evaluations**: `0 / 20`

## 7. Runtime Cost
- **Process Wall**: control `18781.76s`, candidate `19089.69s`
- **Wall Ratio**: `1.0164x`
- Post-hoc descriptive analysis only; row bootstrap does not measure training stochasticity, qualify a strict reference or authorize promotion.
