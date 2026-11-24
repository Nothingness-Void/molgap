# TokenGT-GPTrans 100K Terminal Attribution

`tokengt` (`TokenGTGPTrans`, `5,928,961` parameters) fails with **`endpoint_regression`** against retained baseline `uniform_wd_multi_ema` (`OGBGPTransTiny`, `5,871,201` parameters) under the frozen 60-pass (`46,860` optimizer steps) `pcqm_fixed_100k_v1` contract.

## Trajectory and Endpoint Comparison

| Pass | `uniform_wd_multi_ema` EMA Dev (eV) | `tokengt` EMA Dev (eV) | `tokengt - ctrl` (meV) |
| ---: | ---: | ---: | ---: |
| 1 | 0.451195 | 0.630491 | +179.296 |
| 5 | 0.220146 | 0.361956 | +141.810 |
| 10 | 0.173343 | 0.273722 | +100.379 |
| 20 | 0.150373 | 0.235666 | +85.293 |
| 30 | 0.142075 | 0.219173 | +77.098 |
| 42 | **0.139440** | 0.210134 | +70.694 |
| 50 | 0.139727 | 0.207731 | +68.004 |
| 58 | 0.139810 | **0.207216** | +67.406 |
| 60 | 0.139867 | 0.207272 | +67.405 |

- **Selected prediction MAE (50,000 aligned rows)**: `tokengt = 0.207216078 eV` (pass `58`), `uniform_wd_multi_ema = 0.139440192 eV` (pass `42`), paired delta **`+67.775886 meV`** (95% CI `[+66.040341, +69.435555] meV`, `n_bootstrap = 10000`, `seed = 42`).
- **Trajectory dominance**: `tokengt` trails at all `60 / 60` passes (`early window p1..10 mean delta = +129.036 meV`, `last 10 passes = +67.566 meV`, `last 5 passes = +67.481 meV`). Although relative gap narrows slightly (`early_lag_late_recovery = true`), `tokengt` converges to an elevated `~207 meV` plateau.
- **Stochastic train signal (different metric scale)**: In-pass normalized `live_train_metric` at pass 60 is `0.099292` (`tokengt`) vs `0.055251` (`uniform_wd_multi_ema`), consistent with reduced optimization fit rather than pure late-epoch overfitting under identical capacity (`+0.98%` parameters). Normalized online training error and fixed-weight EMA dev MAE are distinct quantities; no exact train-dev gap is subtracted.

## Error Structure

- **Not scalar bias**: Optimal constant shift removes `0.25%` of the gap (shift-corrected delta `+67.608968 meV`).
- **Row win/loss**: `17,397` wins / `32,603` losses / `0` ties (`34.79%` candidate win rate, median row gain `-41.328 meV`).
- **Universal tail & regime degradation**:
  - Quantiles (cand vs ctrl): `p50` `0.145938` vs `0.090565` (`+55.373 meV`), `p75` `0.279618` vs `0.181328` (`+98.290 meV`), `p90` `0.454658` vs `0.315614` (`+139.044 meV`), `p95` `0.594538` vs `0.425192` (`+169.347 meV`), `p99` `0.986941` vs `0.745411` (`+241.530 meV`).
  - HOMO-LUMO gap strata: `<4.0 eV` `+105.255 meV` (`32.78%` win), `4.0-6.0 eV` `+65.580 meV` (`33.95%` win), `6.0-8.0 eV` `+61.522 meV` (`37.96%` win), `>=8.0 eV` `+70.055 meV` (`38.70%` win).
- **Complementarity (post-hoc descriptive only)**: `50/50` blend MAE `0.156894 eV`; grid-oracle blend (`w = 0.07` candidate) `0.138804 eV` (`+0.637 meV` over control alone), not an actionable improvement.

## Cost and Provenance

- **Process wall**: Candidate `3629.8718715s` on local `NVIDIA GeForce RTX 5060` (PyTorch `2.7.1`, fresh `tokengt.pt` initialization); baseline `9940.3136061s` on Kaggle `Tesla T4` (PyTorch `2.11.0`). Different hardware/runtimes preclude wall-ratio speed claims.
- **Evidence owners**: [measured summary](closure_v1/measured_pcqm-gptrans-tokengt-100k-20261113-s42.json), [role audit](closure_v1/roles_tokengt.json), [closure facts](closure_v1/closure_facts_tokengt.json), `D:/molgap-artifacts/runs/pcqm-gptrans-tokengt-100k-20261113-s42/v1/reports/analysis_summary.json`, and `D:/molgap-artifacts/runs/pcqm-gptrans-tokengt-100k-20261113-s42/v1/tokengt/`.
