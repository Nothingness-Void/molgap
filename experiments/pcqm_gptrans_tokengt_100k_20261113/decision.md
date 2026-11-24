# TokenGT-GPTrans 100K Terminal Decision

2026-11-13 JST. Local RTX 5060 run `pcqm-gptrans-tokengt-100k-20261113-s42` (`attempt v1`).

`tokengt` closes as **`NEGATIVE_UNDER_CONTRACT`**. Do not retry, tune hyperparameters, run a multi-seed check, or advance to 500K/full scale.

Under the frozen 60-pass (`46,860` steps, `5,998,080` presentations) `pcqm_fixed_100k_v1` contract (`lr = 1e-3`, AdamW, EMA `0.999`, FP32/no-TF32, seed `42`), `TokenGTGPTrans` (`gptrans_t/2`, `5,928,961` parameters) selects pass `58` (`optimizer_steps = 45298`, EMA) with `50,000`-row development MAE **`0.207216078 eV`**, regressing by **`+67.775886 meV`** against the retained `uniform_wd_multi_ema` baseline (`0.139440192 eV`, pass `42`, `OGBGPTransTiny`, `5,871,201` parameters).

- **Paired difference (`tokengt - uniform_wd_multi_ema`)**: `+67.775886 meV`, paired row bootstrap 95% CI **`[+66.040341, +69.435555] meV`** (`10,000` resamples, seed `42`, `50,000` aligned development rows).
- **Predeclared gate (`<= -3.0 meV` and negative upper 95% CI)**: Failed (`passes_material_gate = false`, `passes_noninferiority_gate = false`, `probability_candidate_better = 0.0`).
- **Caveats**: Cross-hardware/runtime execution (local RTX 5060 PyTorch `2.7.1` with fresh `tokengt.pt` initialization vs Kaggle Tesla T4 PyTorch `2.11.0` `OGBGPTransTiny`) and single training seed (`42`); paired row bootstrap measures row sampling uncertainty rather than training-seed variance. Detailed breakdown is in [attribution](attribution.md).
