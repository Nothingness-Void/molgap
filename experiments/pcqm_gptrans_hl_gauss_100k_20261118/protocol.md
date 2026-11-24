# HL-Gauss 512-Bin Distributional Readout at 100K: Protocol

2026-11-18 JST. Desktop-owned Track B pair (`nothingnessvoid/molgap-gptrans-hl-gauss-100k-20261118-s42`).
Arms: `pair_update_norm` (same-job control) vs `hl_gauss_512` (replaces the scalar linear readout of `gptrans_t/2` with a 512-bin HL-Gauss head over `[-4.6, 5.0]` normalized Gap units, trained with pure HL-Gauss cross-entropy and decoded via CDF median).

## Fixed Contract
- Split: `pcqm_fixed_100k_v1` (100K train, 50K internal development). Official validation and test roles remain sealed.
- Init & Training: shared `gptrans.pt` (preserving `readout.0` and `readout.1`), AdamW (`lr=1e-3`, `wd=0.05`, 4-pass warmup + 60-pass cosine), `batch_size=128`, FP32/no-TF32, EMA `0.999`, seed `42`, T4x2 one arm per device.
- Material gate: `hl_gauss_512 - pair_update_norm <= -3.0 meV` on selected EMA dev MAE and paired 2000-bootstrap 95% upper bound `< 0`.
