# RRWP + SwiGLU PairTransition at 100K: Protocol

2026-11-18 JST. Desktop-owned Track B pair (`kaseichou/molgap-gptrans-rrwp-pairffn-100k-20261118-s42`).
Arms: `pair_update_norm` (same-job control) vs `rrwp_pairffn` (adds 16-step symmetric RRWP pair injection at block 0 and Pre-LN SwiGLU `PairTransition(32 -> 64 -> 32)` across all 12 blocks of `gptrans_t/2`).

## Fixed Contract
- Split: `pcqm_fixed_100k_v1` (100K train, 50K internal development). Official validation and test roles remain sealed.
- Init & Training: shared `gptrans.pt`, AdamW (`lr=1e-3`, `wd=0.05`, 4-pass warmup + 60-pass cosine), `batch_size=128`, FP32/no-TF32, EMA `0.999`, seed `42`, T4x2 one arm per device, 4.0h wall ceiling (`14400s`).
- Material gate: `rrwp_pairffn - pair_update_norm <= -3.0 meV` on selected EMA dev MAE and paired 2000-bootstrap 95% upper bound `< 0`.
