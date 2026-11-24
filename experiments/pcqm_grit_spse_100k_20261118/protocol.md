# Pure-2D GRIT-SPSE at 100K: Protocol

2026-11-18 JST. Desktop-owned Track B pair (`nvoid912/molgap-grit-spse-r2-100k-20261118-s42`). Earlier `s42` (`145782179`) failed preflight (`ValueError: Resume is missing required optimizer moments`) due to unused final-block pair projection (`update_pair=False` added on the last `_GRITBlock`).
Arms: `pair_update_norm` (same-job GPTrans-T control) vs `grit_spse` (100% Pure-2D `GRITSPSEModel` with CPU `SPSE` 3..8-step walk/cycle counts, `P^1..P^16` pairwise RRWP, Bahdanau SignedSqrt edge-vector attention, Adaptive Degree Scaler `ADS`, and Pre-LN SwiGLU across 8 blocks).

## Fixed Contract
- Split: `pcqm_fixed_100k_v1` (100K train, 50K internal development). Official validation and test roles remain sealed.
- Training: AdamW (`lr=1e-3`, `wd=0.05`, 4-pass warmup + 60-pass cosine), `batch_size=128`, FP32/no-TF32, EMA `0.999` (aux EMA `0.9999`), seed `42`, T4x2 one arm per device.
- Material gate: `grit_spse - pair_update_norm <= -3.0 meV` on selected EMA dev MAE and paired 2000-bootstrap 95% upper bound `< 0`.
