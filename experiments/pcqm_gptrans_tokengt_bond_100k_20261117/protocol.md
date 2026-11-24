# TokenGT Bond-Token Debut at 100K: Protocol

2026-11-17 JST. Desktop-owned Track B pair (`nothingnessvoid/molgap-gptrans-tokengt-bond-100k-20261117-s42`).
Arms: `pair_update_norm` (same-job control) vs `tokengt_bond` (adds explicit undirected bond tokens, exact atom-bond incidence bias, RMS QK-Norm, and head sigmoid output gating at layers 3, 6, 9, 12 of `gptrans_t/2`).

## Fixed Contract
- Data: `pcqm_fixed_100k_v1` (100K train, 50K internal development). Official validation and test roles remain sealed.
- Init & Training: shared `gptrans.pt`, AdamW (`lr=1e-3`, `wd=0.05`, 4-pass warmup + 60-pass cosine), `batch_size=128`, FP32/no-TF32, EMA `0.999`, seed `42`, T4x2 one arm per device.
- Material gate: `tokengt_bond - pair_update_norm <= -3.0 meV` on selected EMA dev MAE and paired 2000-bootstrap 95% upper bound `< 0`.
