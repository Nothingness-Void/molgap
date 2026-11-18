# DeMol-CIN 100K: Protocol

2026-11-18 JST. Desktop-owned Track B pair (`nothingnessvoid/molgap-demol-cin-100k-20261118-s42`).
Arms: `pair_update_norm` (same-job control from `gptrans.pt`) vs `demol_cin` (standalone Pure-2D sparse cellular complex network with 0-cell atoms, 1-cell undirected bonds + line-graph BLA, 2-cell chordless rings of size 3–8 capped at 16, and an $O(|V|+|E_u|+|R|)$ virtual hub).

## Fixed Contract
- Split: `pcqm_fixed_100k_v1` (100K train, 50K internal development). Official validation and test roles remain sealed.
- Training: AdamW (`lr=1e-3`, `wd=0.05`, 4-pass warmup + 60-pass cosine), `batch_size=128`, FP32/no-TF32, EMA `0.999` (+ aux EMA `0.9999`), seed `42`, T4x2 one arm per device.
- Material gate: `demol_cin - pair_update_norm <= -3.0 meV` on selected EMA dev MAE and paired 2000-bootstrap 95% upper bound `< 0`.
