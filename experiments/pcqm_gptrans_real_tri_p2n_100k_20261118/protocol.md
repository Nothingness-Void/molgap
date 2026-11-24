# Real-Atom Triangle + Pair-to-Node Feedback at 100K: Protocol

2026-11-18 JST. Desktop-owned Track B pair (`nothingnessvoid/molgap-gptrans-real-tri-p2n-100k-20261118-s42`).
Arms: `pair_update_norm` (same-job control) vs `real_tri_p2n` (adds pre-GPA real-atom triangle update and real-atom Pair->Node vector feedback at layers 2, 4, 6, 8, 10 of `gptrans_t/2`).

## Fixed Contract
- Split: `pcqm_fixed_100k_v1` (100K train, 50K internal development). Official validation and test roles remain sealed.
- Init & Training: shared `gptrans.pt`, AdamW (`lr=1e-3`, `wd=0.05`, 4-pass warmup + 60-pass cosine), `batch_size=128`, FP32/no-TF32, EMA `0.999`, seed `42`, T4x2 one arm per device, `16800s` wall ceiling.
- Material gate: `real_tri_p2n - pair_update_norm <= -3.0 meV` on selected EMA dev MAE and paired 2000-bootstrap 95% upper bound `< 0`.
