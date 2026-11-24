# Real-Atom Triangle Pair Update + L12 Virtual-Query Pair Pooling Readout at 100K: Protocol

2026-11-18 JST. Desktop-owned Track B pair (`nvoid912/molgap-gptrans-real-tri-vread-100k-20261118-s42`).
Arms: `pair_update_norm` (same-job control) vs `real_tri_vread` (adds asymmetric low-rank real-atom triangle multiplicative pair update at layers 2, 4, 6, 8, 10 plus L12 virtual-query attention pooling over real-atom pair states into `pair[:,:,0,0]` of `gptrans_t/2`).

## Fixed Contract
- Split: `pcqm_fixed_100k_v1` (100K train, 50K internal development). Official validation and test roles remain sealed.
- Init & Training: shared `gptrans.pt`, AdamW (`lr=1e-3`, `wd=0.05`, 4-pass warmup + 60-pass cosine), `batch_size=128`, FP32/no-TF32, EMA `0.999`, seed `42`, T4x2 one arm per device.
- Material gate: `real_tri_vread - pair_update_norm <= -3.0 meV` on selected EMA dev MAE and paired 2000-bootstrap 95% upper bound `< 0`.
