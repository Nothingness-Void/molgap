# K1 500K BN-Free GraphRMSNorm Salvage under Batched Muon: Protocol

2026-11-17 JST. Desktop-owned Track B pair (`kaseichou/molgap-k1-graph-rms-muon-500k-20261117-s42`).
Arms: `muon_bn` (original 18-BN K1) vs `muon_graph_rms` (all 18 `BatchNorm1d` sites replaced by per-graph channel `GraphRMSNorm` preserving `k1.pt` affine parameters), both trained under identical `k1-muon-batched-adamw-fp32-v1`.

## Fixed Contract & Continuation
- Data: `pcqm_fixed_500k_v1` (500K train, 50K internal development). Official validation and test roles remain sealed.
- Training: `batch_size=128`, FP32/no-TF32, 60 passes (`234,360` steps), `lr=4e-4`, `wd=1e-5`, `cosine_epoch`, selection `best-development-live`, T4x2 one arm per device.
- Two physical Kaggle sessions per run: planned pause at step `187,488` (pass 48), continuation to terminal step `234,360` (pass 60).
- Material gate: `muon_graph_rms - muon_bn <= -3.0 meV` on selected Live dev MAE and paired 2000-bootstrap 95% upper bound `< 0`.
