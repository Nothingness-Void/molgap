# K1 500K Line-Graph Bond Global Self-Attention + Joint Readout: Protocol

2026-11-18 JST. Desktop-owned Track B pair (`nothingnessvoid/molgap-k1-line-graph-bond-500k-20261118-s42`).
Arms: `muon_bn` (original K1) vs `muon_line_graph` (K1 with `k1_line_graph_bond` addon: undirected bond line-graph 4-head global self-attention with shortest-path distance bias at `edge_updates.{2,5,8}` and bond-aware head readout), both trained under identical `k1-muon-batched-adamw-fp32-v1`.

## Fixed Contract & Continuation
- Data: `pcqm_fixed_500k_v1` (500K train, 50K internal development). Official validation and test roles remain sealed.
- Training: `batch_size=128`, FP32/no-TF32, 60 passes (`234,360` steps), `lr=4e-4`, `wd=1e-5`, `cosine_epoch`, selection `best-development-live`, T4x2 one arm per device.
- Two physical Kaggle sessions per run: planned pause at step `156,240` (pass 40), continuation to terminal step `234,360` (pass 60).
- Material gate: `muon_line_graph - muon_bn <= -3.0 meV` on selected Live dev MAE and paired 2000-bootstrap 95% upper bound `< 0`.
