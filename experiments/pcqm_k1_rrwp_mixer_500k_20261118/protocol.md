# K1 500K RRWP + Global Atom Mixer under Batched Muon: Protocol

2026-11-18 JST. Desktop-owned Track B pair (`nvoid912/molgap-k1-rrwp-mixer-500k-20261118-s42`).
Arms: `muon_bn` (original 18-BN K1) vs `muon_rrwp_mixer` (18-BN K1 + `k1_rrwp_mixer`: 16-step symmetric RRWP edge injection at `edge_updates.1..8` and 6-head global atom self-attention with RRWP + persistent bond bias at `neural_atom_mixers.3,6,9`), both trained under identical `k1-muon-batched-adamw-fp32-v1`.

## Fixed Contract & Continuation
- Data: `pcqm_fixed_500k_v1` (500K train, 50K internal development). Official validation and test roles remain sealed.
- Training: `batch_size=128`, FP32/no-TF32, 60 passes (`234,360` steps), `lr=4e-4`, `wd=1e-5`, `cosine_epoch`, selection `best-development-live`, T4x2 one arm per device.
- Two physical Kaggle sessions per run: planned pause at step `156,240` (pass 40), continuation to terminal step `234,360` (pass 60).
- Material gate: `muon_rrwp_mixer - muon_bn <= -3.0 meV` on selected Live dev MAE and paired 2000-bootstrap 95% upper bound `< 0`.
