# Official validation MAE of the full K1 Muon B1024 x6 run

Report-only scoring of two checkpoints of the finished Colab run
`k1-b1024-lr6x-full-20261109` on the PCQM4Mv2 official valid split.
[Decision](decision.md) owns the authority, plan, gate, and result;
[STATUS](STATUS.md) owns the dated closure state.

Script: `platforms/local/pcqm_official_valid_eval.py` (under `platforms/`,
not here). `test_dev` and `test_challenge` were never read. No per-row
prediction files were written; evidence paths are listed in the decision.
