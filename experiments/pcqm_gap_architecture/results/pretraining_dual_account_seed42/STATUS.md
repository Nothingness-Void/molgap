# Dual-account pretraining screen status

Both frozen seed-42 paired screens were submitted on 2026-09-08 and reached
Kaggle `RUNNING` state:

- Kaggle1: `nothingnessvoid/molgap-pcqm-structure-pretrain-s42`, version 1.
- Kaggle2: `kaseichou/molgap-pcqm-etkdg-denoise-pretrain-s42`, version 1.

Each job requested T4x2 and hard-fails before loading the accepted cache unless
exactly two Tesla T4 devices are visible. Terminal outputs have not yet been
accepted. Task ordering remains owned by `ROADMAP.md`; the frozen experiment
contract is `pretraining_dual_account_seed42_protocol.md`.
