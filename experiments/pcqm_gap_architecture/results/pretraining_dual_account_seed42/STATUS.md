# Dual-account pretraining screen status

Both frozen seed-42 paired screens completed on 2026-09-08 and passed the
no-model mechanical acceptance:

- Kaggle1: `nothingnessvoid/molgap-pcqm-structure-pretrain-s42`, version 1.
- Kaggle2: `kaseichou/molgap-pcqm-etkdg-denoise-pretrain-s42`, version 1.

Both used two Tesla T4 devices, the same accepted cache, the same initial
GraphState9 encoder hash, and kept official validation/test-dev unread. Neither
mechanism passed the compute-normalized material-gain gate. The exact scientific
disposition is in `decision.md`; task ordering remains owned by `ROADMAP.md`.
