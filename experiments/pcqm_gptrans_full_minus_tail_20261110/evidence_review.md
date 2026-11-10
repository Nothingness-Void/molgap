# Evidence

- Earlier GPT-T full (6 and 12 epochs) reached 0.1090 and 0.1038 eV on official validation, worse than K1 and EdgeState GPS; those used official validation for selection.
- The K1 B1024x6 full run (60 epochs, 0.09934 eV, unselected) showed train MAE 0.0850 and no diagnosis, because the frozen full recipe has no held-out role.
- 500K K1 runs rank identically on the 50K development role and on tail_holdout_v1 (platforms/local/pcqm_tail_holdout_eval.py), with tail about 7 meV higher.
- fusion of K1 and GPT-T was closed at 0.1022 eV with weaker recipes.
