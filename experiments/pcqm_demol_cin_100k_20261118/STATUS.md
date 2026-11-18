# Status

- Prior run `pcqm-demol-cin-100k-20261118-s42` (`nothingnessvoid/molgap-demol-cin-100k-20261118-s42`, `kernel_id=145744975`) failed at Kaggle import (`KernelWorkerStatus.ERROR`) because `src/molgap/demol_cin.py` imported `.ogb_features`, which imports `rdkit`. Journal preserved untouched at `D:/molgap-artifacts/runs/pcqm-demol-cin-100k-20261118-s42/v1`.
- Fixed in `r2-s42` by replacing `.ogb_features` with `AtomEncoder` and `BondEncoder` from `ogb.graphproppred.mol_encoder` (zero `rdkit` dependency).
- Submitted `pcqm-demol-cin-100k-20261118-r2-s42`: `SUBMISSION_BOUND` (`nothingnessvoid/molgap-demol-cin-r2-100k-20261118-s42`, `kernel_id=145779299`, `version=1`, `KernelWorkerStatus.RUNNING`).
