# Phase 8 1M Candidate External Evaluation

## Scope

Kaggle P100 evaluation of the 1M dual-GPS + SchNet FusionHead candidate against
the selected routed dual-GPS v4 baseline. Both models used the same
deterministic ETKDG conformer for each molecule (`seed=42 + row_index`).
The shared external labels contain 999 OOD-1000 and 981 P8 targeted-hard rows;
three rows failed ETKDG graph construction, leaving 1,977 valid molecules.

The v4 route was recomputed, not read from historical predictions: use the v4
dual-GPS fusion only when its base-fusion predicted Gap is below 4 eV.

## Results

| scope | n | v4 avg MAE | 1M avg MAE | delta | v4 Gap MAE | 1M Gap MAE | delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| all | 1,977 | 0.103654 | 0.101887 | -0.001767 | 0.122581 | 0.118702 | -0.003878 |
| OOD-1000 | 999 | 0.112783 | 0.114312 | +0.001530 | 0.132652 | 0.133262 | +0.000610 |
| P8 targeted hard | 978 | 0.094329 | 0.089194 | -0.005135 | 0.112293 | 0.103830 | -0.008464 |

Paired bootstrap evidence for the candidate minus v4 Gap MAE:

- all: `-0.003878 eV`, 95% CI `[-0.006982, -0.000834]`;
- OOD-1000: `+0.000610 eV`, 95% CI `[-0.003322, +0.004618]`;
- P8 targeted hard: `-0.008464 eV`, 95% CI `[-0.013451, -0.003913]`.

## Decision

**Do not promote the 1M candidate over routed-v4 as the global B3LYP default.**
The targeted-hard gain is real and material, but the broad OOD-1000 block does
not improve and the all-set average improvement is not significant at 95% CI.
Keep the candidate as a documented hard-coverage specialist until a future,
separately justified routing policy can improve both blocks without reopening
the archived dynamic-router branch.

## Artifacts

- `common_eval_kaggle_metrics.json`: complete metrics and bootstrap draws summary.
- `common_eval_kaggle_predictions.csv`: paired per-molecule predictions and errors.
- Kaggle kernel: `nothingnessvoid/molgap-1m-external-evaluation`, version 4.
