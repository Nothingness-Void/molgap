# Phase 4: GNN SchNet + ETKDG Consistency

> Historical method and evidence. Live project state is in `CURRENT_STATE.md`.

## Goal
Introduce 3D GNN (SchNet) and establish train-inference consistency with ETKDG conformers.

## Data
- Same 30k as Phase 3, CHONSFCl, MW 200-503
- 3D coordinates: ETKDG (RDKit) or PM6 (from PubChemQC)

## Results
| Model | MAE | R² | Consistent? |
|-------|-----|-----|-------------|
| AttentiveFP (2D) | 0.163 | 0.879 | yes |
| SchNet PM6 baseline | 0.113 | 0.930 | NO |
| SchNet PM6 Optuna | 0.095 | 0.950 | NO |
| SchNet ETKDG baseline | 0.155 | 0.885 | yes |
| **SchNet ETKDG Optuna** | **0.147** | **0.896** | **yes** |

## Critical finding: Train-inference consistency
PM6 conformers (from PubChemQC) give higher R² in testing, but at inference time only ETKDG is available (no PM6 for new molecules). Using PM6 for training + ETKDG for inference = distribution mismatch → unreliable predictions. **Must use ETKDG for both.**

## Scripts
| Script | Purpose |
|--------|---------|
| `scripts/phase4/gnn_schnet_3d.py` | SchNet training (baseline parameters) |
| `scripts/phase4/schnet_optuna.py` | Optuna hyperparameter search |
| `scripts/phase4/_retrain_best.py` | Retrain with best Optuna params |

## Results
`results/phase4/`

## Model
`models/gnn_schnet_3d_tuned.pt` (Phase 4 best, superseded by Phase 6)

## Dependencies
- `data/raw/phase3_chonsfcl_mw200_500_30k.csv`
- `src/molgap/schnet.py` (SchNetWrapper)
- `src/molgap/utils.py` (graph building, splits, metrics)
