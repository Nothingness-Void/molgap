# Phase 1: Traditional ML Baseline

> Historical method and evidence. Live project state is in `CURRENT_STATE.md`.

## Goal
Establish ML upper bound on the simplest subset (CHON, MW 200-300).

## Data
- Source: PubChemQC B3LYP/6-31G*, 10k→30k molecules
- Elements: C, H, O, N only
- MW: 200-300
- Features: RDKit descriptors + Morgan fingerprints

## Best result
LightGBM Optuna: MAE=0.150, R²=0.921

## Key findings
- LightGBM > XGBoost > CatBoost > Stacking
- ChemBERTa/MolFormer embeddings: no improvement
- 10k→30k: +3% R² gain

## Scripts
| Script | Purpose |
|--------|---------|
| `production/history/scripts_phase1/train_baseline.py` | Ridge/RF/ExtraTrees/LightGBM baseline |
| `production/history/scripts_phase1/tune_lightgbm.py` | Optuna hyperparameter tuning |
| `production/history/scripts_phase1/advanced_models.py` | XGBoost/CatBoost/Stacking |
| `production/history/scripts_phase1/train_with_embeddings.py` | ChemBERTa/MolFormer test |
| `production/history/scripts_phase1/analyze_results.py` | Parity plots, feature importance |
| `production/history/scripts_phase1/scaffold_split_train.py` | Scaffold split evaluation |
| `production/history/scripts_phase1/confidence_analysis.py` | Prediction confidence |
| `production/history/scripts_phase1/y_randomization.py` | Y-randomization sanity check |
| `production/history/scripts_phase1/feature_contribution_analysis.py` | Feature group importance |

## Results
`results/phase1/` — baseline metrics, tuning params, parity plots, feature importance

## Dependencies
- `production/01_acquire/scripts/fetch_stream.py` → `production/01_acquire/scripts/clean.py` → `production/01_acquire/scripts/features.py` (data prep)
- Output split indices: `results/common/train_valid_test_split_indices.npz`
