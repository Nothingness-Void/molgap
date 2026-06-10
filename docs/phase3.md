# Phase 3: Data Scaling + ML Optimization

## Goal
Push traditional ML to its limit on the full element/MW scope before moving to GNN.

## Data
- 30k molecules, CHONSFCl, MW 200-500
- Feature selection: 6028 → 2811 features

## Best result
LightGBM Optuna + feature selection: MAE=0.160, R²=0.885

## Key findings
- Feature selection + retuning: R² 0.876 → 0.885
- Still below R²=0.9 — motivates GNN approach

## Scripts
| Script | Purpose |
|--------|---------|
| `scripts/phase3/scaleup.py` | Fetch 30k + feature engineering |
| `scripts/phase3/select_and_optimize.py` | Feature selection + Optuna retuning |

## Results
`results/phase3/`

## Dependencies
- Pipeline scripts for data prep
- Phase 2 conclusion: CHONSFCl MW200-500 is the target scope

## Output data
`data/raw/phase3_chonsfcl_mw200_500_30k.csv` — reused by Phase 4, 6
