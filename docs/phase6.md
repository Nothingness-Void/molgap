# Phase 6: MW Expansion (200-1000)

## Goal
Expand training data to MW 200-1000 so large commercial molecules (MW>500) become interpolation, not extrapolation.

## Data
- 44,827 molecules = 30k (Phase 3, MW200-500) + ~15k (new, MW500-1000)
- Elements: CHONSFCl
- New data file: `data/raw/phase6_chonsfcl_mw500_1000_15k.csv`

## Results
| Experiment | MAE | R² | Notes |
|------------|-----|-----|-------|
| SchNet ETKDG default (P4 params) | 0.158 | 0.890 | cutoff=6.0, dropout=0.2 |
| **SchNet ETKDG Optuna** | **0.162** | **0.882** | cutoff=8.0, dropout=0.1, Colab |

### Gaussian B3LYP validation (10 commercial OLED molecules)
| Metric | Phase 4 | Phase 6 |
|--------|---------|---------|
| HOMO MAE | 0.216 | **0.184** |
| LUMO MAE | 0.196 | **0.181** |
| Gap MAE | 0.352 | **0.223** |

### OOD validation (500 PubChemQC molecules)
| Metric | Phase 4 | Phase 6 |
|--------|---------|---------|
| avg R² | 0.730 | **0.797** |
| RMSE | 0.390 | **0.335** |

## Key findings
- Internal test R² slightly dropped (0.896→0.882) due to increased data diversity
- But external validation improved across the board (Gap MAE -37%, OOD R² +9%)
- OOD R²≈0.8 is the current accuracy ceiling
- Bottlenecks: ETKDG conformer noise (~0.1 eV) + limited training coverage (44k vs 85M)

## Scripts
| Script | Purpose |
|--------|---------|
| `scripts/phase6/fetch_large_mw.py` | Fetch MW 500-1000 molecules |
| `scripts/phase6/colab_optuna.ipynb` | Optuna tuning on Colab |
| `scripts/phase6/retrain_expanded.py` | Retrain on merged 44.8k dataset |
| `scripts/phase6/test_full_dataset.py` | Test on full dataset |
| `scripts/phase6/predict_commercial_p6.py` | Commercial molecule prediction |
| `scripts/phase6/ood_validation_p6.py` | 500-mol OOD evaluation |

## Results
`results/phase6/`

## Model
`models/gnn_schnet_3d_optuna_expanded.pt` — Phase 6 best model at the time;
superseded by later Phase 7/8 models.

## Dependencies
- `data/raw/phase3_chonsfcl_mw200_500_30k.csv` (Phase 3 data)
- `data/raw/phase6_chonsfcl_mw500_1000_15k.csv` (new large MW data)
- `src/molgap/schnet.py`, `src/molgap/utils.py`
