# MolGap Experiment Phases

## Phase 1: Model Optimization
Fixed data scope: CHON, MW 200-300.

| Sub | 内容 | 数据 | 状态 |
|-----|------|------|------|
| 1.1 | Baseline models (Ridge, ExtraTrees, RF, LightGBM) | 10k | ✅ |
| 1.2 | Optuna LightGBM hyperparameter tuning | 10k | ✅ |
| 1.3 | Embedding experiments (ChemBERTa, MolFormer, fusion) | 10k | ✅ |
| 1.4 | Advanced models (XGBoost, CatBoost, stacking, per-target, DART) | 30k | ✅ |
| 1.5 | Data scaling (10k → 30k) | 30k | ✅ |

### Phase 1 Conclusions
- Best model: **LightGBM (Optuna tuned)** — avg MAE=0.1498, R²=0.921 on 30k
- Embeddings do not help (traditional features are better)
- XGBoost is a close alternative (MAE=0.1521)
- More data consistently helps (+3% from 10k→30k)
- Stacking/ensemble did not improve over single models

## Phase 2: Generalization Study
Fixed model: tuned LightGBM. Fixed data size: 10k per step.

| Sub | 元素 | MW | avg MAE | avg R² | 状态 |
|-----|------|-----|---------|--------|------|
| 2.1 | C,H,N,O | 200-300 | 0.162 | 0.901 | ✅ baseline |
| 2.2 | C,H,N,O | 200-500 | 0.163 | 0.889 | ✅ |
| 2.3 | C,H,N,O,S | 200-500 | 0.167 | 0.879 | ✅ |
| 2.4 | C,F,H,N,O,S | 200-500 | 0.173 | 0.878 | ✅ |
| 2.5 | C,Cl,F,H,N,O,S | 200-500 | 0.175 | 0.874 | ✅ |

### Phase 2 Conclusions
- No cliff-edge degradation: R² drops smoothly from 0.901 to 0.874
- HOMO most sensitive to diversity; LUMO most stable
- Halogen atoms (Cl) cause the biggest per-step LUMO drop

## Phase 3: Production Scale-Up (TODO)
- 3.1 Scale step4 (CHONSFCl MW 200-500) to 30k-50k
- 3.2 Retrain with best params, verify R² recovery
- 3.3 Curate commercial molecule list
- 3.4 Predict commercial molecules
- 3.5 Build final database

## Master Experiment Log
All 33 experiments are recorded in:
```
results/master_experiment_log.csv
```
Regenerate with:
```bash
python scripts/experiments/17_build_master_experiment_table.py
```
