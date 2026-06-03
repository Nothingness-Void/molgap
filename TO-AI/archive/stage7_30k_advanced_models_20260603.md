# Stage 7 — 30k Data Expansion + Advanced Models Archive

Archived at: 2026-06-03.

## What was done

### 1. Data expansion: 10k → 30k
```bash
python scripts/pipeline/01_fetch_stream.py --run --max-records 30000 --chunk-bytes 100000000
```
Result: 30000 raw → 29257 after clean+features → 2372 selected features.

### 2. Optuna LightGBM tuning on 30k (30 trials)
Best params:
```json
{
  "n_estimators": 800,
  "learning_rate": 0.069,
  "num_leaves": 74,
  "max_depth": 10,
  "min_child_samples": 19,
  "subsample": 0.924,
  "colsample_bytree": 0.643,
  "reg_alpha": 0.0003,
  "reg_lambda": 3.17
}
```

### 3. Results comparison: 10k vs 30k

```text
10k tuned LightGBM:
  Random:   avg MAE=0.1522  R2=0.9117
  Scaffold: avg MAE=0.1899  R2=0.8815

30k tuned LightGBM:
  Random:   avg MAE=0.1498  R2=0.9205  (MAE improved 1.6%)
  Scaffold: avg MAE=0.1851  R2=0.8799  (MAE improved 2.5%)
```

### 4. Per-target results (30k tuned, random split)
```text
HOMO: MAE=0.1298  RMSE=0.1803  R2=0.8955
LUMO: MAE=0.1416  RMSE=0.2048  R2=0.9481
Gap:  MAE=0.1780  RMSE=0.2638  R2=0.9180
```

### 5. Advanced models script
`scripts/experiments/15_advanced_models.py` was created to compare:
- Tuned LightGBM (baseline)
- Per-target LightGBM
- Stacking ensemble (LightGBM + ExtraTrees + Ridge)
- DART LightGBM
- XGBoost
- CatBoost

Status: COMPLETED.

Results ranking (30k, random split test):
```text
1. xgboost           avg MAE=0.1521  R2=0.9178
2. per_target_lgbm   avg MAE=0.1538  R2=0.9186
3. tuned_lgbm        avg MAE=0.1561  R2=0.9176
4. stacking          avg MAE=0.1572  R2=0.9173
5. catboost           avg MAE=0.1740  R2=0.9035
6. dart_lgbm          avg MAE=0.3078  R2=0.7178
```

XGBoost is marginally the best by MAE. Per-target LightGBM has slightly higher R². Stacking did not improve over single models. CatBoost underperformed. DART failed badly.

### 6. Dependencies added
- xgboost
- catboost
- optuna (already added in stage 5b)

## New files created
```text
scripts/experiments/15_advanced_models.py
```

## Key insight
Data expansion from 10k to 30k provided a meaningful improvement (R² 0.912 → 0.921). Further expansion to 50k-100k would likely continue to help, but requires longer fetch times.
