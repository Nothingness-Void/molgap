# Stage 9: Phase 3.4 Model Optimization (2026-06-04)

## Task
在 CHONSFCl MW 200-500 30k 数据集上，使用 gain-based 特征筛选 + Optuna 多模型超参优化，争取 R² >= 0.9。

## Feature Selection
```
原始特征: 6028
保留特征: 2811 (gain > 0)
  morgan: 905, atompair: 946, torsion: 636, desc: 195, maccs: 129
砍掉:     3217 (53%)
```

## Optuna Tuning
- LightGBM: 80 trials, best valid MAE=0.1625
- XGBoost: 60 trials, best valid MAE=0.1655

### LightGBM Best Params
```json
{
  "n_estimators": 1100,
  "learning_rate": 0.0403,
  "num_leaves": 117,
  "max_depth": 13,
  "min_child_samples": 5,
  "subsample": 0.733,
  "colsample_bytree": 0.528,
  "reg_alpha": 0.133,
  "reg_lambda": 0.736
}
```

## Test Set Results (train+valid → test)
```
Model Comparison (sorted by avg R2):
  Tuned_LGBM         MAE=0.1596  R2=0.8853  ← Best
  PerTarget_LGBM     MAE=0.1596  R2=0.8853
  Tuned_XGB          MAE=0.1616  R2=0.8817
  HistGBT            MAE=0.1692  R2=0.8761
  Phase3_baseline    MAE=0.1706  R2=0.8755
  CatBoost           MAE=0.1859  R2=0.8604

Tuned LGBM per-target:
  homo : MAE=0.1369  R2=0.8530
  lumo : MAE=0.1463  R2=0.9229
  gap  : MAE=0.1958  R2=0.8801
```

## Conclusions
- Optuna + 特征筛选: MAE 0.1706→0.1596 (-6.4%), R² 0.8755→0.8853
- 未达到 R²=0.9 目标，差距 0.015
- LightGBM 仍是最佳模型，XGBoost 接近
- CatBoost 固定参数表现差，可能需要单独调参
- PerTarget 和 MultiOutput LGBM 结果完全一致

## Output Files
```
results/phase3/optimize/
  model_comparison.csv
  optimize_summary.json
  best_params_lgbm.json
  best_params_xgb.json
  selected_feature_stats.json
  feature_gain.csv
  optimize_log.txt
```
