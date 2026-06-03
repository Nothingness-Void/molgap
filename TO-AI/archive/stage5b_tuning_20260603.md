# Stage 5b LightGBM Hyperparameter Tuning Archive

Archived at: 2026-06-03.

## Implemented file

```text
scripts/experiments/13_tune_lightgbm.py
```

## Dependencies added

```text
optuna
```

## Run command

```bash
python scripts/experiments/13_tune_lightgbm.py --input data/processed/features_selected.csv --n-trials 30
```

## Best hyperparameters (Optuna, 30 trials, TPE sampler)

```json
{
  "n_estimators": 800,
  "learning_rate": 0.0607,
  "num_leaves": 39,
  "max_depth": 10,
  "min_child_samples": 23,
  "subsample": 0.888,
  "colsample_bytree": 0.604,
  "reg_alpha": 0.00556,
  "reg_lambda": 0.00920
}
```

Best validation avg MAE: 0.1543

## Test results — tuned vs baseline

```text
Random split:
  baseline avg MAE = 0.1755, avg R2 = 0.8952
  tuned    avg MAE = 0.1522, avg R2 = 0.9117  (MAE -13.3%)

Scaffold split:
  baseline avg MAE = 0.2047, avg R2 = 0.8642
  tuned    avg MAE = 0.1899, avg R2 = 0.8815  (MAE -7.2%)
```

## Per-target test results (tuned)

```text
Random split:
  HOMO: MAE=0.1355  RMSE=0.2010  R2=0.882
  LUMO: MAE=0.1413  RMSE=0.2126  R2=0.946
  Gap:  MAE=0.1799  RMSE=0.2906  R2=0.908

Scaffold split:
  HOMO: MAE=0.1637  RMSE=0.2267  R2=0.851
  LUMO: MAE=0.1766  RMSE=0.2532  R2=0.917
  Gap:  MAE=0.2294  RMSE=0.3139  R2=0.876
```

## Outputs

```text
results/tuning/best_params.json
results/tuning/optuna_study_summary.csv
results/tuning/tuned_vs_baseline_comparison.csv
results/tuning/tuned_test_predictions_random.csv
results/tuning/tuned_test_predictions_scaffold.csv
results/tuning/tuning_result_summary.json
models/tuned_lightgbm_random.joblib
models/tuned_lightgbm_scaffold.joblib
```

## Additional: 80-trial run (full 5451 features, wider search)

The first 80-trial run (on full features, n_estimators up to 1500) also completed:

```text
Best valid MAE: 0.1527
Best params: n_estimators=1400, num_leaves=76, max_depth=12

Random test:  avg MAE=0.1476, avg R2=0.9147
Scaffold test: avg MAE=0.1907, avg R2=0.8778
```

Comparison: 80-trial (full features) is marginally better on random split but worse on scaffold split than the 30-trial (selected features). The extra features and trees do not improve generalization. The selected-feature model is preferred for deployment (faster, better scaffold generalization).

Note: the 80-trial run overwrote the results/tuning/ directory. The saved models and JSONs now reflect the 80-trial params. To use the 30-trial params, re-run with `--input features_selected.csv --n-trials 30`.

## Stage 5 overall conclusions

1. Best feature set: Morgan + RDKit descriptors (confirmed by benchmark).
2. Best model: LightGBM (strongly outperforms Ridge across all feature sets).
3. Blend gap (alpha=0.3-0.6) gives a small improvement over direct gap prediction, especially on scaffold split.
4. Tuning improved MAE by 13.3% (random) and 7.2% (scaffold).
5. Scaffold R2=0.88 is strong enough; embedding experiments are optional enhancement, not required.
