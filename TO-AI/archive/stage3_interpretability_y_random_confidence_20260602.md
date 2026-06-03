# Stage 3 Interpretability, Y-Randomization, and Confidence Archive

Archived at: 2026-06-02.

## Implemented scripts

```text
src/05_analyze_results.py
src/06_y_randomization.py
src/07_confidence_analysis.py
```

Shared utilities updated:

```text
src/utils.py
```

Added helpers:

```text
load_model_bundle()
load_split_indices_or_raise()
get_feature_target_arrays()
```

## 1. Interpretability and error analysis

Command:

```bash
python src/05_analyze_results.py
```

Outputs:

```text
results/analysis/target_metrics_summary.csv
results/analysis/parity_homo.png
results/analysis/parity_lumo.png
results/analysis/parity_gap.png
results/analysis/residual_homo.png
results/analysis/residual_lumo.png
results/analysis/residual_gap.png
results/analysis/top_errors_lightgbm.csv
results/analysis/feature_importance_lightgbm.csv
results/analysis/feature_importance_homo.png
results/analysis/feature_importance_lumo.png
results/analysis/feature_importance_gap.png
```

Per-target LightGBM test metrics:

```text
HOMO: MAE=0.1441, RMSE=0.2099, R2=0.8709
LUMO: MAE=0.1588, RMSE=0.2334, R2=0.9346
Gap : MAE=0.1981, RMSE=0.3028, R2=0.8997
```

Interpretation:

- LUMO has the strongest R2.
- HOMO has the lowest R2 among the three targets.
- Gap has the largest MAE/RMSE and should be watched in later optimization.

## 2. Y-randomization analysis

Command:

```bash
python src/06_y_randomization.py --n-runs 20
```

Outputs:

```text
results/y_randomization/y_randomization_summary.csv
results/y_randomization/y_randomization_summary.json
results/y_randomization/y_randomization_r2_distribution.png
results/y_randomization/y_randomization_mae_distribution.png
```

Result:

```text
real avg MAE: 0.16699581058030824
real avg R2 : 0.9016993066744492
random avg MAE mean: 0.6607
random avg R2  mean: -0.0720
```

Interpretation:

The randomized-label models collapse to near-zero/negative R2 and much worse MAE. This supports that the real model is learning meaningful structure-property relationships rather than chance correlations or obvious leakage.

## 3. Confidence / uncertainty proxy analysis

Command:

```bash
python src/07_confidence_analysis.py
```

Outputs:

```text
results/confidence/confidence_predictions.csv
results/confidence/confidence_summary.csv
results/confidence/error_vs_uncertainty.png
results/confidence/error_by_confidence_bin.csv
results/confidence/error_by_confidence_bin.png
results/confidence/applicability_distance_summary.csv
```

Method:

- Model-disagreement uncertainty from ExtraTrees, RandomForest, and LightGBM predictions.
- Applicability-domain distance from nearest neighbor in scaled PCA feature space.
- Combined rank-based uncertainty score.
- High / medium / low confidence bins.

Observed confidence-bin trend:

```text
High confidence:   n=335, gap MAE=0.1210, gap disagreement=0.0378
Medium confidence: n=332, gap MAE=0.2123, gap disagreement=0.0705
Low confidence:    n=333, gap MAE=0.2613, gap disagreement=0.1267
```

Interpretation:

The confidence proxy is useful: lower-confidence bins show higher observed errors, especially for gap. This can be used later when ranking commercial molecules: predictions with high uncertainty should be treated as screening suggestions, not firm values.

## Notes and caveats

- These confidence values are uncertainty proxies, not calibrated probability intervals.
- The current baseline is still random-split based. Scaffold split should be implemented next to test new-scaffold generalization.
- SHAP was intentionally not added in this pass to avoid extra dependency/runtime. Built-in LightGBM feature importance is used first.

## Recommended next step

Stage 4 should add scaffold split evaluation:

```text
random split vs scaffold split
```

Recommended implementation:

```text
src/08_scaffold_split_train.py
```

Expected outputs:

```text
results/scaffold/model_comparison_scaffold.csv
results/scaffold/test_predictions_lightgbm_scaffold.csv
results/scaffold/scaffold_split_summary.csv
```
