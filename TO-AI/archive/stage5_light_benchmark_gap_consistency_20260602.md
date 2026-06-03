# Stage 5 Lightweight Benchmark and Gap Consistency Archive

Archived at: 2026-06-02.

## Implemented files

```text
src/10_light_benchmark.py
src/11_gap_consistency_analysis.py
```

## Run command

```bash
python src/10_light_benchmark.py && python src/11_gap_consistency_analysis.py
```

First run failed only at plotting because of an invalid seaborn `barplot(col=None)` argument. The script was fixed and re-run successfully.

## 1. Lightweight feature/model benchmark

Purpose:

Compare lightweight feature sets and models suitable for the user's 16 GB RAM environment.

Feature sets:

```text
morgan_only
rdkit_desc_only
morgan_plus_rdkit
```

Models:

```text
ridge
lightgbm
```

Splits:

```text
random
scaffold
```

Outputs:

```text
results/benchmark/light_feature_model_benchmark.csv
results/benchmark/light_feature_model_benchmark_summary.csv
results/benchmark/light_feature_model_benchmark_best_by_split.csv
results/benchmark/benchmark_average_r2.png
results/benchmark/benchmark_average_mae.png
```

Best test result by split:

```text
Random split:
  best = morgan_plus_rdkit + lightgbm
  average MAE = 0.1755
  average R2  = 0.8952

Scaffold split:
  best = morgan_plus_rdkit + lightgbm
  average MAE = 0.2047
  average R2  = 0.8642
```

Important note:

This benchmark uses a lighter LightGBM setting (`n_estimators=300`) to keep runtime low, so its numbers are slightly below the main baseline (`n_estimators=500`). The ranking is still useful.

Key benchmark insights:

```text
Morgan + RDKit is best on both random and scaffold splits.
RDKit descriptors only are surprisingly strong and much better than Morgan only.
Morgan only is the weakest, especially on scaffold split.
LightGBM strongly outperforms Ridge across all feature sets.
```

Selected test results:

```text
Random / Morgan+RDKit / LightGBM:
  avg MAE = 0.1755, avg R2 = 0.8952

Random / RDKit only / LightGBM:
  avg MAE = 0.1850, avg R2 = 0.8867

Random / Morgan only / LightGBM:
  avg MAE = 0.2266, avg R2 = 0.8290

Scaffold / Morgan+RDKit / LightGBM:
  avg MAE = 0.2047, avg R2 = 0.8642

Scaffold / RDKit only / LightGBM:
  avg MAE = 0.2153, avg R2 = 0.8561

Scaffold / Morgan only / LightGBM:
  avg MAE = 0.2943, avg R2 = 0.7215
```

## 2. Gap consistency analysis

Purpose:

Compare direct gap prediction with physically consistent gap computed from predicted orbitals:

```text
gap_orbital = lumo_pred - homo_pred
```

Also tested blended gap:

```text
gap_blend = alpha * gap_direct + (1 - alpha) * gap_orbital
```

Alpha grid:

```text
0.0, 0.1, ..., 1.0
```

Outputs:

```text
results/gap_consistency/gap_strategy_comparison.csv
results/gap_consistency/gap_strategy_best_by_split.csv
results/gap_consistency/random_gap_predictions_with_strategies.csv
results/gap_consistency/scaffold_gap_predictions_with_strategies.csv
results/gap_consistency/gap_strategy_mae.png
results/gap_consistency/gap_strategy_r2.png
```

Best strategy:

```text
Random split:
  best = blend_gap alpha=0.6
  gap MAE = 0.1920
  gap RMSE = 0.2977
  gap R2 = 0.9030

Scaffold split:
  best = blend_gap alpha=0.3
  gap MAE = 0.2161
  gap RMSE = 0.2804
  gap R2 = 0.8366
```

Key gap insights:

```text
Random split:
  direct gap MAE = 0.1981, R2 = 0.8997
  orbital gap MAE = 0.2009, R2 = 0.8952
  blend gap improves slightly to MAE = 0.1920, R2 = 0.9030

Scaffold split:
  direct gap MAE = 0.2370, R2 = 0.8027
  orbital gap MAE = 0.2221, R2 = 0.8283
  blend gap improves further to MAE = 0.2161, R2 = 0.8366
```

Interpretation:

For random split, direct gap and orbital-derived gap are close, but blending gives a small improvement. For scaffold split, orbital-derived gap is clearly better than direct gap, and blending is best. This suggests gap prediction for new scaffolds benefits from enforcing partial physical consistency with HOMO/LUMO predictions.

## Recommended next step

Because RDKit descriptors are strong and gap benefits from blended consistency, the next useful work is not data expansion but targeted lightweight optimization:

1. Add a `gap_blend_alpha` option in prediction/database outputs.
2. Add a small LightGBM tuning script focused only on `morgan_plus_rdkit` and `rdkit_desc_only`.
3. Consider an interpretable descriptor subset benchmark for material-relevant descriptors.
