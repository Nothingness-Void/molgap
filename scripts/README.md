# Script Layout

## Pipeline
- `scripts/pipeline/01_fetch_stream.py`
- `scripts/pipeline/02_clean.py`
- `scripts/pipeline/03_features.py`
- `scripts/pipeline/04_train_baseline.py`
- `scripts/pipeline/08_scaffold_split_train.py`

## Evaluation
- `scripts/evaluation/05_analyze_results.py`
- `scripts/evaluation/06_y_randomization.py`
- `scripts/evaluation/07_confidence_analysis.py`
- `scripts/evaluation/11_gap_consistency_analysis.py`
- `scripts/evaluation/12_feature_contribution_analysis.py`

## Experiments
- `scripts/experiments/10_light_benchmark.py`

## Deferred TODO
- `scripts/todo/09_predict_commercial.py`

## Shared Code
- `src/molgap/utils.py`

## Common Commands
```bash
python scripts/pipeline/02_clean.py
python scripts/pipeline/03_features.py
python scripts/pipeline/04_train_baseline.py
python scripts/evaluation/05_analyze_results.py
python scripts/pipeline/08_scaffold_split_train.py
python scripts/experiments/10_light_benchmark.py
```
