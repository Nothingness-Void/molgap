# MolGap Workflow

## Last updated
2026-06-05

## Goal
Build a molecular-property prediction workflow for organic electronic-material molecules. The three targets are:
- `HOMO`
- `LUMO`
- `HOMO-LUMO gap`

## Current workflow structure

### Pipeline
Shared preprocessing and bookkeeping:

```text
scripts/pipeline/fetch_stream.py
scripts/pipeline/clean.py
scripts/pipeline/features.py
scripts/pipeline/feature_selection.py
scripts/pipeline/build_master_experiment_table.py
src/molgap/utils.py
```

### Phase 1 — Model Optimization
Chemistry scope:
`CHON`, `MW 200-300`

Main outputs:
- baseline comparison
- tuning
- embeddings
- advanced model comparison

Canonical scripts:

```text
scripts/phase1/train_baseline.py
scripts/phase1/tune_lightgbm.py
scripts/phase1/train_with_embeddings.py
scripts/phase1/advanced_models.py
```

### Phase 2 — Generalization Study
Expand chemistry gradually and observe degradation:

```text
scripts/phase2/generalization_study.py
```

### Phase 3 — Production Scale-Up
Harder chemistry space:
`CHONSFCl`, `MW 200-500`, `30k`

Canonical scripts:

```text
scripts/phase3/scaleup.py
scripts/phase3/select_and_optimize.py
```

### Phase 4 — Ensemble and GNN
Try to surpass the best Phase-3 LightGBM result.

Canonical scripts:

```text
scripts/phase4/ensemble_blend.py
scripts/phase4/per_target_optuna.py
scripts/phase4/gnn_attentivefp.py
scripts/phase4/gnn_schnet_3d.py
scripts/phase4/schnet_lgbm_fusion.py
scripts/phase4/comparison_report.py
```

### Phase 5 — Commercial Prediction
Exists, but not the active priority:

```text
scripts/phase5/predict_commercial.py
```

## Current best results

### Easier chemistry setting
`30k CHON, MW 200-300`
- Best model: tuned LightGBM
- `avg MAE=0.1498`
- `avg R²=0.9205`

### Harder chemistry setting
`30k CHONSFCl, MW 200-500`
- Best traditional model: tuned LightGBM
- `avg MAE=0.1596`
- `avg R²=0.8853`

- Best overall model: SchNet 3D
- `avg MAE=0.1492`
- `avg R²=0.8942`

## Canonical result directories

```text
results/common/
results/phase2/generalization/
results/phase3/
results/phase3/optimize/
results/phase4/
results/phase5/database/
```

The master experiment table is:

```text
results/master_experiment_log.csv
```

Regenerate it with:

```bash
.venv\Scripts\python.exe scripts/pipeline/build_master_experiment_table.py
```

## Current recommended focus
Do not treat commercial prediction as the current mainline. The current mainline is:

1. finalize the model comparison narrative
2. decide the final reported benchmark
3. decide whether direct gap, blended gap, or both should be emphasized
4. optionally push one more round of SchNet / scale-up if you want to try crossing `R²=0.9`

## Reproduction hints

### CPU-side best traditional model
```bash
.venv\Scripts\python.exe scripts/phase3/select_and_optimize.py --lgbm-trials 80 --xgb-trials 60
```

### Best overall model
```bash
.venv\Scripts\python.exe scripts/phase4/gnn_schnet_3d.py
```

### Final comparison rebuild
```bash
.venv\Scripts\python.exe scripts/phase4/comparison_report.py
.venv\Scripts\python.exe scripts/pipeline/build_master_experiment_table.py
```
