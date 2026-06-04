# MolGap Handover

## Path update (2026-06-04 restructure)
Scripts reorganized by Phase:
- `scripts/pipeline/` — 通用管线 (fetch, clean, features, feature_selection)
- `scripts/phase1/` — Model Optimization (baseline, tuning, analysis, embeddings)
- `scripts/phase2/` — Generalization Study
- `scripts/phase3/` — Production Scale-Up
- `scripts/phase5/` — Commercial Prediction
- `scripts/colab/` — Colab notebooks
- `src/molgap/utils.py` — shared utilities
- Old `scripts/evaluation/`, `scripts/experiments/`, `scripts/todo/` directories removed.
- See `scripts/README.md` for full listing.

## Last updated
2026-06-04

## User intent
The user wants to continue developing the `molgap` project incrementally. They asked to create a local `TO-AI` folder containing workflow, TODO, handover, and related files so that context and progress are not lost after closing the assistant session.

## Project summary
`molgap` aims to build a machine-learning prediction database for commercially available organic electronic-material molecules. The target properties are:

- HOMO energy
- LUMO energy
- HOMO-LUMO gap

The data source is PubChemQC B3LYP/6-31G*//PM6 via HuggingFace. The current project starts with CHON-only molecules, molecular weight 200–300 g/mol, no salts.

## Important scientific/technical decisions already made

### Energies are treated as eV
Although `Project.md` originally mentioned checking whether values need Hartree-to-eV conversion, `scripts/pipeline/01_fetch_stream.py` self-check documents that the observed values are already in eV:

- HOMO/LUMO magnitudes are around -4 to -8.
- `gap == lumo - homo` holds.
- Do not multiply by 27.2114.

### Current raw data is sample-level only
The existing file:

```text
data/raw/pubchemqc_chon_mw200_300.csv
```

contains only about 280 filtered rows. These are already filtered by the current script, but they are based on partial Range reads from the beginning of large JSON files. This is not the final full filtered dataset.

### Recommended first modeling data size
- 280 rows: pipeline test only.
- 1k rows: preliminary code/model sanity check.
- 5k–10k rows: first useful baseline.
- 30k–100k rows: stronger report-quality dataset.

The recommended first formal modeling target is 10k+ filtered molecules.

### Modeling route
Start simple, then iterate:

1. Morgan fingerprint + RDKit descriptors.
2. Ridge / RandomForest / ExtraTrees / LightGBM baseline.
3. Error analysis and fixed split.
4. ChemBERTa / MolFormer SMILES embeddings.
5. Feature fusion.
6. Scaffold split.
7. Commercial molecule prediction.
8. Optional GNN / Uni-Mol / Gaussian validation.

## Reference project
The user provided a similar previous project:

```text
D:/文档/GitHub/Graduation-project
```

Useful files inspected:

```text
D:/文档/GitHub/Graduation-project/utils/data_utils.py
D:/文档/GitHub/Graduation-project/特征工程.py
D:/文档/GitHub/Graduation-project/DNN_模型验证.py
D:/文档/GitHub/Graduation-project/数据处理部分代码.py
```

Useful ideas from the reference project:

- Save and reuse train/test split indices.
- Use RDKit `Descriptors.CalcMolDescriptors()` for full 2D descriptors.
- Drop high-missing descriptor columns.
- Drop constant columns.
- Fill remaining NaN by median.
- Save prediction tables with true value, predicted value, residual, and absolute error.

Do not directly copy polymer-solvent interaction features into `molgap`, because `molgap` is a single-molecule property prediction project.

## Current files created in this handover step

```text
TO-AI/work-flow.md
TO-AI/todo.md
TO-AI/handover.md
TO-AI/notes.md
TO-AI/archive/stage1_pipeline_archive_20260602_144731.md
```

Recommended additional file to create next if desired:

```text
TO-AI/notes.md
```

## Recommended next implementation step
Stage 1 has been implemented and verified on the initial 281-row sample dataset. Stage 2 has also been completed with a 10k expanded dataset and a full baseline rerun. Stage 3 interpretability, Y-randomization, and confidence analysis has also been implemented and verified.

Implemented files:

```text
src/molgap/utils.py
scripts/pipeline/02_clean.py
scripts/pipeline/03_features.py
scripts/pipeline/04_train_baseline.py
scripts/evaluation/05_analyze_results.py
scripts/evaluation/06_y_randomization.py
scripts/evaluation/07_confidence_analysis.py
```

Stage 2 fetch command already run:

```bash
python scripts/pipeline/01_fetch_stream.py --run --max-records 10000 --chunk-bytes 100000000
```

Fetch result:

```text
records parsed: 15432
records kept: 10000
```

10k baseline result:

```text
clean rows: 10000 / 10000
feature rows: 10000
final feature columns: 2244
split: train=7999 valid=1001 test=1000
best validation model: lightgbm
LightGBM valid avg MAE: 0.1713, avg R2: 0.9069
LightGBM test avg MAE: 0.1670, avg R2: 0.9017
```

Stage 3 analysis results:

```text
HOMO: MAE=0.1441, RMSE=0.2099, R2=0.8709
LUMO: MAE=0.1588, RMSE=0.2334, R2=0.9346
Gap : MAE=0.1981, RMSE=0.3028, R2=0.8997
```

Y-randomization result:

```text
real avg MAE: 0.1670
real avg R2 : 0.9017
random avg MAE mean: 0.6607
random avg R2 mean : -0.0720
```

Confidence analysis result:

```text
High confidence gap MAE   : 0.1210
Medium confidence gap MAE : 0.2123
Low confidence gap MAE    : 0.2613
```

## 2026-06-03 update: 30k data + advanced models

Data expanded 10k → 30k. Tuned LightGBM on 30k:
- Random: avg MAE=0.1498, R²=0.9205 (was 0.1522/0.9117 on 10k)
- Scaffold: avg MAE=0.1851, R²=0.8799

Advanced model comparison (per-target LGBM, stacking, XGBoost, CatBoost, DART) was started but stacking/CatBoost are very slow on 30k. Check `results/advanced/` for completion.

Recommended next step is model-focused work only: benchmark/tune the current models, compare gap strategies, and decide whether to move next into embeddings or feature fusion. Commercial prediction is deferred in TODO.

Stage 4 scaffold split result:

```text
Scaffold split: train=7847 valid=1030 test=1123
LightGBM scaffold test avg MAE: 0.1999
LightGBM scaffold test avg R2 : 0.8703
Gap scaffold MAE: 0.2370
Gap scaffold R2 : 0.8027
```

Random vs scaffold comparison:

```text
Random split LightGBM avg R2  : 0.9017
Scaffold split LightGBM avg R2: 0.8703
```

Deferred commercial pipeline:

```text
scripts/todo/09_predict_commercial.py
data/commercial/commercial_molecules_template.csv
results/database/commercial_molgap_predictions_v1.csv
```

The template prediction smoke test succeeded with 3/3 molecules predicted.

Suggested next files/actions:

```text
data/commercial/commercial_molecules.csv
results/database/commercial_molgap_predictions_v1.csv
```

When model development is stable again, fill `commercial_molecules.csv` with verified commercial molecules from TCI/Sigma/Ossila/Lumtec/etc., then run:

```bash
python scripts/todo/09_predict_commercial.py --input data/commercial/commercial_molecules.csv --output results/database/commercial_molgap_predictions_v1.csv
```

## Suggested output conventions

Clean data:

```text
data/processed/pubchemqc_chon_mw200_300_clean.csv
```

Feature table:

```text
data/processed/features_morgan2048_desc.csv
```

Split indices:

```text
results/train_valid_test_split_indices.npz
```

Models:

```text
models/baseline_ridge.joblib
models/baseline_extratrees.joblib
models/baseline_randomforest.joblib
```

Metrics/predictions:

```text
results/model_comparison_baseline.csv
results/metrics_ridge.json
results/metrics_extratrees.json
results/test_predictions_extratrees.csv
```

## Potential issue to remember
The project directory `D:/文档/GitHub/molgap` was reported as not being a git repository by the Claude Code environment. Avoid assuming git commands are available for this project unless initialized later.

## Installed Claude Code plugin note
The user installed the marketplace:

```text
forrestchang/andrej-karpathy-skills
```

The assistant then successfully ran:

```bash
claude plugin install andrej-karpathy-skills@karpathy-skills
```

A later plugin list command was interrupted by the user, so installation was not re-verified by listing, but the install command reported success.

## 2026-06-04 update: Phase 3.4 Optimization Complete

Feature selection (6028→2811) + Optuna tuning on CHONSFCl 30k:
- Best: Tuned LightGBM avg MAE=0.1596, R²=0.8853
- Improvement over Phase 3 baseline: MAE -6.4%, R² +0.01
- R²=0.9 target not reached (gap=0.015)
- Results in `results/phase3/optimize/`
- Archive: `TO-AI/archive/stage9_phase3_optimization_20260604.md`

