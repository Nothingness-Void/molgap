# MolGap TODO

## Path update
- Production pipeline scripts now live under `scripts/pipeline/`.
- Evaluation and validation scripts now live under `scripts/evaluation/`.
- Experimental benchmark scripts now live under `scripts/experiments/`.
- Deferred application scripts now live under `scripts/todo/`.
- Shared utilities moved from `src/utils.py` to `src/molgap/utils.py`.
- Historical archive notes may still mention the pre-refactor `src/*.py` paths.

## Active priority

### P0 — Focus on model development only
- [x] Create `TO-AI/` handover folder.
- [x] Write workflow documentation.
- [x] Write handover summary.
- [x] Write current TODO list.
- [x] Implement `src/molgap/utils.py`.
- [x] Implement `scripts/pipeline/02_clean.py`.
- [x] Implement `scripts/pipeline/03_features.py`.
- [x] Implement `scripts/pipeline/04_train_baseline.py`.
- [x] Verify small-sample pipeline end to end.
- [x] Tune LightGBM / ExtraTrees on the 10k dataset.
- [x] Run the lightweight benchmark and confirm the best model/feature set.
- [x] Decide the next model branch: embeddings or feature fusion.

## Stage 1 — Small-sample pipeline

### `src/molgap/utils.py`
- [x] Add RDKit-safe SMILES parsing.
- [x] Add canonical SMILES conversion.
- [x] Add split-index save/load helpers inspired by `Graduation-project/utils/data_utils.py`.
- [x] Add regression metric helper for multi-target outputs.

### `scripts/pipeline/02_clean.py`
Input:

```text
data/raw/pubchemqc_chon_mw200_300.csv
```

Tasks:

- [x] Check required columns: `cid,mw,formula,smiles,homo,lumo,gap`.
- [x] Convert numeric columns to numeric dtype.
- [x] Drop rows with missing target values.
- [x] RDKit-parse SMILES.
- [x] Create `canonical_smiles`.
- [x] Drop invalid SMILES.
- [x] Drop duplicate `canonical_smiles`.
- [x] Filter `gap <= 0`.
- [x] Check `abs((lumo - homo) - gap)` and remove inconsistent rows.
- [x] Save cleaned CSV.
- [x] Print cleaning statistics.

Output:

```text
data/processed/pubchemqc_chon_mw200_300_clean.csv
```

### `scripts/pipeline/03_features.py`
Input:

```text
data/processed/pubchemqc_chon_mw200_300_clean.csv
```

Tasks:

- [x] Compute Morgan fingerprint, radius=2, 2048 bits.
- [x] Compute RDKit 2D descriptors via `Descriptors.CalcMolDescriptors()`.
- [x] Replace inf with NaN.
- [x] Drop high-missing descriptor columns.
- [x] Drop constant columns.
- [x] Fill remaining NaN by median.
- [x] Preserve metadata columns: `cid,mw,formula,smiles,canonical_smiles,homo,lumo,gap`.
- [x] Save feature CSV.

Output:

```text
data/processed/features_morgan2048_desc.csv
```

### `scripts/pipeline/04_train_baseline.py`
Input:

```text
data/processed/features_morgan2048_desc.csv
```

Tasks:

- [x] Split train/valid/test with fixed random seed.
- [x] Save split indices to `results/train_valid_test_split_indices.npz`.
- [x] Train Ridge baseline.
- [x] Train ExtraTrees baseline.
- [x] Train RandomForest baseline.
- [x] Train LightGBM if installed.
- [x] Evaluate HOMO/LUMO/gap separately.
- [x] Save metrics JSON/CSV.
- [x] Save test prediction CSV.
- [x] Save models to `models/`.

Outputs:

```text
models/baseline_*.joblib
results/metrics_*.json
results/model_comparison_baseline.csv
results/test_predictions_*.csv
```

## Stage 1 verification result
Completed on the current 281-row sample dataset.

Clean result:

```text
raw rows: 281
clean rows: 281
invalid SMILES: 0
duplicates: 0
bad gap consistency rows: 0
```

Feature result:

```text
feature rows: 281
Morgan bits requested: 2048
RDKit descriptors raw: 217
constant feature columns dropped: 631
final feature columns: 1634
```

Baseline result, sample only:

```text
train=223 valid=29 test=29
best validation model: extratrees
ExtraTrees valid avg MAE: 0.3236
ExtraTrees test avg MAE: 0.2663
```

Important: these metrics are only for pipeline verification because the dataset is too small.

## Stage 2 — Dataset expansion
- [x] Rename or document current raw CSV as sample data if needed.
- [x] Run larger Range extraction targeting 10k filtered rows.
- [x] Re-run clean/features/train pipeline.
- [x] Confirm whether 10k data is enough for initial report-level baseline.

Completed command:

```bash
python scripts/pipeline/01_fetch_stream.py --run --max-records 10000 --chunk-bytes 100000000
```

Stage 2 result:

```text
records parsed: 15432
records kept: 10000
clean rows: 10000
feature columns: 2244
best validation model: lightgbm
LightGBM valid avg MAE: 0.1713, avg R2: 0.9069
LightGBM test avg MAE: 0.1670, avg R2: 0.9017
```

## Stage 3 — Stronger baseline
- [x] Add parity plots.
- [x] Add residual plots.
- [x] Add top-error molecule report.
- [x] Add LightGBM feature-importance analysis.
- [x] Add Y-randomization validation.
- [x] Add confidence/uncertainty proxy analysis.
- [x] Tune ExtraTrees / LightGBM after data reaches 10k+.

Stage 3 outputs:

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
results/y_randomization/y_randomization_summary.csv
results/y_randomization/y_randomization_summary.json
results/confidence/confidence_predictions.csv
results/confidence/confidence_summary.csv
results/confidence/error_by_confidence_bin.csv
```

Stage 3 key result:

```text
Y-randomized avg R2 mean: -0.0720 vs real avg R2: 0.9017
High confidence gap MAE: 0.1210
Medium confidence gap MAE: 0.2123
Low confidence gap MAE: 0.2613
```

## Stage 4 — Scaffold split generalization
- [x] Add `scripts/pipeline/08_scaffold_split_train.py`.
- [x] Generate Bemis-Murcko scaffolds.
- [x] Split by scaffold group into train/valid/test.
- [x] Train LightGBM and baseline models on scaffold split.
- [x] Compare random split vs scaffold split.

Stage 4 scaffold result:

```text
Scaffold split train=7847 valid=1030 test=1123
LightGBM scaffold avg MAE=0.1999
LightGBM scaffold avg R2=0.8703
Random split LightGBM avg R2=0.9017
```

## Stage 5 — Model optimization
- [x] Run `scripts/experiments/10_light_benchmark.py`.
- [x] Re-check whether the current best feature set is still Morgan + RDKit + LightGBM.
- [x] Review `scripts/evaluation/11_gap_consistency_analysis.py` and decide whether blended gap should become the reported gap metric.
- [x] Tune LightGBM hyperparameters with Optuna (`scripts/experiments/13_tune_lightgbm.py`).
- [x] Decide whether scaffold performance is strong enough before starting embeddings.

Stage 5 conclusions:
- Best feature set: Morgan + RDKit descriptors + LightGBM (confirmed).
- Blend gap (α=0.3–0.6) improves gap slightly; report both direct and blend.
- Tuned LightGBM: random avg MAE=0.1522 R²=0.9117; scaffold avg MAE=0.1899 R²=0.8815.
- Scaffold R²=0.88 is strong enough to proceed; embedding experiments are optional enhancement.

## Stage 6 — Embedding experiments
- [x] Generate ChemBERTa embeddings (Colab, `scripts/colab/molgap_embeddings.ipynb`).
- [x] Generate MolFormer embeddings (Colab).
- [x] Add `scripts/experiments/14_train_with_embeddings.py`.
- [x] Compare embedding-only vs traditional features.
- [x] Try feature fusion.

Stage 6 conclusion:
- Traditional features (Morgan + RDKit) alone are the best (avg MAE=0.1587).
- Embedding-only models are much worse (MAE 0.24-0.29).
- Fusion does not improve over traditional features.
- Final model decision: tuned LightGBM + Morgan + RDKit descriptors.

## Stage 7 — Commercial database construction (Deferred TODO)
- [x] Add `data/commercial/commercial_molecules_template.csv`.
- [x] Add `scripts/todo/09_predict_commercial.py`.
- [x] Verify commercial prediction pipeline on template rows.
- [ ] Curate real `data/commercial/commercial_molecules.csv`.
- [ ] Predict real commercial molecule database.
- [ ] Inspect low-confidence / out-of-domain commercial molecules.
- [ ] Prepare final database table and documentation.

## Stage 8 — Model completion and reporting
- [x] Add scaffold split.
- [x] Compare random split vs scaffold split.
- [ ] Consolidate the final model comparison table for random vs scaffold evaluation.
- [ ] Decide which gap strategy should be reported as the main result.
- [ ] Prepare the final model-side summary for report writing.

## Notes
- Current raw CSV has only 281 filtered rows and is not enough for final modeling.
- Use it only to verify pipeline correctness.
- First formal modeling target: 10k+ filtered rows.
- Preferred first representation: Morgan fingerprint + RDKit 2D descriptors.
- Preferred first models: Ridge, ExtraTrees, RandomForest, then LightGBM.
- Embedding route should start with ChemBERTa and MolFormer.

