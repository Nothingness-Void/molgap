# MolGap Workflow

## Path update
- Production pipeline scripts now live under `scripts/pipeline/`.
- Evaluation and validation scripts now live under `scripts/evaluation/`.
- Experimental benchmark scripts now live under `scripts/experiments/`.
- Deferred application scripts now live under `scripts/todo/`.
- Shared utilities moved from `src/utils.py` to `src/molgap/utils.py`.
- Historical archive notes may still mention the pre-refactor `src/*.py` paths.

## Project goal
Build a machine-learning database for commercially available organic electronic-material molecules. The model predicts three quantum-chemical properties from molecular structure:

- HOMO energy
- LUMO energy
- HOMO-LUMO gap

The target application is OLED / organic thin-film / organic solar-cell related small molecules and building blocks.

## Key constraints
- Main data source: `molssiai-hub/pubchemqc-b3lyp` on HuggingFace.
- Use the internal subset: `data/b3lyp_pm6_chon300nosalt/train/*.json`.
- Do not download the full PubChemQC dataset.
- Current molecule scope: molecular weight 200–300 g/mol, CHON-only, no salts.
- PubChemQC values are calculated values, not experimental values.
- Current script self-check indicates `energy-alpha-homo`, `energy-alpha-lumo`, and `energy-alpha-gap` are already in eV and satisfy `gap = lumo - homo`; do not multiply by 27.2114.

## Current repository state
Current project root:

```text
D:/文档/GitHub/molgap
```

Important files:

```text
Project.md
requirements.txt
scripts/pipeline/01_fetch_stream.py
data/raw/pubchemqc_chon_mw200_300.csv
data/processed/.gitkeep
data/commercial/.gitkeep
models/.gitkeep
results/.gitkeep
```

`scripts/pipeline/01_fetch_stream.py` currently supports:

- `--selfcheck` for unit/field sanity check.
- `--run` for streaming sample extraction.
- HTTP Range requests.
- `ijson` incremental parsing.
- CHON + MW 200–300 filtering.
- Slim CSV output with fields: `cid,mw,formula,smiles,homo,lumo,gap`.

Important limitation:

The current `run_stream()` only reads the first `chunk_bytes` of each large JSON file. Therefore the current raw CSV is a sample/smoke-test dataset, not the full filtered PubChemQC subset.

## Recommended iterative development route

### Stage 0 — Project foundation
Add shared utilities and keep the pipeline reproducible.

Recommended file:

```text
src/molgap/utils.py
```

Functions to include:

- `safe_mol(smiles)`
- `canonicalize_smiles(smiles)`
- `load_or_create_split_indices(...)`
- `save_split_indices(...)`
- `calculate_regression_metrics(...)`
- JSON/CSV save helpers

Reference project:

```text
D:/文档/GitHub/Graduation-project
```

Useful reference file:

```text
D:/文档/GitHub/Graduation-project/utils/data_utils.py
```

It contains `load_saved_split_indices(...)`, useful for fixed train/test splits.

### Stage 1 — Small-sample pipeline
Use the current ~280-row raw CSV only to verify the full code path, not for final scientific conclusions.

Implement:

```text
scripts/pipeline/02_clean.py
scripts/pipeline/03_features.py
scripts/pipeline/04_train_baseline.py
```

Expected flow:

```text
data/raw/pubchemqc_chon_mw200_300.csv
  -> data/processed/pubchemqc_chon_mw200_300_clean.csv
  -> data/processed/features_morgan2048_desc.csv
  -> models/baseline_*.joblib + results/metrics_*.json
```

### Stage 2 — Expand dataset to 10k+
First target: at least 10,000 filtered rows.

Suggested command style after pipeline is ready:

```bash
python scripts/pipeline/01_fetch_stream.py --run --max-records 10000 --chunk-bytes 100000000
```

If needed, try larger chunks: 200 MB or 300 MB per file.

Avoid implementing full 349 GB traversal until the downstream pipeline is stable.

### Stage 3 — Traditional baseline
Features:

- Morgan fingerprint / ECFP4, radius=2, 2048 bits.
- RDKit 2D descriptors.

Models:

- Ridge
- RandomForestRegressor
- ExtraTreesRegressor
- LightGBM if available

Targets:

- `homo`
- `lumo`
- `gap`

Metrics for each target:

- MAE
- RMSE
- R2

### Stage 4 — Model optimization and error analysis
Add:

- Fixed train/valid/test split saved to `results/train_valid_test_split_indices.npz`.
- Parity plots for HOMO/LUMO/gap.
- Residual histograms.
- `results/test_predictions_*.csv` with true, predicted, residual, absolute error.
- Top-error molecule analysis.

### Stage 5 — SMILES embedding comparison
Only after a stable traditional baseline and preferably 10k+ data.

Recommended first embedding models:

- ChemBERTa
- MolFormer

Optional later:

- Mol2Vec
- MolBERT
- GROVER
- Uni-Mol

Recommended files:

```text
src/06_embed_smiles.py
src/07_train_embeddings.py
```

Recommended outputs:

```text
data/processed/embeddings/chemberta.npy
data/processed/embeddings/chemberta_metadata.csv
data/processed/embeddings/molformer.npy
data/processed/embeddings/molformer_metadata.csv
results/model_comparison_embeddings.csv
```

Use attention-mask mean pooling for transformer embeddings.

### Stage 6 — Feature fusion
Compare:

- Morgan + RDKit
- ChemBERTa only
- MolFormer only
- Morgan + RDKit + ChemBERTa
- Morgan + RDKit + MolFormer

Recommended models:

- ExtraTrees
- LightGBM

### Stage 7 — Scaffold split
Use RDKit Bemis-Murcko scaffold split to test generalization to unseen scaffolds.

Compare:

```text
random split vs scaffold split
```

### Stage 8 — Commercial molecule prediction database
Create:

```text
data/commercial/commercial_molecules.csv
```

Suggested columns:

```text
name,supplier,catalog_id,cid,smiles,formula,mw,category,note
```

Prediction output:

```text
results/commercial_predictions.csv
```

Current status:
- Deferred for now.
- Do not treat this as the active next step.
- Resume only after the model-focused stages are stable.

### Stage 9 — Advanced models and Gaussian validation
Only after the baseline and embedding comparisons are stable.

Possible additions:

- GNN / MPNN / GIN
- Uni-Mol with generated 3D conformers
- Gaussian B3LYP/6-31G(d) validation for selected commercial candidates

## Current recommended next action
Do not move to commercial prediction yet. Keep the current focus on the model side:

```text
1. tune current baseline models
2. run lightweight benchmark
3. review gap-consistency strategy
4. decide between embeddings and feature fusion
```

If you need to rerun the core model pipeline, use:

```text
1. src/molgap/utils.py
2. scripts/pipeline/02_clean.py
3. scripts/pipeline/03_features.py
4. scripts/pipeline/04_train_baseline.py
```

Then run:

```bash
python scripts/pipeline/02_clean.py
python scripts/pipeline/03_features.py
python scripts/pipeline/04_train_baseline.py
```

Useful next model commands:

```bash
python scripts/pipeline/08_scaffold_split_train.py
python scripts/experiments/10_light_benchmark.py
python scripts/evaluation/11_gap_consistency_analysis.py
python scripts/evaluation/12_feature_contribution_analysis.py
```

The current ~280 rows are only for pipeline verification. Once the code path works, expand to 10k+ filtered molecules and re-run the same pipeline.

