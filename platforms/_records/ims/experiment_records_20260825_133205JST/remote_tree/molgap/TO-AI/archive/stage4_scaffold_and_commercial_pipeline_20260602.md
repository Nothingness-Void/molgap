# Stage 4 Scaffold Split and Commercial Prediction Pipeline Archive

Archived at: 2026-06-02.

## Implemented files

```text
src/08_scaffold_split_train.py
src/09_predict_commercial.py
data/commercial/commercial_molecules_template.csv
```

Updated reusable feature logic:

```text
src/utils.py
src/03_features.py
```

## 1. Scaffold split validation

Command:

```bash
python src/08_scaffold_split_train.py --skip-extratrees
```

Resulting split:

```text
train=7847
valid=1030
test=1123
```

Outputs:

```text
results/scaffold/scaffold_split_summary.csv
results/scaffold/scaffold_split_indices.npz
results/scaffold/model_comparison_scaffold.csv
results/scaffold/random_vs_scaffold_comparison.csv
results/scaffold/test_predictions_ridge_scaffold.csv
results/scaffold/test_predictions_lightgbm_scaffold.csv
results/scaffold/metrics_ridge_scaffold_test.json
results/scaffold/metrics_lightgbm_scaffold_test.json
models/scaffold_ridge.joblib
models/scaffold_lightgbm.joblib
```

Scaffold split metrics:

```text
Ridge test:
  average MAE = 0.2703
  average R2  = 0.7675

LightGBM test:
  average MAE = 0.1999
  average R2  = 0.8703
```

LightGBM random vs scaffold comparison:

```text
Random split:
  average MAE = 0.1670
  average R2  = 0.9017

Scaffold split:
  average MAE = 0.1999
  average R2  = 0.8703
```

Per-target scaffold LightGBM:

```text
HOMO: MAE=0.1694, R2=0.8749
LUMO: MAE=0.1932, R2=0.9333
Gap : MAE=0.2370, R2=0.8027
```

Interpretation:

- Scaffold split is harder than random split, as expected.
- LightGBM still performs reasonably well on unseen scaffolds.
- Gap generalization is the weakest target and should be prioritized in later optimization.

## 2. Commercial molecule prediction pipeline

Added template:

```text
data/commercial/commercial_molecules_template.csv
```

Template columns:

```text
name,supplier,catalog_id,cid,smiles,formula,mw,category,application,reference_url,notes
```

Prediction command tested:

```bash
python src/09_predict_commercial.py --input data/commercial/commercial_molecules_template.csv --output results/database/commercial_molgap_predictions_v1.csv
```

Outputs:

```text
results/database/commercial_molgap_predictions_v1.csv
results/database/commercial_prediction_summary.json
```

Smoke-test result:

```text
n_input = 3
n_predicted = 3
n_invalid_or_failed = 0
reference_model = lightgbm
prediction_source = ML_from_PubChemQC
model_version = lightgbm_morgan_rdkit_v1
```

Output database fields include:

```text
name
supplier
catalog_id
cid
smiles
canonical_smiles
formula
mw
category
application
homo_pred
lumo_pred
gap_pred
confidence_bin
uncertainty_score
model_disagreement
applicability_distance
prediction_status
prediction_source
model_version
reference_url
notes
```

Important caveat:

The template rows are examples for pipeline smoke testing only. They must be replaced by a curated commercial molecule list with verified supplier/catalog/SMILES information before use in a real database.

## Recommended next step

Create a real curated commercial molecule CSV:

```text
data/commercial/commercial_molecules.csv
```

Start with 50–100 verified molecules from sources such as:

```text
TCI
Sigma-Aldrich / Merck
Ossila
Lumtec
Alfa Aesar
PubChem
```

Then run:

```bash
python src/09_predict_commercial.py --input data/commercial/commercial_molecules.csv --output results/database/commercial_molgap_predictions_v1.csv
```

After that, inspect confidence bins and out-of-domain molecules before expanding the commercial list.
