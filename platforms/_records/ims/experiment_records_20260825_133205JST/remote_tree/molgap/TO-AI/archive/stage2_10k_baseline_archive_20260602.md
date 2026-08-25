# Stage 2 10k Dataset Baseline Archive

Archived at: 2026-06-02 after expanded PubChemQC extraction and full pipeline rerun.

## Expanded data fetch

Command:

```bash
python src/01_fetch_stream.py --run --max-records 10000 --chunk-bytes 100000000
```

Result:

```text
found 87 files
records parsed : 15432
records kept   : 10000 (MW 200-300, CHON only)
elapsed        : 18.3s
output         : data/raw/pubchemqc_chon_mw200_300.csv
```

The 10k target was reached during the third PubChemQC JSON file, so the remaining files were not scanned.

## Pipeline rerun

Command:

```bash
python src/02_clean.py && python src/03_features.py && python src/04_train_baseline.py
```

## Cleaning result

```text
raw rows       : 10000
clean rows     : 10000
removed total  : 0
missing values removed: 0
non-positive gap rows removed: 0
inconsistent gap rows removed: 0
invalid SMILES removed: 0
duplicate canonical SMILES removed: 0
```

Output:

```text
data/processed/pubchemqc_chon_mw200_300_clean.csv
```

## Feature result

```text
input rows             : 10000
feature rows           : 10000
failed molecules       : 0
Morgan bits requested  : 2048
RDKit descriptors raw  : 217
constant feature columns dropped: 21
final feature columns  : 2244
NaN filled             : 0 -> 0
```

Output:

```text
data/processed/features_morgan2048_desc.csv
```

## Baseline training result

Split:

```text
train=7999
valid=1001
test=1000
```

Models trained:

```text
ridge
extratrees
randomforest
lightgbm
```

Validation/test summary:

```text
ridge:
  valid avg MAE=0.2394 avg R2=0.8412
  test  avg MAE=0.2313 avg R2=0.8393

extratrees:
  valid avg MAE=0.1794 avg R2=0.8736
  test  avg MAE=0.1747 avg R2=0.8665

randomforest:
  valid avg MAE=0.2087 avg R2=0.8420
  test  avg MAE=0.1982 avg R2=0.8466

lightgbm:
  valid avg MAE=0.1713 avg R2=0.9069
  test  avg MAE=0.1670 avg R2=0.9017
```

Best validation model:

```text
lightgbm
```

Main outputs:

```text
models/baseline_ridge.joblib
models/baseline_extratrees.joblib
models/baseline_randomforest.joblib
models/baseline_lightgbm.joblib
results/train_valid_test_split_indices.npz
results/model_comparison_baseline.csv
results/test_predictions_ridge.csv
results/test_predictions_extratrees.csv
results/test_predictions_randomforest.csv
results/test_predictions_lightgbm.csv
results/metrics_*_valid.json
results/metrics_*_test.json
```

## Interpretation

The 10k baseline is the first useful modeling result. LightGBM is currently the strongest model on the fixed random split. These results are still from a random split and should later be compared against scaffold split to assess generalization to new molecular scaffolds.

## Recommended next step

Stage 3 should add result visualization and error analysis:

- parity plots for HOMO/LUMO/gap
- residual histograms
- top-error molecule tables
- per-target metric summary table
- optional cleanup of LightGBM warning by consistently using DataFrames at train/predict time

After that, increase data to 30k+ or start ChemBERTa/MolFormer embedding comparison.
