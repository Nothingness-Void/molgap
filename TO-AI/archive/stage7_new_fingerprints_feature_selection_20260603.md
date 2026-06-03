# Stage 7 New Fingerprints + Lightweight Feature Selection Archive

Archived at: 2026-06-03.

## What was done

1. **Added 3 new fingerprint types** to `src/molgap/utils.py`:
   - MACCS keys (166 bits) — encodes specific functional group/substructure patterns
   - Atom Pair fingerprint (2048 bits, hashed) — encodes atom type pairs + topological distance
   - Topological Torsion fingerprint (2048 bits, hashed) — encodes 4-atom path torsion patterns

2. **Parallelized feature generation** in `src/molgap/utils.py`:
   - Added `build_feature_rows_parallel()` using `multiprocessing.Pool` with `imap` + `chunksize=256`
   - Workers = CPU cores - 1 (15 on current machine)
   - Updated `scripts/pipeline/03_features.py` to use parallel version

3. **Added lightweight gain-based feature selection** script:
   - `scripts/pipeline/03b_feature_selection.py`
   - Trains quick LightGBM, drops features with total gain = 0
   - Dimensionality: 6527 → 5451 (drop constant) → **2324** (drop zero-gain)

4. **Retrained all baseline models** with selected features

## Modified files

```text
src/molgap/utils.py                          — added calc_maccs_keys(), calc_atompair_bits(), calc_torsion_bits(), build_feature_rows_parallel()
scripts/pipeline/03_features.py              — switched to parallel feature generation
scripts/pipeline/03b_feature_selection.py     — NEW: lightweight feature selection
```

## Run commands

```bash
# Step 1: Generate features (all fingerprints)
python scripts/pipeline/03_features.py

# Step 2: Feature selection (gain-based)
python scripts/pipeline/03b_feature_selection.py

# Step 3: Train with selected features
python scripts/pipeline/04_train_baseline.py --input data/processed/features_selected.csv
```

## Outputs

```text
data/processed/features_morgan2048_desc.csv          — full features (5451 dims after drop constant)
data/processed/features_selected.csv                 — selected features (2324 dims)
results/feature_selection_gain.csv                   — per-feature gain ranking
results/comparison_report_new_fingerprints.csv        — old vs new comparison table
models/baseline_*.joblib                             — retrained models
results/metrics_*_valid.json, metrics_*_test.json    — updated metrics
results/model_comparison_baseline.csv                — updated comparison
```

## Feature selection result

| Fingerprint type | Original dims | Kept after selection | Retention rate |
|-----------------|---------------|---------------------|----------------|
| Morgan ECFP4    | 2048          | 761                 | 37%            |
| MACCS keys      | 166           | 110                 | 66%            |
| Atom Pair       | 2048          | 778                 | 38%            |
| Torsion         | 2048          | 497                 | 24%            |
| RDKit desc      | 217           | 178                 | 82%            |
| **Total**       | **6527**      | **2324**            | **36%**        |

MACCS has highest retention (66%) — most of its 166 bits carry real information.

## Key result: LightGBM test set comparison (old → new)

| Target  | Old MAE | New MAE | Change  | Old R²  | New R²  | Change |
|---------|---------|---------|---------|---------|---------|--------|
| HOMO    | 0.1441  | 0.1372  | -4.8%   | 0.8709  | 0.8832  | +1.4%  |
| LUMO    | 0.1588  | 0.1512  | -4.8%   | 0.9346  | 0.9419  | +0.8%  |
| Gap     | 0.1981  | 0.1876  | -5.3%   | 0.8997  | 0.9045  | +0.5%  |
| **Avg** | **0.1670** | **0.1587** | **-5.0%** | **0.9017** | **0.9099** | **+0.9%** |

All 4 models improved across the board. Ridge benefited most (MAE -6.2%, R² +1.8%).

## Physicochemical interpretation of top features

Top gain contributors remain RDKit descriptors (~90%):
1. **BCUT2D_MRHI** — Burden matrix eigenvalue (molar refractivity), encodes polarizability distribution
2. **VSA_EState4/2** — van der Waals surface area by electrotopological state
3. **HallKierAlpha** — atom size / hybridization correction, reflects orbital polarizability
4. **BCUT2D_MRLOW** — minimum polarizability eigenvalue
5. **MinPartialCharge** — Gasteiger partial charge extremum

These features describe **atomic polarizability distribution**, **conjugation degree**, and **local charge** — directly relevant to HOMO/LUMO energy levels.

## Hardware used

- CPU: AMD Ryzen 7 6800H (8 cores / 16 threads)
- RAM: 15.2 GB
- Feature generation: ~2 min (parallel), Training: ~9 min
