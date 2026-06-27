# Pipeline: Data Acquisition & Feature Engineering

## Overview
Shared data pipeline used by all phases. Fetches from PubChemQC via PUG REST API, cleans, and generates features.

## Scripts
| Script | Purpose |
|--------|---------|
| `scripts/pipeline/fetch_stream.py` | Stream molecules from PubChemQC (ijson) |
| `scripts/pipeline/clean.py` | Deduplicate, filter by elements/MW, validate SMILES |
| `scripts/pipeline/features.py` | RDKit descriptors + Morgan fingerprints |
| `scripts/pipeline/feature_selection.py` | Variance/correlation-based feature pruning |
| `scripts/pipeline/build_master_experiment_table.py` | Aggregate experiment results |
| `scripts/pipeline/build_progress_visualization.py` | Progress tracking plots |

## Data files
| File | Phase | Scope |
|------|-------|-------|
| `data/raw/phase3_chonsfcl_mw200_500_30k.csv` | 3,4,6 | 30k, MW200-500 |
| `data/raw/phase6_chonsfcl_mw500_1000_1k.csv` | 6 | 1k test fetch |
| `data/raw/phase6_chonsfcl_mw500_1000_15k.csv` | 6 | 15k, MW500-1000 |
| `data/raw/phase7_chonsfcl_mw200_1000_300k.csv` | 7 | v1 300k control |
| `data/raw/phase8_replacement_300k.csv` | 8 | v2 fixed-size replacement 300k |

## Shared modules (`src/molgap/`)

| Module | Contents |
|--------|----------|
| `constants.py` | All path constants, model registry, model hyperparameters, `TARGET_COLS` |
| `graphs.py` | `smiles_to_pyg()`, `smiles_to_pyg_ensemble()`, `smiles_list_to_pyg()`, `build_labeled_graphs()` |
| `inference.py` | `load_hybrid()`, `predict_smiles_batch_hybrid()`, legacy 3D-only helpers, v1 UQ wrapper |
| `schnet.py` | `SchNetWrapper` (SchNet + optional charges + optional 2D descriptors + multi-target head) |
| `utils.py` | Splits, metrics, SMILES canonicalization, RDKit descriptors, Gasteiger charges |

Package is installed editable (`pip install -e .`) so `import molgap` works everywhere without sys.path hacks.
For the current recommended B3LYP model use `load_hybrid()`; it defaults to
`phase8_replacement_hybrid`.
