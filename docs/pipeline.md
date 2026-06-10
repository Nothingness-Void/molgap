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

## Shared modules (`src/molgap/`)

| Module | Contents |
|--------|----------|
| `constants.py` | All path constants (`MODEL_PHASE6`, `GRAPHS_PHASE6`, etc.), model hyperparameters, `TARGET_COLS` |
| `graphs.py` | `smiles_to_pyg()`, `smiles_to_pyg_ensemble()`, `smiles_list_to_pyg()`, `build_labeled_graphs()` |
| `inference.py` | `load_model()`, `predict_smiles()`, `predict_smiles_batch()`, `predict_smiles_ensemble()` |
| `schnet.py` | `SchNetWrapper` (SchNet + optional charges + optional 2D descriptors + multi-target head) |
| `utils.py` | Splits, metrics, SMILES canonicalization, RDKit descriptors, Gasteiger charges |

Package is installed editable (`pip install -e .`) so `import molgap` works everywhere without sys.path hacks.
