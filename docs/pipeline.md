# Data Pipeline

This document maps shared acquisition and feature-engineering commands. Model
selection and inference APIs belong in `CURRENT_STATE.md` and
`ARCHITECTURE.md`.

## Commands

| Script | Role |
|---|---|
| `scripts/pipeline/fetch_stream.py` | Stream source molecules |
| `scripts/pipeline/clean.py` | Deduplicate, filter, and validate SMILES |
| `scripts/pipeline/features.py` | Build RDKit descriptors and Morgan fingerprints |
| `scripts/pipeline/feature_selection.py` | Run variance/correlation feature pruning |
| `scripts/pipeline/build_master_experiment_table.py` | Aggregate experiment evidence |
| `scripts/pipeline/build_progress_visualization.py` | Build progress plots from recorded results |

Phase-specific acquisition and acceptance commands are listed in the relevant
`scripts/phaseN/README.md` when one exists.

## Data Boundaries

- Raw downloaded tables: `data/raw/`.
- Commercial molecule inputs: `data/commercial/`.
- Regenerable graph and embedding caches: `data/cache/`.
- Immutable metrics and decisions: `results/`.

Identity normalization, PubChemQC filtering, graphs, and shared utilities are
owned by `src/molgap/`; use `ARCHITECTURE.md` to select the module to edit.
