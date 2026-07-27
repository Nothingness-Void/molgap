# Data Pipeline

This document maps shared acquisition and feature-engineering commands. Model
selection and inference APIs belong in `CURRENT_STATE.md` and
`ARCHITECTURE.md`.

## Commands

These are the acquisition-stage CLIs, all under `production/01_acquire/scripts/`.

| Script | Role |
|---|---|
| `fetch_stream.py` | Stream source molecules |
| `clean.py` | Deduplicate, filter, and validate SMILES |
| `features.py` | Build RDKit descriptors and Morgan fingerprints |
| `feature_selection.py` | Run variance/correlation feature pruning |
| `build_master_experiment_table.py` | Aggregate experiment evidence |
| `build_progress_visualization.py` | Build progress plots from recorded results |

Later stages keep their own CLIs under `production/<stage>/scripts/`; see
`production/README.md` for the stage order. Experiment-specific runners live with
their experiment under `experiments/`.

## Data Boundaries

- Raw downloaded tables: `data/raw/`.
- Commercial molecule inputs: `data/commercial/`.
- Regenerable graph and embedding caches: `data/cache/`.
- Immutable production evidence: the owning `production/<stage>/` directory.
- Immutable experiment evidence: the owning `experiments/<question>/` directory.

Identity normalization, PubChemQC filtering, graphs, and shared utilities are
owned by `src/molgap/`; use `ARCHITECTURE.md` to select the module to edit.
