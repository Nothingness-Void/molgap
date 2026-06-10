# Phase 7: Model Improvement Experiments

## Goal
Break past OOD R²≈0.8 ceiling identified in Phase 6.

## Sub-experiments

### 7A: Conformer Ensemble (DONE)
Generate K ETKDG conformers per molecule, predict each, average.

| k | avg R² | avg MAE |
|---|--------|---------|
| 1 | 0.791 | 0.234 |
| 3 | 0.808 | 0.225 |
| 5 | 0.809 | 0.224 |
| 8 | 0.816 | 0.221 |

Per-molecule conformer std: HOMO 0.075, LUMO 0.085, Gap 0.104 eV.

**Conclusion**: +2.5% R², marginal improvement. Conformer noise quantified but not the primary bottleneck.

Script: `scripts/phase7/conformer_ensemble.py`
Results: `results/phase7/conformer_ensemble/`

### 7B: Hybrid 2D+3D SchNet (RUNNING)
Fuse RDKit 2D descriptors into SchNet via `desc_proj` layer in SchNetWrapper.

- Running on Kaggle: `scripts/phase7/kaggle_hybrid_2d3d.ipynb`
- Local prep: `scripts/phase7/inject_desc_to_graphs.py`, `scripts/phase7/hybrid_2d3d_experiment.py`
- Graph cache with descriptors: `results/phase7/pyg_3d_graphs_etkdg_expanded_with_desc.pt`
- Normalization stats: `results/phase7/desc_normalization.json`

**Status**: Awaiting Kaggle results. Compare with Phase 6 baseline (R²=0.882).

### 7C: xTB Conformer (SUSPENDED)
Replace ETKDG with GFN2-xTB optimized conformers. Suspended — conformer ensemble (7A) showed limited improvement, so finer conformers unlikely to help much. Lowest priority.

### 7D: 300k Data Scaling (TODO — HIGHEST PRIORITY)
Scale training data 44.8k → 300k via additional PubChemQC fetch.

- Script: `scripts/phase7/fetch_300k.py`
- Kaggle notebook: `scripts/phase7/kaggle_optuna_300k.ipynb`
- Expected: OOD R² 0.80 → 0.88-0.92
- Estimated effort: 2-3 days on Kaggle

## Dependencies
- Phase 6 model + data as baseline
- `src/molgap/schnet.py` (SchNetWrapper with n_desc support for 7B)
