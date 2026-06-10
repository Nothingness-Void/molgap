# MolGap — OLED Molecular Property Prediction

## What this project does
Train SchNet (3D GNN) on PubChemQC B3LYP/6-31G* data, then batch-predict HOMO/LUMO/Gap (eV) for commercial organic electronic molecules (OLED, thin-film, OPV) to build a property database.

## Current state
- **Best model**: Phase 6 SchNet ETKDG Optuna — R²=0.882, OOD R²=0.797, Gaussian Gap MAE=0.223 eV
- **Training data**: 44.8k molecules, CHONSFCl, MW 200-1000
- **Model file**: `models/gnn_schnet_3d_optuna_expanded.pt`
- **In progress**: Phase 7 Hybrid 2D+3D (Kaggle)
- **Next**: 300k data scaling → build commercial molecule prediction database
- **Priorities & checklist**: `ROADMAP.md`

## Phase docs (read only what you need)
| Doc | When to read |
|-----|-------------|
| `docs/pipeline.md` | Working on data fetch, cleaning, features, or shared modules |
| `docs/phase1.md` | Traditional ML baseline (LightGBM) |
| `docs/phase2.md` | Generalization / element-MW expansion study |
| `docs/phase3.md` | 30k data scaling + ML optimization |
| `docs/phase4.md` | SchNet GNN + ETKDG consistency (**critical lesson here**) |
| `docs/phase5.md` | OOD / Gaussian / experimental validation |
| `docs/phase6.md` | MW 200-1000 expansion (**current best model**) |
| `docs/phase7.md` | Conformer ensemble / Hybrid 2D+3D / 300k scaling (**active work**) |

## Key technical constraints
- **Python env**: Always use `.venv\Scripts\python.exe` — system Python lacks torch/pyg
- **Train-inference consistency**: Training and inference MUST use identical conformer method (ETKDG). Never mix PM6 training coords with ETKDG inference.
- **GPU training**: Done on Kaggle/Colab (free GPU). Local has RTX 5060.
- **Target columns**: `homo`, `lumo`, `gap` (eV, B3LYP Kohn-Sham, NOT experimental values)
- **Don't re-run completed experiments** — cite existing results from `results/phase{N}/`
- **Test scripts locally before delivering**

## Pipeline: SMILES → Prediction
1. SMILES → RDKit Mol → ETKDG 3D conformer
2. 3D coords + atomic numbers → PyG `Data` graph (Gasteiger charges, edge by cutoff)
3. `SchNetWrapper` forward → 3 outputs (HOMO, LUMO, Gap)

## Project layout
```
src/molgap/
  constants.py   # All path constants and model configs (single source of truth)
  graphs.py      # smiles_to_pyg(), smiles_list_to_pyg(), build_labeled_graphs()
  inference.py   # load_model(), predict_smiles(), predict_smiles_batch()
  schnet.py      # SchNetWrapper (3D GNN + optional 2D fusion)
  utils.py       # Splits, metrics, SMILES helpers, fingerprints
scripts/pipeline/      # Data fetch, clean, features
scripts/phase{1-7}/    # Per-phase experiment scripts
data/raw/              # PubChemQC CSVs
data/commercial/       # Commercial molecule lists
models/                # .pt checkpoints
results/phase{1-7}/    # Metrics, plots, JSONs
docs/                  # Per-phase detailed docs
```

## Package install
`pip install -e .` (editable install via pyproject.toml) — eliminates sys.path hacks
