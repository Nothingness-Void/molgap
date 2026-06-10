# MolGap

Machine learning prediction of HOMO, LUMO, and HOMO-LUMO gap for organic electronic molecules (OLED, organic thin-film, OPV).

Trained on [PubChemQC](https://huggingface.co/datasets/molssiai-hub/pubchemqc-b3lyp) B3LYP/6-31G\* data (~85M molecules), using a 3D graph neural network (SchNet) with ETKDG conformers.

## Quick Start

```bash
# Install (editable mode)
pip install -e .

# Predict a single molecule
python -c "
from molgap.inference import predict_smiles
print(predict_smiles('c1ccc2c(c1)cc1ccc3ccccc3c1n2'))
"
# → {'homo': -5.577, 'lumo': -1.883, 'gap': 3.693}

# Batch prediction
python -c "
from molgap.inference import predict_smiles_batch
df = predict_smiles_batch(['c1ccccc1', 'c1ccc(cc1)N(c1ccccc1)c1ccccc1'])
print(df)
"
```

## Current Best Model

| Metric | Value |
|--------|-------|
| Architecture | SchNet (3D GNN) + Gasteiger charges |
| Training data | 44.8k molecules, CHONSFCl, MW 200-1000 |
| Internal test R² | 0.882 |
| OOD R² (500 molecules) | 0.797 |
| Gaussian B3LYP Gap MAE | 0.223 eV |
| Checkpoint | `models/gnn_schnet_3d_optuna_expanded.pt` |

## Prediction Pipeline

```
SMILES → RDKit Mol → ETKDG 3D conformer → PyG graph (+ Gasteiger charges) → SchNet → HOMO/LUMO/Gap (eV)
```

**Important**: Predicted values are B3LYP Kohn-Sham orbital energies, not experimental ionization potentials / electron affinities. Systematic offsets exist: HOMO ~0.5-0.7 eV shallower, LUMO ~1.3-2.1 eV shallower than experiment.

## Project Structure

```
src/molgap/
  constants.py       # Centralized paths and model configurations
  graphs.py          # SMILES → PyG 3D graph conversion
  inference.py       # Model loading and prediction API
  schnet.py          # SchNetWrapper (3D GNN + optional 2D descriptor fusion)
  utils.py           # Splits, metrics, SMILES/RDKit helpers

scripts/
  pipeline/          # Data fetch, clean, feature engineering
  phase1/            # Traditional ML baseline (LightGBM, R²=0.921)
  phase2/            # Generalization study (element/MW expansion)
  phase3/            # 30k data scaling + ML optimization
  phase4/            # SchNet GNN + ETKDG consistency (R²=0.896)
  phase5/            # Validation (OOD, Gaussian B3LYP, experimental)
  phase6/            # MW expansion to 200-1000 (R²=0.882)
  phase7/            # Conformer ensemble, Hybrid 2D+3D, 300k scaling

docs/                # Per-phase detailed documentation
data/raw/            # PubChemQC CSV data
data/commercial/     # Commercial molecule lists
models/              # Trained model checkpoints (.pt)
results/             # Per-phase metrics, plots, JSONs
```

## Experiment History

| Phase | Data | Best R² | Key Finding |
|-------|------|---------|-------------|
| 1 | 10-30k CHON MW 200-300 | 0.921 (LightGBM) | Traditional ML baseline |
| 2 | 10k/step, expanding | 0.901→0.874 | Smooth R² decay with diversity |
| 3 | 30k CHONSFCl MW 200-500 | 0.885 (LightGBM) | Feature selection helps, but ceiling at ~0.89 |
| 4 | 30k CHONSFCl MW 200-503 | 0.896 (SchNet) | 3D GNN beats ML; PM6/ETKDG mismatch discovered |
| 5 | 100 OOD + 10 commercial | OOD R²=0.849 | B3LYP systematic bias quantified |
| 6 | 44.8k MW 200-1000 | 0.882 (SchNet) | MW expansion, Gaussian Gap MAE improved 37% |
| 7 | Various | +2.5% R² (ensemble) | Conformer ensemble marginal; 2D+3D hybrid in progress |

See `docs/phase{N}.md` for detailed per-phase documentation, and `ROADMAP.md` for the current priority checklist.

## Requirements

- Python >= 3.9
- PyTorch + PyTorch Geometric
- RDKit
- scikit-learn, pandas, numpy, tqdm, optuna

```bash
pip install -e .
pip install torch torch_geometric rdkit scikit-learn pandas numpy tqdm optuna lightgbm
```

## API Reference

### `molgap.inference`

```python
# Single prediction
predict_smiles(smiles: str) -> dict[str, float] | None

# Batch prediction
predict_smiles_batch(smiles_list: list[str]) -> pd.DataFrame

# Ensemble prediction (multiple conformers, averaged)
predict_smiles_ensemble(smiles: str, k: int = 8) -> dict[str, float] | None

# Low-level: load model manually
load_model(model_path=None, params=None, graphs_path=None)
    -> (model, y_mean, y_std, device)
```

### `molgap.graphs`

```python
# Single SMILES → PyG Data
smiles_to_pyg(smiles: str) -> Data | None

# Batch conversion
smiles_list_to_pyg(smiles_list: list[str]) -> (list[Data], list[int])

# Build training graphs with labels
build_labeled_graphs(smiles_list, targets) -> list[Data]
```

## Data Source

[PubChemQC B3LYP/6-31G\*//PM6](https://huggingface.co/datasets/molssiai-hub/pubchemqc-b3lyp) — ~85 million molecules with DFT-computed electronic properties. Hosted on Hugging Face, fetched via streaming API.

## License

Research use. PubChemQC data is subject to its original license terms.
