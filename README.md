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

> **Current recommended model, performance, and next steps:** see
> [`CURRENT_STATE.md`](CURRENT_STATE.md). Per-phase history: [`docs/`](docs/).
> This README only covers what is stable: what the project is, install, basic inference.

## Prediction Pipeline

```
SMILES ─┬─ 2D bond graph ───────────────→ GPS 2D ──┐
        └─ ETKDG 3D conformer + charges → SchNet 3D ┴─ gate fusion → HOMO/LUMO/Gap (eV)
```

**Important**: Predicted values are B3LYP Kohn-Sham orbital energies, not experimental
IP/EA. Known systematic offsets vs experiment exist (see `CURRENT_STATE.md`); Gap is
the most reliable output.

## Project Structure

Code map and module boundaries ("to change X, edit which file") live in
[`ARCHITECTURE.md`](ARCHITECTURE.md). In short: reusable logic in `src/molgap/`,
thin CLI wrappers in `scripts/phase{1-7}/`, outputs in `results/`, checkpoints in
`models/`, per-phase docs in `docs/`.

## Experiment History

Per-phase background, experiments, and conclusions live in [`docs/phase{N}.md`](docs/).
Phase 7 (300k + 2D/3D hybrid) is the current best — see [`docs/phase7.md`](docs/phase7.md).
Task priorities are in [`ROADMAP.md`](ROADMAP.md).

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
