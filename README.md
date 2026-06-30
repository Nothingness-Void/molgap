# MolGap

Machine learning prediction of HOMO, LUMO, and HOMO-LUMO gap for organic electronic molecules (OLED, organic thin-film, OPV).

Trained on [PubChemQC](https://huggingface.co/datasets/molssiai-hub/pubchemqc-b3lyp)
B3LYP/6-31G\* data (~85M molecules). The current default B3LYP base is the Phase 8
expansion500k hybrid: GPS 2D + SchNet 3D with ETKDG conformers.

## Quick Start

```bash
# Install (editable mode)
pip install -e .

# Predict with the current recommended B3LYP hybrid
python -c "
from molgap.inference import load_hybrid, predict_smiles_batch_hybrid
models = load_hybrid()  # defaults to phase8_expansion_hybrid
vi, preds = predict_smiles_batch_hybrid(
    ['c1ccc2c(c1)cc1ccc3ccccc3c1n2'], models=models
)
print(preds[0].tolist())
"

# Batch prediction
python -c "
from molgap.inference import load_hybrid, predict_smiles_batch_hybrid
models = load_hybrid()
smiles = ['c1ccccc1', 'c1ccc(cc1)N(c1ccccc1)c1ccccc1']
valid_idx, preds = predict_smiles_batch_hybrid(smiles, models=models)
print(valid_idx, preds)
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
thin CLI wrappers in `scripts/phase{N}/`, outputs in `results/`, checkpoints in
`models/`, per-phase docs in `docs/`.

## Experiment History

Per-phase background, experiments, and conclusions live in [`docs/phase{N}.md`](docs/).
Phase 8 selected the expansion500k hybrid as the current v3 B3LYP base; see
[`docs/phase8.md`](docs/phase8.md) and
[`results/phase8/full_expansion_500k_summary.md`](results/phase8/full_expansion_500k_summary.md).
Phase 9 revalidated GW Delta-learning against v3; see
[`results/phase9/v3_delta_decision.md`](results/phase9/v3_delta_decision.md).
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
# Current recommended B3LYP hybrid
load_hybrid(key="phase8_expansion_hybrid")
predict_smiles_batch_hybrid(smiles_list: list[str], models=...)
    -> (valid_idx, preds)

# Prior v2 base
load_hybrid(key="phase8_replacement_hybrid")

# Legacy 3D-only SchNet helpers
predict_smiles(smiles: str) -> dict[str, float] | None
predict_smiles_batch(smiles_list: list[str]) -> pd.DataFrame

# Ensemble prediction (multiple conformers, averaged)
predict_smiles_ensemble(smiles: str, k: int = 8) -> dict[str, float] | None

# v3 GW Delta + calibrated UQ/OOD bundle
bundle = load_uq_bundle(results_subdir="phase10_v3")
predict_smiles_with_uq(smiles, bundle=bundle)

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
