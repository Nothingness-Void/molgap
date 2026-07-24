# MolGap

Machine-learning prediction of gas-phase B3LYP and near-GW HOMO, LUMO, and
HOMO-LUMO gap for organic molecules.

Predictions are electronic-structure values, not experimental solid-state
IP/EA. The current recommended model, open decision gate, and remote jobs are
listed only in `CURRENT_STATE.md`.

## Install

Use the repository virtual environment:

```powershell
.venv\Scripts\python.exe -m pip install -e ".[test]"
```

Core runtime packages include PyTorch, PyTorch Geometric, RDKit, pandas, NumPy,
scikit-learn, and Optuna. Platform-specific environments are documented by the
relevant operations guide.

## Basic Inference

Choose the loader named in `CURRENT_STATE.md`. The routed-hybrid API, for
example, is used as follows:

```python
from molgap.inference import (
    load_routed_dual_gps_hybrid,
    predict_smiles_batch_routed_dual_gps,
)

models = load_routed_dual_gps_hybrid()
valid_idx, predictions, routed = predict_smiles_batch_routed_dual_gps(
    ["c1ccccc1"],
    models=models,
)
```

Outputs are ordered as `homo`, `lumo`, and `gap` in eV. Runtime constraints are
defined in `AGENTS.md`.

## Navigation

The complete reading protocol, document map, and hard constraints are in
`AGENTS.md`. Do not reconstruct live status by scanning phase or result files.

## Public API

The lazy package exports and implementation live in `src/molgap/__init__.py`
and `src/molgap/inference.py`. Supported families include:

- registry-based single-hybrid loading and batch prediction;
- routed dual-GPS hybrid loading and batch prediction;
- legacy 3D-only helpers;
- conformer-ensemble helpers;
- Delta/UQ helpers for explicitly selected historical bundles.

Inspect function docstrings for return shapes and optional arguments. Do not
infer the recommended registry key from an old experiment document.
