"""Unified model loading and inference pipeline."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from .constants import (
    MODEL_PHASE6, GRAPHS_PHASE6, PARAMS_PHASE6, TARGET_COLS, SEED,
)
from .graphs import smiles_to_pyg, smiles_list_to_pyg, smiles_to_pyg_ensemble
from .schnet import SchNetWrapper
from .utils import create_split_indices


def load_normalization_stats(
    graphs_path: str | None = None,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Load y_mean and y_std from a graph cache's training split."""
    graphs_path = graphs_path or GRAPHS_PHASE6
    graphs = torch.load(graphs_path, weights_only=False)
    train_idx, _, _ = create_split_indices(len(graphs), random_state=seed)
    train_y = np.stack([graphs[i].y.squeeze(0).numpy() for i in train_idx])
    y_mean = train_y.mean(axis=0)
    y_std = train_y.std(axis=0)
    y_std[y_std < 1e-6] = 1.0
    del graphs
    return y_mean, y_std


def load_model(
    model_path: str | None = None,
    params: dict | None = None,
    graphs_path: str | None = None,
    *,
    use_charges: bool = True,
    n_desc: int = 0,
    device: torch.device | str | None = None,
) -> tuple[SchNetWrapper, np.ndarray, np.ndarray, torch.device]:
    """Load a SchNet model with its normalization stats.

    Defaults to the Phase 6 best model if no arguments are given.
    """
    model_path = model_path or MODEL_PHASE6
    params = params or PARAMS_PHASE6
    graphs_path = graphs_path or GRAPHS_PHASE6

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    y_mean, y_std = load_normalization_stats(graphs_path)

    model = SchNetWrapper(**params, use_charges=use_charges, n_desc=n_desc).to(device)
    model.load_state_dict(
        torch.load(model_path, weights_only=True, map_location=device)
    )
    model.eval()
    return model, y_mean, y_std, device


def predict_graphs(
    model: SchNetWrapper,
    pyg_list: list,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    device: torch.device,
    *,
    batch_size: int = 64,
) -> np.ndarray:
    """Run inference on a list of PyG graphs. Returns denormalized predictions."""
    from torch_geometric.loader import DataLoader

    loader = DataLoader(pyg_list, batch_size=batch_size)
    preds = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            charges = batch.charges if hasattr(batch, "charges") else None
            desc = batch.desc if hasattr(batch, "desc") else None
            out = model(batch.z, batch.pos, batch.batch, charges=charges, desc=desc)
            preds.append(out.cpu().numpy() * y_std + y_mean)
    return np.vstack(preds)


def predict_smiles(
    smiles: str,
    model: SchNetWrapper | None = None,
    y_mean: np.ndarray | None = None,
    y_std: np.ndarray | None = None,
    device: torch.device | None = None,
) -> dict[str, float] | None:
    """Predict HOMO/LUMO/Gap for a single SMILES. Returns dict or None on failure."""
    if model is None:
        model, y_mean, y_std, device = load_model()

    data = smiles_to_pyg(smiles)
    if data is None:
        return None

    preds = predict_graphs(model, [data], y_mean, y_std, device)
    return {t: float(preds[0, i]) for i, t in enumerate(TARGET_COLS)}


def predict_smiles_batch(
    smiles_list: list[str],
    model: SchNetWrapper | None = None,
    y_mean: np.ndarray | None = None,
    y_std: np.ndarray | None = None,
    device: torch.device | None = None,
    *,
    batch_size: int = 64,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Predict HOMO/LUMO/Gap for a list of SMILES. Returns DataFrame with results."""
    if model is None:
        model, y_mean, y_std, device = load_model()

    graphs, valid_idx = smiles_list_to_pyg(
        smiles_list, show_progress=show_progress,
    )

    if not graphs:
        return pd.DataFrame(columns=["smiles", "homo", "lumo", "gap", "success"])

    preds = predict_graphs(model, graphs, y_mean, y_std, device, batch_size=batch_size)

    rows = []
    pred_i = 0
    for i, smi in enumerate(smiles_list):
        if i in valid_idx:
            rows.append({
                "smiles": smi,
                "homo": float(preds[pred_i, 0]),
                "lumo": float(preds[pred_i, 1]),
                "gap": float(preds[pred_i, 2]),
                "success": True,
            })
            pred_i += 1
        else:
            rows.append({
                "smiles": smi,
                "homo": np.nan,
                "lumo": np.nan,
                "gap": np.nan,
                "success": False,
            })
    return pd.DataFrame(rows)


def predict_smiles_ensemble(
    smiles: str,
    k: int = 8,
    model: SchNetWrapper | None = None,
    y_mean: np.ndarray | None = None,
    y_std: np.ndarray | None = None,
    device: torch.device | None = None,
) -> dict[str, float] | None:
    """Predict with k conformers and average. Returns dict with mean and std."""
    if model is None:
        model, y_mean, y_std, device = load_model()

    graphs = smiles_to_pyg_ensemble(smiles, k=k)
    if not graphs:
        return None

    preds = predict_graphs(model, graphs, y_mean, y_std, device)
    result = {}
    for i, t in enumerate(TARGET_COLS):
        result[t] = float(preds[:, i].mean())
        result[f"{t}_std"] = float(preds[:, i].std())
    return result
