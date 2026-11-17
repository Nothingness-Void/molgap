"""Reusable external evaluation for frozen dual-2D static blends."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from molgap.graphs import smiles_to_2d_pyg
from molgap.router import paired_bootstrap_mean

from .evaluation import TARGETS, apply_static_weights
from .models import make_expert


def build_external_graphs(smiles: Iterable[str]) -> tuple[list, np.ndarray]:
    """Build ordered 2D graphs and return their source-row positions."""
    graphs, positions = [], []
    for position, value in enumerate(smiles):
        graph = smiles_to_2d_pyg(str(value))
        if graph is None:
            continue
        graph.external_row = torch.tensor([position], dtype=torch.long)
        graphs.append(graph)
        positions.append(position)
    return graphs, np.asarray(positions, dtype=np.int64)


@torch.no_grad()
def predict_seed_experts(
    graphs: list,
    *,
    checkpoint_dir,
    seed: int,
    device: torch.device,
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ordered Local/GPS predictions, shape ``[n, 3, 2]``."""
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False, num_workers=0)
    outputs, order = [], None
    for kind in ("local", "global"):
        model = make_expert(kind).to(device)
        model.load_state_dict(torch.load(
            checkpoint_dir / f"expert_{kind}" / f"seed{seed}.pt",
            map_location=device,
            weights_only=True,
        ))
        model.eval()
        prediction, positions = [], []
        for batch in loader:
            batch = batch.to(device)
            prediction.append(model(
                batch.x, batch.edge_index, batch.edge_attr, batch.batch
            ).float().cpu().numpy())
            positions.append(batch.external_row.view(-1).cpu().numpy())
        current_order = np.concatenate(positions).astype(np.int64)
        if order is None:
            order = current_order
        elif not np.array_equal(order, current_order):
            raise RuntimeError(f"{kind} prediction order changed")
        outputs.append(np.concatenate(prediction).astype(np.float64))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return np.stack(outputs, axis=-1), order


def evaluate_seed(
    y: np.ndarray,
    expert_predictions: np.ndarray,
    static_weights: np.ndarray,
    reference_name: str,
    *,
    target_indices: tuple[int, ...] = (0, 1, 2),
    bootstrap_seed: int = 42,
) -> dict:
    """Evaluate a frozen static blend against its predeclared internal reference."""
    y = np.asarray(y, dtype=np.float64)
    expert_predictions = np.asarray(expert_predictions, dtype=np.float64)
    predictions = {
        "local": expert_predictions[:, :, 0],
        "global": expert_predictions[:, :, 1],
        "static_weights": apply_static_weights(expert_predictions, static_weights),
    }
    reference = predictions[reference_name]
    result = {"n": int(len(y)), "reference": reference_name, "methods": {}}
    for name, prediction in predictions.items():
        absolute = np.abs(prediction - y)
        result["methods"][name] = {
            TARGETS[index]: {"mae": float(absolute[:, index].mean())}
            for index in target_indices
        }
    deltas = {}
    for index in target_indices:
        target = TARGETS[index]
        delta = np.abs(predictions["static_weights"][:, index] - y[:, index]) - np.abs(
            reference[:, index] - y[:, index]
        )
        deltas[target] = paired_bootstrap_mean(delta, seed=bootstrap_seed + index)
        deltas[target]["improvement_eV"] = -deltas[target]["delta"]
    result["static_vs_reference"] = deltas
    return result
