"""Reusable inference and evaluation helpers for frozen dual-GPS experts."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import TensorDataset
from torch_geometric.loader import DataLoader as GeometricDataLoader

from .fusion import DualGPSFusionHead
from .gps import GPSWrapper
from .graphs import smiles_to_2d_pyg
from .router import paired_bootstrap_mean


TARGETS = ("homo", "lumo", "gap")


@dataclass(frozen=True)
class DualGPSExpertPaths:
    name: str
    gps7: Path
    gps9: Path
    head: Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stage_dual_gps_embedding_prefixes(
    experts: Mapping[str, tuple[Path, Path]],
    *,
    rows: int,
    out_dir: Path,
    report_path: Path,
) -> dict:
    """Persist aligned FP16 embedding prefixes as independent expert chunks."""
    if rows <= 0 or not experts:
        raise ValueError("rows and experts must be non-empty")
    out_dir.mkdir(parents=True, exist_ok=True)
    expected_source = torch.arange(rows, dtype=torch.long)
    records = []
    for name, (gps7_path, gps9_path) in experts.items():
        gps7 = torch.load(gps7_path, map_location="cpu", weights_only=False, mmap=True)
        gps9 = torch.load(gps9_path, map_location="cpu", weights_only=False, mmap=True)
        source7 = gps7["source_idx"].long()
        source9 = gps9["source_idx"].long()
        if len(source7) < rows or not torch.equal(source7, source9):
            raise RuntimeError(f"Invalid or misaligned embeddings for {name}")
        if not torch.equal(source7[:rows], expected_source):
            raise RuntimeError(f"{name} does not contain a contiguous {rows:,}-row prefix")
        payload = {
            "gps7": gps7["embeddings"][:rows].to(torch.float16).clone(),
            "gps9": gps9["embeddings"][:rows].to(torch.float16).clone(),
            "source_idx": expected_source,
            "expert": name,
        }
        if payload["gps7"].shape != (rows, 192) or payload["gps9"].shape != (rows, 192):
            raise RuntimeError(f"Unexpected staged shape for {name}")
        out_path = out_dir / f"{name}_1m_fp16.pt"
        temporary = out_path.with_name(f".{out_path.name}.tmp")
        torch.save(payload, temporary)
        os.replace(temporary, out_path)
        records.append(
            {
                "expert": name,
                "gps7_source": str(gps7_path),
                "gps9_source": str(gps9_path),
                "rows": rows,
                "embedding_dim_each": 192,
                "dtype": "float16",
                "path": str(out_path),
                "bytes": out_path.stat().st_size,
                "sha256": _sha256(out_path),
            }
        )
        del gps7, gps9, payload
    report = {"format": "molgap-dual-gps-prefix-v1", "rows": rows, "experts": records}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_name(f".{report_path.name}.tmp")
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    os.replace(temporary, report_path)
    return report


def _load_state(model: torch.nn.Module, path: Path, device: torch.device) -> torch.nn.Module:
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return model.to(device).eval()


def load_dual_gps_experts(
    specs: Sequence[DualGPSExpertPaths], device: torch.device
) -> dict[str, tuple[torch.nn.Module, torch.nn.Module, torch.nn.Module]]:
    """Load several GPS7/GPS9/head experts with the controlled Phase 8 shape."""
    if not specs:
        raise ValueError("At least one dual-GPS expert is required")
    names = [spec.name for spec in specs]
    if len(set(names)) != len(names):
        raise ValueError(f"Expert names must be unique: {names}")
    gps_args = dict(hidden_channels=192, num_heads=4, dropout=0.05)
    return {
        spec.name: (
            _load_state(GPSWrapper(num_layers=7, **gps_args), spec.gps7, device),
            _load_state(GPSWrapper(num_layers=9, **gps_args), spec.gps9, device),
            _load_state(DualGPSFusionHead(hidden=192), spec.head, device),
        )
        for spec in specs
    }


def load_gps_predictors(
    specs: Mapping[str, Path],
    device: torch.device,
    *,
    num_layers: int = 7,
) -> dict[str, torch.nn.Module]:
    """Load direct three-target GPS predictors with a shared architecture."""
    if not specs:
        raise ValueError("At least one GPS predictor is required")
    gps_args = dict(
        hidden_channels=192, num_layers=num_layers, num_heads=4, dropout=0.05
    )
    return {
        name: _load_state(GPSWrapper(**gps_args), path, device)
        for name, path in specs.items()
    }


def predict_gps_models(
    smiles: Iterable[object],
    models: Mapping[str, torch.nn.Module],
    device: torch.device,
    *,
    graph_batch_size: int = 256,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Predict aligned rows with one or more direct-output GPS models."""
    graphs, kept = [], []
    for position, value in enumerate(smiles):
        graph = smiles_to_2d_pyg(str(value))
        if graph is not None:
            graphs.append(graph)
            kept.append(position)
    if not graphs:
        raise RuntimeError("No valid 2D graphs were constructed")
    predictions = {name: [] for name in models}
    with torch.inference_mode():
        for batch in GeometricDataLoader(
            graphs, batch_size=graph_batch_size, shuffle=False
        ):
            batch = batch.to(device)
            for name, model in models.items():
                predictions[name].append(
                    model(
                        batch.x, batch.edge_index, batch.edge_attr, batch.batch
                    ).float().cpu()
                )
    return np.asarray(kept, dtype=np.int64), {
        name: torch.cat(chunks).numpy() for name, chunks in predictions.items()
    }


def predict_dual_gps_experts(
    smiles: Iterable[object],
    experts: Mapping[str, tuple[torch.nn.Module, torch.nn.Module, torch.nn.Module]],
    device: torch.device,
    *,
    graph_batch_size: int = 256,
    head_batch_size: int = 4096,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Predict aligned rows for arbitrary frozen dual-GPS experts."""
    graphs, kept = [], []
    for position, value in enumerate(smiles):
        graph = smiles_to_2d_pyg(str(value))
        if graph is not None:
            graphs.append(graph)
            kept.append(position)
    if not graphs:
        raise RuntimeError("No valid 2D graphs were constructed")

    encoded7 = {name: [] for name in experts}
    encoded9 = {name: [] for name in experts}
    with torch.inference_mode():
        for batch in GeometricDataLoader(
            graphs, batch_size=graph_batch_size, shuffle=False
        ):
            batch = batch.to(device)
            for name, (gps7, gps9, _) in experts.items():
                encoded7[name].append(
                    gps7.encode(
                        batch.x, batch.edge_index, batch.edge_attr, batch.batch
                    ).float().cpu()
                )
                encoded9[name].append(
                    gps9.encode(
                        batch.x, batch.edge_index, batch.edge_attr, batch.batch
                    ).float().cpu()
                )

    predictions: dict[str, np.ndarray] = {}
    with torch.inference_mode():
        for name, (_, _, head) in experts.items():
            h7 = torch.cat(encoded7[name])
            h9 = torch.cat(encoded9[name])
            chunks = []
            loader = TorchDataLoader(
                TensorDataset(h7, h9), batch_size=head_batch_size, shuffle=False
            )
            for batch7, batch9 in loader:
                chunks.append(head(batch7.to(device), batch9.to(device)).float().cpu())
            predictions[name] = torch.cat(chunks).numpy()
    return np.asarray(kept, dtype=np.int64), predictions


def add_mean_ensembles(
    predictions: Mapping[str, np.ndarray],
    ensembles: Mapping[str, Sequence[str]],
) -> dict[str, np.ndarray]:
    """Return experts plus named equal-weight, target-independent ensembles."""
    output = {name: np.asarray(value) for name, value in predictions.items()}
    for name, members in ensembles.items():
        if name in output:
            raise ValueError(f"Ensemble name collides with an expert: {name}")
        if len(members) < 2:
            raise ValueError(f"Ensemble {name!r} needs at least two members")
        missing = [member for member in members if member not in predictions]
        if missing:
            raise KeyError(f"Ensemble {name!r} has unknown members: {missing}")
        output[name] = np.mean([predictions[member] for member in members], axis=0)
    return output


def metric_block(
    y_true: np.ndarray,
    prediction: np.ndarray,
    target_names: Sequence[str] = TARGETS,
) -> dict[str, dict[str, float]]:
    y_true = np.asarray(y_true, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if y_true.shape != prediction.shape or y_true.shape[1] != len(target_names):
        raise ValueError(
            f"Shape mismatch: truth={y_true.shape}, prediction={prediction.shape}, "
            f"targets={len(target_names)}"
        )
    result = {}
    for index, target in enumerate(target_names):
        result[target] = {
            "mae_eV": float(np.abs(prediction[:, index] - y_true[:, index]).mean()),
            "r2": float(r2_score(y_true[:, index], prediction[:, index])),
        }
    if len(target_names) > 1:
        result["average"] = {
            "mae_eV": float(np.abs(prediction - y_true).mean()),
            "r2": float(np.mean([result[target]["r2"] for target in target_names])),
        }
    return result


def delta_block(
    y_true: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    *,
    target_names: Sequence[str] = TARGETS,
    n_bootstrap: int = 10_000,
    seed: int = 42,
) -> dict[str, dict[str, object]]:
    y_true = np.asarray(y_true, dtype=np.float64)
    baseline = np.asarray(baseline, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    result = {}
    for index, target in enumerate(target_names):
        delta = np.abs(candidate[:, index] - y_true[:, index]) - np.abs(
            baseline[:, index] - y_true[:, index]
        )
        result[target] = paired_bootstrap_mean(
            delta, n_bootstrap=n_bootstrap, seed=seed + index
        )
    if len(target_names) > 1:
        delta = np.abs(candidate - y_true).mean(axis=1) - np.abs(
            baseline - y_true
        ).mean(axis=1)
        result["average"] = paired_bootstrap_mean(
            delta, n_bootstrap=n_bootstrap, seed=seed + len(target_names)
        )
    return result


def targetwise_oracle(
    y_true: np.ndarray, predictions: Mapping[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """Return the optimistic per-target expert oracle and selected indices."""
    names = list(predictions)
    if not names:
        raise ValueError("Oracle needs at least one prediction")
    stack = np.stack([predictions[name] for name in names], axis=0)
    selected = np.abs(stack - np.asarray(y_true)[None, ...]).argmin(axis=0)
    oracle = np.take_along_axis(stack, selected[None, ...], axis=0)[0]
    return oracle, selected
