"""Evaluate frozen GPS7/GPS9/GPS11 fusion candidates on fixed PCQM valid rows."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from molgap.gps import GPSWrapper
from molgap.graphs import smiles_to_2d_pyg
from molgap.multi2d_router_fusion import (
    EXPERTS,
    load_dense_gate_checkpoint,
    predict_dense_gate,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def atomic_csv(payload: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    payload.to_csv(temporary, index=False)
    os.replace(temporary, path)


@torch.inference_mode()
def predict_model(
    model: GPSWrapper,
    graphs: list,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    chunks = []
    model.eval()
    for batch in DataLoader(graphs, batch_size=batch_size, shuffle=False):
        batch = batch.to(device)
        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
            chunks.append(
                model(
                    batch.x,
                    batch.edge_index,
                    batch.edge_attr,
                    batch.batch,
                ).float().cpu()
            )
    return torch.cat(chunks).numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pcqm", type=Path, required=True)
    parser.add_argument("--gps7", type=Path, required=True)
    parser.add_argument("--gps9", type=Path, required=True)
    parser.add_argument("--gps11-160", type=Path, required=True)
    parser.add_argument("--dense-gates", type=Path, nargs="+", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    frame = pd.read_csv(args.pcqm)
    n_input = len(frame)
    required = {"cid", "smiles", "gap"}
    if missing := required.difference(frame):
        raise ValueError(f"PCQM input misses {sorted(missing)}")
    graphs, kept = [], []
    for index, smiles in enumerate(frame.smiles.astype(str)):
        graph = smiles_to_2d_pyg(smiles)
        if graph is not None:
            graphs.append(graph)
            kept.append(index)
    frame = frame.iloc[kept].reset_index(drop=True)
    if len(frame) < 0.99 * n_input:
        raise RuntimeError("Unexpected PCQM graph attrition")
    checkpoints = {
        "gps7": (args.gps7, 7, 192),
        "gps9": (args.gps9, 9, 192),
        "gps11_160": (args.gps11_160, 11, 160),
    }
    predictions = []
    for name in EXPERTS:
        path, layers, hidden = checkpoints[name]
        model = GPSWrapper(
            hidden_channels=hidden,
            num_layers=layers,
            num_heads=4,
            dropout=0.05,
        ).to(device)
        model.load_state_dict(
            torch.load(path, map_location=device, weights_only=True),
            strict=True,
        )
        predictions.append(predict_model(model, graphs, device, args.batch_size))
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    predictions = np.stack(predictions, axis=1)
    dense_predictions = []
    for path in args.dense_gates:
        gate = load_dense_gate_checkpoint(path, device=device)
        dense_prediction, _ = predict_dense_gate(gate, predictions)
        dense_predictions.append(dense_prediction)
    dense_prediction = np.mean(dense_predictions, axis=0)
    dual_prediction = predictions[:, :2].mean(axis=1)
    equal_three_prediction = predictions.mean(axis=1)
    truth = frame.gap.to_numpy(np.float64)
    methods = {
        **{
            expert: predictions[:, index, 2]
            for index, expert in enumerate(EXPERTS)
        },
        "equal_gps7_gps9": dual_prediction[:, 2],
        "equal_three": equal_three_prediction[:, 2],
        "dense_soft_gate": dense_prediction[:, 2],
    }
    metrics = {
        name: {"gap_mae_eV": float(np.abs(value - truth).mean())}
        for name, value in methods.items()
    }
    output = frame.loc[:, ["cid", "smiles", "gap"]].copy()
    for name, value in methods.items():
        output[f"{name}_gap"] = value
    atomic_csv(output, args.out_dir / "predictions.csv")
    result = {
        "experiment": "three_gps_fusion_pcqm_valid",
        "status": "complete",
        "n_input": n_input,
        "n_valid": len(frame),
        "device": str(device),
        "metrics": metrics,
        "inputs": {
            "pcqm": {"path": str(args.pcqm), "sha256": sha256(args.pcqm)},
            "checkpoints": {
                name: {"path": str(value[0]), "sha256": sha256(value[0])}
                for name, value in checkpoints.items()
            },
            "dense_gates": [
                {"path": str(path), "sha256": sha256(path)}
                for path in args.dense_gates
            ],
        },
        "sealed_20k_used": False,
        "production_registry_changed": False,
    }
    atomic_json(result, args.out_dir / "metrics.json")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
