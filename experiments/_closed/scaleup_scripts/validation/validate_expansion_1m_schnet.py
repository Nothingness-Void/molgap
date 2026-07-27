"""Validate the Colab 1M SchNet checkpoint and audit 2D embedding alignment.

The 1M 3D cache is intentionally stored as an existing 500K cache plus a
separate ETKDG top-up.  This script reconstructs the exact Colab graph order
and RandomState(42) split without writing a duplicate combined cache.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, r2_score
from torch_geometric.loader import DataLoader

from molgap.schnet import SchNetWrapper

TARGETS = ("HOMO", "LUMO", "Gap")


def _metrics(pred: np.ndarray, true: np.ndarray) -> dict:
    values = {}
    for column, name in enumerate(TARGETS):
        values[name] = {
            "mae_eV": float(mean_absolute_error(true[:, column], pred[:, column])),
            "r2": float(r2_score(true[:, column], pred[:, column])),
        }
    values["average"] = {
        "mae_eV": float(np.mean([values[name]["mae_eV"] for name in TARGETS])),
        "r2": float(np.mean([values[name]["r2"] for name in TARGETS])),
    }
    return values


def _source_indices(graphs) -> np.ndarray:
    return torch.cat([graph.source_idx.view(-1).cpu() for graph in graphs]).numpy()


def _audit_embedding(path: Path, graph_source: np.ndarray) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    embedding = payload["embeddings"]
    source = payload["source_idx"].cpu().numpy()
    if embedding.ndim != 2 or embedding.shape[1] != 192:
        raise ValueError(f"Unexpected embedding shape in {path}: {tuple(embedding.shape)}")
    if len(source) != len(np.unique(source)):
        raise ValueError(f"Duplicate source_idx in {path}")
    graph_hits = int(np.isin(graph_source, source, assume_unique=False).sum())
    return {
        "path": str(path),
        "embedding_shape": list(embedding.shape),
        "source_min": int(source.min()),
        "source_max": int(source.max()),
        "unique_source_idx": int(len(source)),
        "covers_all_3d_graphs": graph_hits == len(graph_source),
        "covered_3d_graphs": graph_hits,
    }


@torch.no_grad()
def _evaluate(model, graphs, device: torch.device, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False, num_workers=0)
    predictions, targets = [], []
    model.eval()
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        output = model(batch.z, batch.pos, batch.batch, charges=getattr(batch, "charges", None))
        predictions.append(output.float().cpu().numpy())
        targets.append(batch.y.float().cpu().numpy())
    return np.concatenate(predictions), np.concatenate(targets)


@torch.no_grad()
def _extract_embeddings(model, graphs, device: torch.device, batch_size: int, out: Path) -> None:
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False, num_workers=0)
    embeddings, source_idx = [], []
    model.eval()
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        embedding = model.encode(batch.z, batch.pos, batch.batch, charges=getattr(batch, "charges", None))
        embeddings.append(embedding.float().cpu())
        source_idx.append(batch.source_idx.view(-1).cpu())
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"embeddings": torch.cat(embeddings), "source_idx": torch.cat(source_idx)}, out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-graphs", type=Path, required=True)
    parser.add_argument("--topup-graphs", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--gps-embedding", type=Path, required=True)
    parser.add_argument("--gps-depth9-embedding", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--embeddings-out", type=Path, default=None,
                        help="optional full aligned SchNet embedding payload")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    base_graphs = torch.load(args.base_graphs, map_location="cpu", weights_only=False)
    topup_graphs = torch.load(args.topup_graphs, map_location="cpu", weights_only=False)
    graphs = base_graphs + topup_graphs
    source_idx = _source_indices(graphs)
    if len(source_idx) != len(np.unique(source_idx)):
        raise ValueError("3D base/top-up source_idx overlap")
    if not (np.all(source_idx[:len(base_graphs)] < 500000) and np.all(source_idx[len(base_graphs):] >= 500000)):
        raise ValueError("Unexpected base/top-up source_idx boundary")

    split = np.random.RandomState(42).permutation(len(graphs))
    test_start = int(0.9 * len(graphs))
    test_graphs = [graphs[index] for index in split[test_start:]]
    model = SchNetWrapper(
        hidden_channels=192,
        num_filters=192,
        num_interactions=6,
        num_gaussians=50,
        cutoff=6.0,
        dropout=0.0,
        use_charges=True,
    )
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    prediction, target = _evaluate(model, test_graphs, device, args.batch_size)
    if args.embeddings_out is not None:
        _extract_embeddings(model, graphs, device, args.batch_size, args.embeddings_out)

    report = {
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "n_base_graphs": len(base_graphs),
        "n_topup_graphs": len(topup_graphs),
        "n_total_graphs": len(graphs),
        "n_test_graphs": len(test_graphs),
        "test_metrics": _metrics(prediction, target),
        "embeddings_out": None if args.embeddings_out is None else str(args.embeddings_out),
        "gps_embedding": _audit_embedding(args.gps_embedding, source_idx),
        "gps_depth9_embedding": _audit_embedding(args.gps_depth9_embedding, source_idx),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Report -> {args.out}")


if __name__ == "__main__":
    main()
