"""Local continuation utilities for the accepted PCQM4Mv2 GINE expert."""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from ogb.utils.mol import smiles2graph
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_add_pool, global_mean_pool
from torch_geometric.nn.conv import MessagePassing

SEED = 42
OFFICIAL_TRAIN_ROWS = 3_378_606
TRAIN_SAMPLE_ROWS = 250_000
OFFICIAL_VALID_ROWS = 5_000
GRAPH_SHARD_ROWS = 25_000
EXPECTED_SELECTION_SHA256 = (
    "a7fb6eae4530ebbb64a536d4739a5d8232aeb7462fc50d7933224c8be842ba8c"
)
EXPECTED_VALID_SHA256 = (
    "81c31cd49328cccacb6ab0dc2881da7351a5e81e717381566a444f3b65c34fcf"
)
EXPECTED_SPLIT_COUNTS = {"train": 229_335, "dev": 20_662, "official": 5_000}
MODEL_CONFIG = {
    "hidden_channels": 256,
    "num_layers": 5,
    "dropout": 0.10,
    "target": "homolumogap",
    "feature_schema": "ogb_atom_bond",
    "virtual_node": True,
}


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_torch(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_train_indices(seed: int = SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.sort(
        rng.choice(
            np.arange(OFFICIAL_TRAIN_ROWS, dtype=np.int64),
            size=TRAIN_SAMPLE_ROWS,
            replace=False,
        )
    )


def expanded_train_indices(
    total_rows: int = 1_000_000,
    *,
    extra_seed: int = 43,
) -> np.ndarray:
    """Create a deterministic larger sample containing every accepted 250K row."""
    if not TRAIN_SAMPLE_ROWS <= total_rows <= OFFICIAL_TRAIN_ROWS:
        raise ValueError("expanded sample size is outside official train")
    base = selected_train_indices()
    if total_rows == len(base):
        return base
    available = np.ones(OFFICIAL_TRAIN_ROWS, dtype=bool)
    available[base] = False
    remaining = np.flatnonzero(available)
    rng = np.random.default_rng(extra_seed)
    extra = rng.choice(
        remaining,
        size=total_rows - len(base),
        replace=False,
    )
    return np.sort(np.concatenate((base, extra)).astype(np.int64, copy=False))


def array_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype=np.int64).tobytes()).hexdigest()


def load_selected_rows(
    raw_csv: Path,
    accepted_valid_predictions: Path,
) -> tuple[pd.DataFrame, dict]:
    """Recover the exact v5 250K train and fixed 5K validation identities."""
    selected = selected_train_indices()
    if array_sha256(selected) != EXPECTED_SELECTION_SHA256:
        raise RuntimeError("PCQM 250K selection hash changed")
    accepted = pd.read_csv(accepted_valid_predictions)
    valid_indices = accepted["idx"].to_numpy(dtype=np.int64)
    if len(valid_indices) != OFFICIAL_VALID_ROWS:
        raise RuntimeError("accepted PCQM validation prediction count changed")
    if array_sha256(valid_indices) != EXPECTED_VALID_SHA256:
        raise RuntimeError("PCQM fixed-validation identity hash changed")

    frame = pd.read_csv(
        raw_csv,
        usecols=["idx", "smiles", "homolumogap"],
    )
    if len(frame) != 3_746_620 or not np.array_equal(
        frame["idx"].to_numpy(), np.arange(len(frame))
    ):
        raise RuntimeError("unexpected local PCQM4Mv2 identity/order")
    needed = np.concatenate((selected, valid_indices))
    selected_rows = frame.iloc[needed].copy()
    train_rows = selected_rows.iloc[: len(selected)].copy()
    valid_rows = selected_rows.iloc[len(selected) :].copy()
    label_delta = np.abs(
        valid_rows["homolumogap"].to_numpy(dtype=np.float64)
        - accepted["gap_true_eV"].to_numpy(dtype=np.float64)
    )
    if float(label_delta.max()) > 1e-5:
        raise RuntimeError("local PCQM validation labels differ from accepted v5")
    train_rows["source_split"] = np.int8(0)
    valid_rows["source_split"] = np.int8(2)
    rows = pd.concat((train_rows, valid_rows), ignore_index=True)
    contract = {
        "format": "molgap-pcqm-gine-250k-local-contract-v1",
        "source_rows": len(rows),
        "train_sample_rows": len(train_rows),
        "official_valid_rows": len(valid_rows),
        "train_idx_sha256": array_sha256(selected),
        "valid_idx_sha256": array_sha256(valid_indices),
        "valid_label_max_abs_delta_eV": float(label_delta.max()),
        "raw_csv": str(raw_csv),
        "raw_csv_sha256": sha256_file(raw_csv),
        "accepted_valid_predictions": str(accepted_valid_predictions),
        "accepted_valid_predictions_sha256": sha256_file(
            accepted_valid_predictions
        ),
    }
    return rows, contract


def load_expanded_rows(
    raw_csv: Path,
    accepted_valid_predictions: Path,
    *,
    total_train_rows: int = 1_000_000,
    extra_seed: int = 43,
) -> tuple[pd.DataFrame, dict]:
    train_indices = expanded_train_indices(
        total_train_rows, extra_seed=extra_seed
    )
    accepted = pd.read_csv(accepted_valid_predictions)
    valid_indices = accepted["idx"].to_numpy(dtype=np.int64)
    if array_sha256(valid_indices) != EXPECTED_VALID_SHA256:
        raise RuntimeError("PCQM fixed-validation identity hash changed")
    frame = pd.read_csv(
        raw_csv,
        usecols=["idx", "smiles", "homolumogap"],
    )
    needed = np.concatenate((train_indices, valid_indices))
    selected_rows = frame.iloc[needed].copy()
    train_rows = selected_rows.iloc[: len(train_indices)].copy()
    valid_rows = selected_rows.iloc[len(train_indices) :].copy()
    label_delta = np.abs(
        valid_rows["homolumogap"].to_numpy(dtype=np.float64)
        - accepted["gap_true_eV"].to_numpy(dtype=np.float64)
    )
    if float(label_delta.max()) > 1e-5:
        raise RuntimeError("local PCQM validation labels differ from accepted v5")
    train_rows["source_split"] = np.int8(0)
    valid_rows["source_split"] = np.int8(2)
    rows = pd.concat((train_rows, valid_rows), ignore_index=True)
    return rows, {
        "format": "molgap-pcqm-gine-expanded-local-contract-v1",
        "source_rows": len(rows),
        "train_sample_rows": len(train_rows),
        "official_valid_rows": len(valid_rows),
        "contains_accepted_250k": bool(
            np.isin(selected_train_indices(), train_indices).all()
        ),
        "train_idx_sha256": array_sha256(train_indices),
        "valid_idx_sha256": array_sha256(valid_indices),
        "extra_seed": extra_seed,
        "valid_label_max_abs_delta_eV": float(label_delta.max()),
        "raw_csv": str(raw_csv),
        "raw_csv_sha256": sha256_file(raw_csv),
    }


def scaffold_bucket(smiles: str) -> int:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return -1
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        mol=molecule,
        includeChirality=True,
    )
    if not scaffold:
        scaffold = "ACYCLIC:" + Chem.MolToSmiles(
            molecule, isomericSmiles=True
        )
    digest = hashlib.sha1(scaffold.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % 10


def _build_graph(row) -> Data | None:
    try:
        graph = smiles2graph(row.smiles)
    except (AttributeError, RuntimeError, ValueError):
        return None
    if graph is None or int(graph["num_nodes"]) == 0:
        return None
    if int(row.source_split) == 0:
        bucket = scaffold_bucket(row.smiles)
        if bucket < 0:
            return None
        split_code = 1 if bucket == 0 else 0
    else:
        split_code = 2
    return Data(
        x=torch.as_tensor(graph["node_feat"], dtype=torch.long),
        edge_index=torch.as_tensor(graph["edge_index"], dtype=torch.long),
        edge_attr=torch.as_tensor(graph["edge_feat"], dtype=torch.long),
        y=torch.tensor([float(row.homolumogap)], dtype=torch.float32),
        sample_idx=torch.tensor([int(row.idx)], dtype=torch.long),
        split_code=torch.tensor([split_code], dtype=torch.int8),
    )


def build_graph_cache(rows: pd.DataFrame, cache_dir: Path) -> dict:
    """Build resumable v5-compatible OGB graph shards."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    progress_path = cache_dir / "manifest.json"
    existing = sorted(cache_dir.glob("graph_shard_*.pt"))
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if (
            progress.get("status") == "complete"
            and progress.get("source_rows") == len(rows)
            and progress.get("shards") == [path.name for path in existing]
        ):
            for path in existing:
                if sha256_file(path) != progress["shard_sha256"][path.name]:
                    raise RuntimeError(f"graph shard hash changed: {path.name}")
            return progress
    start = len(existing) * GRAPH_SHARD_ROWS
    if start > len(rows):
        raise RuntimeError("graph cache contains too many shards")
    progress = {
        "status": "building",
        "source_rows": len(rows),
        "processed_rows": start,
        "accepted_rows": 0,
        "invalid_rows": 0,
        "shards": [path.name for path in existing],
    }
    for path in existing:
        progress["accepted_rows"] += len(
            torch.load(path, map_location="cpu", weights_only=False)
        )
    progress["invalid_rows"] = start - progress["accepted_rows"]
    for begin in range(start, len(rows), GRAPH_SHARD_ROWS):
        started = time.perf_counter()
        end = min(begin + GRAPH_SHARD_ROWS, len(rows))
        graphs = []
        invalid = []
        for row in rows.iloc[begin:end].itertuples(index=False):
            graph = _build_graph(row)
            if graph is None:
                invalid.append(int(row.idx))
            else:
                graphs.append(graph)
        path = cache_dir / f"graph_shard_{begin // GRAPH_SHARD_ROWS:03d}.pt"
        atomic_torch(path, graphs)
        progress["processed_rows"] = end
        progress["accepted_rows"] += len(graphs)
        progress["invalid_rows"] += len(invalid)
        progress["shards"].append(path.name)
        progress.setdefault("invalid_source_idx", []).extend(invalid)
        atomic_json(progress_path, progress)
        print(
            f"graph shard {path.stem}: {len(graphs)}/{end - begin} "
            f"in {time.perf_counter() - started:.1f}s",
            flush=True,
        )
    progress["status"] = "complete"
    progress["shard_sha256"] = {
        path.name: sha256_file(path)
        for path in sorted(cache_dir.glob("graph_shard_*.pt"))
    }
    atomic_json(progress_path, progress)
    return progress


class PackedGraphDataset(InMemoryDataset):
    """Load one tensor-packed graph shard without Python-object unpickling."""

    def __init__(self, path: Path):
        super().__init__(root=None)
        self.data, self.slices = torch.load(
            path, map_location="cpu", weights_only=False
        )


def _save_packed_graphs(path: Path, graphs: list[Data]) -> None:
    data, slices = InMemoryDataset.collate(graphs)
    atomic_torch(path, (data, slices))


def build_packed_graph_cache(rows: pd.DataFrame, cache_dir: Path) -> dict:
    """Build resumable split-specific graph shards for large local training."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / "manifest.json"
    if manifest_path.exists():
        progress = json.loads(manifest_path.read_text(encoding="utf-8"))
        if progress.get("source_rows") != len(rows):
            raise RuntimeError("packed graph cache source row count changed")
        for shard in progress.get("source_shards", []):
            for filename, expected_hash in shard["files"].items():
                path = cache_dir / filename
                if not path.exists() or sha256_file(path) != expected_hash:
                    raise RuntimeError(f"packed graph shard changed: {filename}")
        if progress.get("status") == "complete":
            return progress
    else:
        unexpected = list(cache_dir.glob("*_shard_*.pt"))
        if unexpected:
            raise RuntimeError("packed graph shards exist without a manifest")
        progress = {
            "format": "molgap-pcqm-packed-graph-cache-v1",
            "status": "building",
            "source_rows": len(rows),
            "processed_rows": 0,
            "accepted_rows": 0,
            "invalid_rows": 0,
            "invalid_source_idx": [],
            "split_counts": {"train": 0, "dev": 0, "official": 0},
            "source_shards": [],
        }

    start = int(progress["processed_rows"])
    expected_shard = start // GRAPH_SHARD_ROWS
    if len(progress["source_shards"]) != expected_shard:
        raise RuntimeError("packed graph cache resume boundary changed")
    split_names = {0: "train", 1: "dev", 2: "official"}
    for begin in range(start, len(rows), GRAPH_SHARD_ROWS):
        started = time.perf_counter()
        end = min(begin + GRAPH_SHARD_ROWS, len(rows))
        by_split = {"train": [], "dev": [], "official": []}
        invalid = []
        for row in rows.iloc[begin:end].itertuples(index=False):
            graph = _build_graph(row)
            if graph is None:
                invalid.append(int(row.idx))
                continue
            by_split[split_names[int(graph.split_code.item())]].append(graph)

        shard_id = begin // GRAPH_SHARD_ROWS
        files = {}
        counts = {}
        for split_name, graphs in by_split.items():
            counts[split_name] = len(graphs)
            if not graphs:
                continue
            path = cache_dir / f"{split_name}_shard_{shard_id:03d}.pt"
            _save_packed_graphs(path, graphs)
            files[path.name] = sha256_file(path)

        progress["processed_rows"] = end
        progress["accepted_rows"] += sum(counts.values())
        progress["invalid_rows"] += len(invalid)
        progress["invalid_source_idx"].extend(invalid)
        for split_name, count in counts.items():
            progress["split_counts"][split_name] += count
        progress["source_shards"].append(
            {
                "source_begin": begin,
                "source_end": end,
                "counts": counts,
                "files": files,
            }
        )
        atomic_json(manifest_path, progress)
        print(
            f"packed graph shard {shard_id:03d}: "
            f"train={counts['train']} dev={counts['dev']} "
            f"official={counts['official']} invalid={len(invalid)} "
            f"in {time.perf_counter() - started:.1f}s",
            flush=True,
        )

    if progress["accepted_rows"] + progress["invalid_rows"] != len(rows):
        raise RuntimeError("packed graph cache rows do not reconcile")
    if progress["split_counts"]["official"] != OFFICIAL_VALID_ROWS:
        raise RuntimeError("packed graph cache official-valid count changed")
    progress["status"] = "complete"
    atomic_json(manifest_path, progress)
    return progress


def load_graph_splits(cache_dir: Path) -> tuple[list, list, list]:
    train = []
    dev = []
    official = []
    for path in sorted(cache_dir.glob("graph_shard_*.pt")):
        for graph in torch.load(path, map_location="cpu", weights_only=False):
            code = int(graph.split_code.item())
            if code == 0:
                train.append(graph)
            elif code == 1:
                dev.append(graph)
            elif code == 2:
                official.append(graph)
            else:
                raise RuntimeError(f"unknown PCQM split code {code}")
    counts = {
        "train": len(train),
        "dev": len(dev),
        "official": len(official),
    }
    if counts != EXPECTED_SPLIT_COUNTS:
        raise RuntimeError(f"PCQM graph split counts differ: {counts}")
    return train, dev, official


def scan_graph_split_counts(cache_dir: Path) -> dict[str, int]:
    counts = {"train": 0, "dev": 0, "official": 0}
    names = {0: "train", 1: "dev", 2: "official"}
    for path in sorted(cache_dir.glob("graph_shard_*.pt")):
        graphs = torch.load(path, map_location="cpu", weights_only=False)
        for graph in graphs:
            counts[names[int(graph.split_code.item())]] += 1
        del graphs
        gc.collect()
    return counts


def scan_packed_split_counts(cache_dir: Path) -> dict[str, int]:
    counts = {"train": 0, "dev": 0, "official": 0}
    for split_name in counts:
        for path in sorted(cache_dir.glob(f"{split_name}_shard_*.pt")):
            counts[split_name] += len(PackedGraphDataset(path))
    return counts


def accept_pcqm_expert_artifacts(
    output_dir: Path,
    cache_dir: Path,
) -> dict:
    """Independently validate a completed local PCQM expert run."""
    completion = json.loads(
        (output_dir / "completion_manifest.json").read_text(encoding="utf-8")
    )
    metrics = json.loads(
        (output_dir / "metrics.json").read_text(encoding="utf-8")
    )
    graph_manifest = json.loads(
        (cache_dir / "manifest.json").read_text(encoding="utf-8")
    )
    checks = {
        "completion_status": completion.get("status") == "complete",
        "graph_cache_status": graph_manifest.get("status") == "complete",
        "graph_rows_reconcile": (
            graph_manifest.get("accepted_rows", 0)
            + graph_manifest.get("invalid_rows", 0)
            == graph_manifest.get("source_rows")
        ),
        "split_counts_match": (
            scan_packed_split_counts(cache_dir) == metrics["split_counts"]
        ),
        "official_test_unused": metrics.get("official_test_used") is False,
        "sealed_20k_unused": metrics.get("sealed_20k_used") is False,
        "production_registry_unchanged": (
            metrics.get("production_registry_changed") is False
        ),
    }
    for filename, identity in completion["artifacts"].items():
        path = output_dir / filename
        checks[f"artifact_{filename}"] = (
            path.exists()
            and path.stat().st_size == identity["bytes"]
            and sha256_file(path) == identity["sha256"]
        )

    predictions = pd.read_csv(
        output_dir / "pcqm_official_valid_5k_predictions.csv"
    )
    numeric = predictions[
        ["gap_true_eV", "gap_prediction_eV", "absolute_error_eV"]
    ].to_numpy(dtype=np.float64)
    recomputed_errors = np.abs(numeric[:, 1] - numeric[:, 0])
    recomputed_mae = float(recomputed_errors.mean())
    checks.update(
        {
            "prediction_rows": len(predictions) == OFFICIAL_VALID_ROWS,
            "prediction_idx_unique": predictions["idx"].is_unique,
            "prediction_idx_hash": (
                array_sha256(predictions["idx"].to_numpy(dtype=np.int64))
                == EXPECTED_VALID_SHA256
            ),
            "predictions_finite": bool(np.isfinite(numeric).all()),
            "absolute_errors_match": bool(
                np.allclose(numeric[:, 2], recomputed_errors, atol=2e-9)
            ),
            "prediction_mae_matches": abs(
                recomputed_mae - metrics["official_valid_5k_gap_mae_eV"]
            )
            < 5e-9,
        }
    )

    checkpoint = torch.load(
        output_dir / "pcqm_gine_best.pt",
        map_location="cpu",
        weights_only=False,
    )
    model = PCQMGINEExpert()
    model.load_state_dict(checkpoint["model"], strict=True)
    checks["checkpoint_config"] = checkpoint.get("model_config") == MODEL_CONFIG
    checks["checkpoint_parameters_finite"] = all(
        torch.isfinite(parameter).all().item()
        for parameter in model.parameters()
    )
    report = {
        "status": "accepted" if all(checks.values()) else "rejected",
        "checks": checks,
        "recomputed_official_valid_5k_gap_mae_eV": recomputed_mae,
        "best_checkpoint_sha256": sha256_file(
            output_dir / "pcqm_gine_best.pt"
        ),
    }
    atomic_json(output_dir / "acceptance.json", report)
    if report["status"] != "accepted":
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"PCQM expert acceptance failed: {failed}")
    return report


class OGBGINConv(MessagePassing):
    def __init__(self, hidden_channels: int):
        super().__init__(aggr="add")
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, 2 * hidden_channels),
            nn.BatchNorm1d(2 * hidden_channels),
            nn.ReLU(),
            nn.Linear(2 * hidden_channels, hidden_channels),
        )
        self.eps = nn.Parameter(torch.zeros(1))
        self.bond_encoder = BondEncoder(hidden_channels)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        edge_embedding = self.bond_encoder(edge_attr)
        aggregated = self.propagate(
            edge_index,
            x=x,
            edge_attr=edge_embedding,
        )
        return self.mlp((1 + self.eps) * x + aggregated)

    def message(self, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        return torch.relu(x_j + edge_attr)


class PCQMGINEExpert(nn.Module):
    """Checkpoint-compatible virtual-node GINE used by accepted PCQM v5."""

    def __init__(
        self,
        hidden_channels: int = 256,
        num_layers: int = 5,
        dropout: float = 0.10,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.atom_encoder = AtomEncoder(hidden_channels)
        self.convs = nn.ModuleList(
            [OGBGINConv(hidden_channels) for _ in range(num_layers)]
        )
        self.batch_norms = nn.ModuleList(
            [nn.BatchNorm1d(hidden_channels) for _ in range(num_layers)]
        )
        self.virtual_mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_channels, 2 * hidden_channels),
                    nn.BatchNorm1d(2 * hidden_channels),
                    nn.ReLU(),
                    nn.Linear(2 * hidden_channels, hidden_channels),
                    nn.ReLU(),
                )
                for _ in range(num_layers - 1)
            ]
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
        )

    def forward(self, batch) -> torch.Tensor:
        h = self.atom_encoder(batch.x)
        virtual = h.new_zeros((int(batch.num_graphs), self.hidden_channels))
        for layer, (conv, norm) in enumerate(
            zip(self.convs, self.batch_norms)
        ):
            h = conv(h + virtual[batch.batch], batch.edge_index, batch.edge_attr)
            h = norm(h)
            if layer != self.num_layers - 1:
                h = torch.relu(h)
            h = nn.functional.dropout(
                h,
                p=self.dropout,
                training=self.training,
            )
            if layer < self.num_layers - 1:
                pooled = global_add_pool(h, batch.batch) + virtual
                virtual = virtual + nn.functional.dropout(
                    self.virtual_mlps[layer](pooled),
                    p=self.dropout,
                    training=self.training,
                )
        return self.head(global_mean_pool(h, batch.batch)).view(-1)


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, list[dict]]:
    model.eval()
    errors = 0.0
    count = 0
    rows = []
    for batch in loader:
        batch = batch.to(device)
        with torch.autocast("cuda", enabled=device.type == "cuda"):
            prediction = model(batch)
        target = batch.y.view(-1)
        errors += torch.abs(prediction - target).sum().item()
        count += int(target.numel())
        rows.extend(
            {
                "idx": int(index),
                "gap_true_eV": f"{float(truth):.9f}",
                "gap_prediction_eV": f"{float(predicted):.9f}",
                "absolute_error_eV": f"{abs(float(predicted) - float(truth)):.9f}",
            }
            for index, truth, predicted in zip(
                batch.sample_idx.view(-1).cpu(),
                target.float().cpu(),
                prediction.float().cpu(),
            )
        )
    return errors / count, rows


def _make_loaders(
    train: list,
    dev: list,
    official: list,
    batch_size: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    return (
        DataLoader(
            train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        ),
        DataLoader(
            dev,
            batch_size=batch_size * 2,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        ),
        DataLoader(
            official,
            batch_size=batch_size * 2,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        ),
    )


def continue_training(
    train: list,
    dev: list,
    official: list,
    *,
    resume_last: Path,
    resume_best: Path,
    resume_log: Path,
    output_dir: Path,
    max_epoch: int = 100,
    patience: int = 12,
    batch_size: int = 512,
) -> dict:
    """Resume accepted v5 optimizer state and stop on frozen scaffold dev."""
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PCQMGINEExpert().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
        min_lr=1e-5,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    checkpoint = torch.load(resume_last, map_location=device, weights_only=False)
    accepted_best = torch.load(resume_best, map_location="cpu", weights_only=False)
    if accepted_best.get("model_config") != MODEL_CONFIG:
        raise RuntimeError("accepted PCQM best checkpoint config changed")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    scaler.load_state_dict(checkpoint["scaler"])
    start_epoch = int(checkpoint["epoch"]) + 1
    best_mae = float(checkpoint["best_dev_mae_eV"])
    best_epoch = int(checkpoint["best_epoch"])
    stale_epochs = int(checkpoint.get("stale_epochs", 0))
    if start_epoch != 70 or best_epoch != 68:
        raise RuntimeError(
            f"expected v5 epoch 69/best 68, got start={start_epoch} best={best_epoch}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "pcqm_gine_best.pt"
    last_path = output_dir / "pcqm_gine_last.pt"
    log_path = output_dir / "pcqm_gine_train_log.csv"
    if not best_path.exists():
        shutil.copy2(resume_best, best_path)
    history = []
    with resume_log.open(newline="", encoding="utf-8") as handle:
        history = list(csv.DictReader(handle))
    train_loader, dev_loader, official_loader = _make_loaders(
        train, dev, official, batch_size
    )

    baseline_official_mae, _ = evaluate(model, official_loader, device)
    model.load_state_dict(accepted_best["model"])
    accepted_best_official_mae, _ = evaluate(model, official_loader, device)
    model.load_state_dict(checkpoint["model"])
    if abs(accepted_best_official_mae - 0.18732001037597656) > 0.002:
        raise RuntimeError(
            "accepted v5 official-valid MAE did not reproduce locally: "
            f"{accepted_best_official_mae:.6f}"
        )
    baseline = {
        "resume_epoch": int(checkpoint["epoch"]),
        "resume_best_epoch": best_epoch,
        "resume_best_dev_mae_eV": best_mae,
        "resume_last_official_valid_mae_eV": baseline_official_mae,
        "accepted_best_official_valid_mae_eV": accepted_best_official_mae,
        "accepted_best_reference_mae_eV": 0.18732001037597656,
        "device": str(device),
    }
    atomic_json(output_dir / "baseline_reproduction.json", baseline)
    print(json.dumps(baseline, indent=2), flush=True)

    for epoch in range(start_epoch, max_epoch):
        started = time.perf_counter()
        model.train()
        train_error = 0.0
        train_count = 0
        for batch in train_loader:
            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=device.type == "cuda"):
                prediction = model(batch)
                target = batch.y.view(-1)
                loss = nn.functional.l1_loss(prediction, target)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            train_error += torch.abs(prediction.detach() - target).sum().item()
            train_count += int(target.numel())
        train_mae = train_error / train_count
        dev_mae, _ = evaluate(model, dev_loader, device)
        scheduler.step(dev_mae)
        improved = dev_mae < best_mae
        if improved:
            best_mae = dev_mae
            best_epoch = epoch
            stale_epochs = 0
            atomic_torch(
                best_path,
                {
                    "model": model.state_dict(),
                    "model_config": MODEL_CONFIG,
                    "epoch": epoch,
                    "dev_mae_eV": dev_mae,
                    "seed": SEED,
                },
            )
        else:
            stale_epochs += 1
        row = {
            "epoch": epoch,
            "train_mae_eV": f"{train_mae:.9f}",
            "dev_mae_eV": f"{dev_mae:.9f}",
            "best_dev_mae_eV": f"{best_mae:.9f}",
            "learning_rate": f"{optimizer.param_groups[0]['lr']:.9g}",
            "elapsed_seconds": f"{time.perf_counter() - started:.3f}",
        }
        history.append(row)
        atomic_csv(log_path, history)
        atomic_torch(
            last_path,
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "best_dev_mae_eV": best_mae,
                "best_epoch": best_epoch,
                "stale_epochs": stale_epochs,
                "seed": SEED,
            },
        )
        atomic_json(
            output_dir / "progress.json",
            {
                "status": "training",
                "epoch": epoch,
                "best_epoch": best_epoch,
                "best_dev_mae_eV": best_mae,
                "stale_epochs": stale_epochs,
                "last_row": row,
            },
        )
        print(
            f"ep{epoch:03d} train={train_mae:.5f} dev={dev_mae:.5f} "
            f"best={best_mae:.5f}@{best_epoch} "
            f"lr={optimizer.param_groups[0]['lr']:.2e} "
            f"{row['elapsed_seconds']}s{' *' if improved else ''}",
            flush=True,
        )
        if stale_epochs >= patience:
            print(f"early stop after {stale_epochs} stale epochs", flush=True)
            break

    best = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    official_mae, prediction_rows = evaluate(model, official_loader, device)
    atomic_csv(output_dir / "pcqm_official_valid_5k_predictions.csv", prediction_rows)
    metrics = {
        "experiment": "pcqm_gine_local_continuation_v6",
        "train_rows": len(train),
        "scaffold_dev_rows": len(dev),
        "official_valid_rows": len(official),
        "best_epoch": int(best["epoch"]),
        "best_scaffold_dev_mae_eV": float(best["dev_mae_eV"]),
        "official_valid_5k_gap_mae_eV": official_mae,
        "delta_vs_accepted_v5_eV": official_mae - 0.18732001037597656,
        "device": str(device),
        "official_test_used": False,
        "sealed_20k_used": False,
        "production_registry_changed": False,
    }
    atomic_json(output_dir / "metrics.json", metrics)
    atomic_json(
        output_dir / "completion_manifest.json",
        {
            "status": "complete",
            "metrics": metrics,
            "artifacts": {
                path.name: {
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for path in (
                    best_path,
                    last_path,
                    log_path,
                    output_dir / "pcqm_official_valid_5k_predictions.csv",
                    output_dir / "metrics.json",
                )
            },
        },
    )
    return metrics


def _evaluate_packed_split(
    model: nn.Module,
    cache_dir: Path,
    split_name: str,
    device: torch.device,
    batch_size: int,
    *,
    keep_rows: bool = False,
) -> tuple[float, list[dict]]:
    total_error = 0.0
    total_count = 0
    all_rows = []
    for path in sorted(cache_dir.glob(f"{split_name}_shard_*.pt")):
        dataset = PackedGraphDataset(path)
        if len(dataset):
            mae, rows = evaluate(
                model,
                DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=0,
                ),
                device,
            )
            total_error += mae * len(dataset)
            total_count += len(dataset)
            if keep_rows:
                all_rows.extend(rows)
        del dataset
    if not total_count:
        raise RuntimeError(f"packed split {split_name} is empty")
    return total_error / total_count, all_rows


def train_expanded_streaming(
    *,
    cache_dir: Path,
    initial_best: Path,
    output_dir: Path,
    epochs: int = 20,
    patience: int = 6,
    batch_size: int = 512,
    learning_rate: float = 5e-5,
    freeze_batch_norm_stats: bool = True,
) -> dict:
    """Fine-tune the v6 expert on a larger sample using one graph shard at a time."""
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(initial_best, map_location="cpu", weights_only=False)
    if checkpoint.get("model_config") != MODEL_CONFIG:
        raise RuntimeError("expanded PCQM initialization config changed")
    model = PCQMGINEExpert().to(device)
    model.load_state_dict(checkpoint["model"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
        min_lr=1e-5,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "pcqm_gine_best.pt"
    last_path = output_dir / "pcqm_gine_last.pt"
    log_path = output_dir / "pcqm_gine_train_log.csv"
    shard_paths = sorted(cache_dir.glob("train_shard_*.pt"))
    if not shard_paths:
        raise RuntimeError("expanded PCQM graph cache is empty")
    split_counts = scan_packed_split_counts(cache_dir)
    baseline_dev, _ = _evaluate_packed_split(
        model, cache_dir, "dev", device, batch_size * 2
    )
    best_mae = baseline_dev
    best_epoch = -1
    stale_epochs = 0
    history = []
    atomic_torch(
        best_path,
        {
            "model": model.state_dict(),
            "model_config": MODEL_CONFIG,
            "epoch": -1,
            "dev_mae_eV": baseline_dev,
            "seed": SEED,
            "initial_checkpoint": str(initial_best),
        },
    )
    atomic_json(
        output_dir / "baseline.json",
        {
            "expanded_split_counts": split_counts,
            "initial_checkpoint": str(initial_best),
            "initial_dev_mae_eV": baseline_dev,
            "official_valid_read": False,
            "device": str(device),
            "freeze_batch_norm_stats": freeze_batch_norm_stats,
        },
    )
    print(
        f"expanded baseline dev={baseline_dev:.6f} counts={split_counts}",
        flush=True,
    )
    generator = random.Random(SEED)
    for epoch in range(epochs):
        started = time.perf_counter()
        model.train()
        if freeze_batch_norm_stats:
            # Source-ordered shards bias running statistics toward the final shard.
            for module in model.modules():
                if isinstance(module, nn.BatchNorm1d):
                    module.eval()
        train_error = 0.0
        train_count = 0
        epoch_shards = list(shard_paths)
        generator.shuffle(epoch_shards)
        for path in epoch_shards:
            dataset = PackedGraphDataset(path)
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
            )
            for batch in loader:
                batch = batch.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", enabled=device.type == "cuda"):
                    prediction = model(batch)
                    target = batch.y.view(-1)
                    loss = nn.functional.l1_loss(prediction, target)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
                train_error += torch.abs(
                    prediction.detach() - target
                ).sum().item()
                train_count += int(target.numel())
            del loader, dataset
        dev_mae, _ = _evaluate_packed_split(
            model, cache_dir, "dev", device, batch_size * 2
        )
        scheduler.step(dev_mae)
        improved = dev_mae < best_mae
        if improved:
            best_mae = dev_mae
            best_epoch = epoch
            stale_epochs = 0
            atomic_torch(
                best_path,
                {
                    "model": model.state_dict(),
                    "model_config": MODEL_CONFIG,
                    "epoch": epoch,
                    "dev_mae_eV": dev_mae,
                    "seed": SEED,
                    "initial_checkpoint": str(initial_best),
                },
            )
        else:
            stale_epochs += 1
        row = {
            "epoch": epoch,
            "train_mae_eV": f"{train_error / train_count:.9f}",
            "dev_mae_eV": f"{dev_mae:.9f}",
            "best_dev_mae_eV": f"{best_mae:.9f}",
            "learning_rate": f"{optimizer.param_groups[0]['lr']:.9g}",
            "elapsed_seconds": f"{time.perf_counter() - started:.3f}",
        }
        history.append(row)
        atomic_csv(log_path, history)
        atomic_torch(
            last_path,
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "best_dev_mae_eV": best_mae,
                "best_epoch": best_epoch,
                "stale_epochs": stale_epochs,
                "seed": SEED,
            },
        )
        atomic_json(
            output_dir / "progress.json",
            {
                "status": "training",
                "best_epoch": best_epoch,
                "best_dev_mae_eV": best_mae,
                "stale_epochs": stale_epochs,
                "last_row": row,
            },
        )
        print(
            f"scale ep{epoch:02d} train={train_error / train_count:.5f} "
            f"dev={dev_mae:.5f} best={best_mae:.5f}@{best_epoch} "
            f"lr={optimizer.param_groups[0]['lr']:.2e} "
            f"{row['elapsed_seconds']}s{' *' if improved else ''}",
            flush=True,
        )
        if stale_epochs >= patience:
            break

    best = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    official_mae, official_rows = _evaluate_packed_split(
        model,
        cache_dir,
        "official",
        device,
        batch_size * 2,
        keep_rows=True,
    )
    prediction_path = output_dir / "pcqm_official_valid_5k_predictions.csv"
    atomic_csv(prediction_path, official_rows)
    metrics = {
        "experiment": "pcqm_gine_local_scaleup_1m_v7",
        "split_counts": split_counts,
        "best_epoch": int(best["epoch"]),
        "best_scaffold_dev_mae_eV": float(best["dev_mae_eV"]),
        "official_valid_5k_gap_mae_eV": official_mae,
        "delta_vs_v6_250k_eV": official_mae - 0.18527247314453124,
        "device": str(device),
        "official_test_used": False,
        "sealed_20k_used": False,
        "production_registry_changed": False,
        "freeze_batch_norm_stats": freeze_batch_norm_stats,
    }
    atomic_json(output_dir / "metrics.json", metrics)
    atomic_json(
        output_dir / "completion_manifest.json",
        {
            "status": "complete",
            "metrics": metrics,
            "artifacts": {
                path.name: {
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for path in (
                    best_path,
                    last_path,
                    log_path,
                    prediction_path,
                    output_dir / "metrics.json",
                )
            },
        },
    )
    return metrics


def run_local_scaleup(
    *,
    raw_csv: Path,
    accepted_valid_predictions: Path,
    initial_best: Path,
    cache_dir: Path,
    output_dir: Path,
    total_train_rows: int = 1_000_000,
    epochs: int = 20,
    patience: int = 6,
    batch_size: int = 512,
    learning_rate: float = 1e-5,
    freeze_batch_norm_stats: bool = True,
) -> dict:
    rows, contract = load_expanded_rows(
        raw_csv,
        accepted_valid_predictions,
        total_train_rows=total_train_rows,
    )
    atomic_json(output_dir / "input_contract.json", contract)
    graph_manifest = build_packed_graph_cache(rows, cache_dir)
    atomic_json(output_dir / "graph_acceptance.json", graph_manifest)
    return train_expanded_streaming(
        cache_dir=cache_dir,
        initial_best=initial_best,
        output_dir=output_dir,
        epochs=epochs,
        patience=patience,
        batch_size=batch_size,
        learning_rate=learning_rate,
        freeze_batch_norm_stats=freeze_batch_norm_stats,
    )


def run_local_continuation(
    *,
    raw_csv: Path,
    accepted_valid_predictions: Path,
    resume_last: Path,
    resume_best: Path,
    resume_log: Path,
    cache_dir: Path,
    output_dir: Path,
    max_epoch: int = 100,
    patience: int = 12,
    batch_size: int = 512,
) -> dict:
    rows, contract = load_selected_rows(raw_csv, accepted_valid_predictions)
    atomic_json(output_dir / "input_contract.json", contract)
    graph_manifest = build_graph_cache(rows, cache_dir)
    atomic_json(output_dir / "graph_acceptance.json", graph_manifest)
    train, dev, official = load_graph_splits(cache_dir)
    return continue_training(
        train,
        dev,
        official,
        resume_last=resume_last,
        resume_best=resume_best,
        resume_log=resume_log,
        output_dir=output_dir,
        max_epoch=max_epoch,
        patience=patience,
        batch_size=batch_size,
    )
