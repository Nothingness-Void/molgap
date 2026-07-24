"""Reproducible QM9 architecture-screen data and training utilities."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from .egnn import EGNNWrapper
from .gine import GINEWrapper
from .gps import GPSWrapper
from .graphs import smiles_to_pyg
from .schnet import SchNetWrapper
from .tensornet import TensorNetWrapper

QM9_PROCESSED_URL = "https://data.pyg.org/datasets/qm9_v3.zip"
QM9_RAW_URL = (
    "https://deepchemdata.s3-us-west-1.amazonaws.com/"
    "datasets/molnet_publish/qm9.zip"
)
TARGET_NAMES = ("HOMO", "LUMO", "Gap")
TARGET_COLUMNS = (2, 3, 4)
DEFAULT_CACHE = Path("data/cache/qm9")
DEFAULT_RESULTS = Path("results/phase8/experiments/qm9_architecture_screen")
DEFAULT_MODELS = Path("models/experiments/qm9_architecture_screen")

ENCODER_CONFIGS = {
    "gine6": {
        "kind": "topology",
        "hidden_channels": 192,
        "num_layers": 6,
        "dropout": 0.05,
        "batch_size": 256,
    },
    "gps7": {
        "kind": "topology",
        "hidden_channels": 192,
        "num_layers": 7,
        "num_heads": 4,
        "dropout": 0.05,
        "batch_size": 256,
    },
    "gps9": {
        "kind": "topology",
        "hidden_channels": 192,
        "num_layers": 9,
        "num_heads": 4,
        "dropout": 0.05,
        "batch_size": 192,
    },
    "gps9_160": {
        "kind": "topology",
        "hidden_channels": 160,
        "num_layers": 9,
        "num_heads": 4,
        "dropout": 0.05,
        "batch_size": 192,
    },
    "gps9_128": {
        "kind": "topology",
        "hidden_channels": 128,
        "num_layers": 9,
        "num_heads": 4,
        "dropout": 0.05,
        "batch_size": 192,
    },
    "gps9_meanmax": {
        "kind": "topology",
        "hidden_channels": 192,
        "num_layers": 9,
        "num_heads": 4,
        "dropout": 0.05,
        "pooling": "mean_max",
        "batch_size": 192,
    },
    "gps11_160": {
        "kind": "topology",
        "hidden_channels": 160,
        "num_layers": 11,
        "num_heads": 4,
        "dropout": 0.05,
        "batch_size": 192,
    },
    "schnet": {
        "kind": "geometry",
        "hidden_channels": 176,
        "num_filters": 160,
        "num_interactions": 6,
        "num_gaussians": 50,
        "cutoff": 10.0,
        "dropout": 0.05,
        "batch_size": 128,
    },
    "tensornet": {
        "kind": "geometry",
        "hidden_channels": 128,
        "num_layers": 2,
        "num_rbf": 32,
        "cutoff": 5.0,
        "dropout": 0.0,
        "batch_size": 32,
    },
    "egnn": {
        "kind": "geometry",
        "hidden_channels": 128,
        "num_layers": 4,
        "num_rbf": 32,
        "cutoff": 5.0,
        "dropout": 0.05,
        "batch_size": 128,
    },
}


@dataclass(frozen=True)
class ScreenSplit:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    seed: int

    @property
    def all_indices(self) -> np.ndarray:
        return np.concatenate((self.train, self.validation, self.test))

    @property
    def fingerprint(self) -> str:
        value = self.all_indices.astype(np.int64).tobytes()
        return hashlib.sha256(value).hexdigest()[:16]


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fixed_split(
    n_total: int,
    train_size: int,
    validation_size: int,
    test_size: int,
    seed: int,
) -> ScreenSplit:
    requested = train_size + validation_size + test_size
    if requested > n_total:
        raise ValueError(f"Requested {requested} rows from QM9 with {n_total} rows")
    order = np.random.RandomState(seed).permutation(n_total)[:requested]
    train_end = train_size
    validation_end = train_end + validation_size
    return ScreenSplit(
        train=order[:train_end],
        validation=order[train_end:validation_end],
        test=order[validation_end:],
        seed=seed,
    )


def _download(url: str, destination: Path) -> None:
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    urllib.request.urlretrieve(url, temporary)
    os.replace(temporary, destination)


def prepare_qm9_files(cache_dir: Path = DEFAULT_CACHE) -> dict[str, Path]:
    processed = cache_dir / "preprocessed" / "qm9_v3.pt"
    raw_sdf = cache_dir / "raw" / "gdb9.sdf"
    if not processed.exists():
        archive = cache_dir / "download" / "qm9_v3.zip"
        _download(QM9_PROCESSED_URL, archive)
        processed.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as source:
            source.extractall(processed.parent)
    if not raw_sdf.exists():
        archive = cache_dir / "download" / "qm9_raw.zip"
        _download(QM9_RAW_URL, archive)
        raw_sdf.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as source:
            source.extractall(raw_sdf.parent)
    return {"processed": processed, "raw_sdf": raw_sdf}


def load_qm9_records(cache_dir: Path = DEFAULT_CACHE) -> list[dict]:
    paths = prepare_qm9_files(cache_dir)
    records = torch.load(paths["processed"], map_location="cpu", weights_only=False)
    if not isinstance(records, list) or not records:
        raise ValueError(f"Unexpected QM9 payload: {paths['processed']}")
    return records


def target_tensor(record: dict) -> torch.Tensor:
    return record["y"].view(-1)[list(TARGET_COLUMNS)].float()


def target_stats(records: list[dict], train_indices: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    targets = torch.stack([target_tensor(records[int(i)]) for i in train_indices])
    mean = targets.mean(dim=0)
    std = targets.std(dim=0).clamp_min(1e-6)
    return mean, std


def _topology_graph(record: dict, source_idx: int, mean: torch.Tensor, std: torch.Tensor) -> Data:
    target = target_tensor(record)
    return Data(
        x=record["x"].float(),
        z=record["z"].long(),
        edge_index=record["edge_index"].long(),
        edge_attr=record["edge_attr"].float(),
        y=((target - mean) / std).view(1, -1),
        y_eV=target.view(1, -1),
        source_idx=torch.tensor([source_idx], dtype=torch.long),
    )


def _dft_graph(record: dict, source_idx: int, mean: torch.Tensor, std: torch.Tensor) -> Data:
    target = target_tensor(record)
    return Data(
        z=record["z"].long(),
        pos=record["pos"].float(),
        y=((target - mean) / std).view(1, -1),
        y_eV=target.view(1, -1),
        source_idx=torch.tensor([source_idx], dtype=torch.long),
    )


def build_etkdg_cache(
    records: list[dict],
    indices: np.ndarray,
    mean: torch.Tensor,
    std: torch.Tensor,
    *,
    cache_dir: Path = DEFAULT_CACHE,
    seed: int = 42,
    mmff_iters: int = 200,
    shard_size: int = 2000,
) -> tuple[dict[int, Data], dict]:
    paths = prepare_qm9_files(cache_dir)
    protocol = f"qm9-etkdg-v3-sanitize-false-mmff{mmff_iters}".encode()
    cache_identity = protocol + b"\0" + indices.astype(np.int64).tobytes()
    key = hashlib.sha256(cache_identity).hexdigest()[:16]
    output = cache_dir / "etkdg" / f"graphs_{key}_seed{seed}.pt"
    report_path = output.with_suffix(".json")
    if output.exists() and report_path.exists():
        payload = torch.load(output, map_location="cpu", weights_only=False)
        return payload, json.loads(report_path.read_text(encoding="utf-8"))

    from rdkit import Chem, RDLogger

    output.parent.mkdir(parents=True, exist_ok=True)
    shard_dir = output.parent / "shards" / f"{key}_seed{seed}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    RDLogger.DisableLog("rdApp.*")
    supplier = Chem.SDMolSupplier(
        str(paths["raw_sdf"]), removeHs=False, sanitize=False
    )
    graphs: dict[int, Data] = {}
    failures: list[int] = []
    started = time.perf_counter()
    resumed_shards = 0
    index_list = indices.tolist()
    for start in range(0, len(index_list), shard_size):
        stop = min(start + shard_size, len(index_list))
        shard_path = shard_dir / f"{start:06d}_{stop:06d}.pt"
        if shard_path.exists():
            shard = torch.load(shard_path, map_location="cpu", weights_only=False)
            shard_graphs = shard["graphs"]
            shard_failures = shard["failure_indices"]
            resumed_shards += 1
        else:
            shard_graphs: dict[int, Data] = {}
            shard_failures: list[int] = []
            for source_idx in index_list[start:stop]:
                record = records[source_idx]
                raw_idx = int(str(record["name"]).split("_")[-1]) - 1
                mol = supplier[raw_idx]
                if mol is None:
                    shard_failures.append(source_idx)
                    continue
                try:
                    no_hydrogen = Chem.RemoveHs(mol, sanitize=False)
                    smiles = Chem.MolToSmiles(
                        no_hydrogen, canonical=True, isomericSmiles=True
                    )
                    if Chem.MolFromSmiles(smiles) is None:
                        shard_failures.append(source_idx)
                        continue
                except Exception:
                    shard_failures.append(source_idx)
                    continue
                graph = smiles_to_pyg(
                    smiles,
                    use_charges=False,
                    mmff_iters=mmff_iters,
                    random_seed=(seed * 1_000_003 + source_idx) % 2_147_483_647,
                )
                if graph is None:
                    shard_failures.append(source_idx)
                    continue
                target = target_tensor(record)
                graph.y = ((target - mean) / std).view(1, -1)
                graph.y_eV = target.view(1, -1)
                graph.source_idx = torch.tensor([source_idx], dtype=torch.long)
                shard_graphs[source_idx] = graph
            _atomic_torch_save(
                shard_path,
                {
                    "graphs": shard_graphs,
                    "failure_indices": shard_failures,
                    "start": start,
                    "stop": stop,
                },
            )
        graphs.update(shard_graphs)
        failures.extend(shard_failures)
        print(
            f"ETKDG {stop}/{len(indices)} success={len(graphs)} "
            f"elapsed={time.perf_counter() - started:.0f}s",
            flush=True,
        )

    report = {
        "requested": int(len(indices)),
        "succeeded": len(graphs),
        "failed": len(failures),
        "failure_indices": failures,
        "seed": seed,
        "mmff_iters": mmff_iters,
        "shard_size": shard_size,
        "resumed_shards": resumed_shards,
        "sdf_sanitize": False,
        "cache_version": 3,
        "elapsed_s": time.perf_counter() - started,
        "index_sha256": hashlib.sha256(indices.astype(np.int64).tobytes()).hexdigest(),
    }
    _atomic_torch_save(output, graphs)
    _atomic_json(report_path, report)
    return graphs, report


def make_graph_splits(
    records: list[dict],
    split: ScreenSplit,
    geometry: str,
    mean: torch.Tensor,
    std: torch.Tensor,
    cache_dir: Path,
    seed: int,
) -> tuple[dict[str, list[Data]], dict]:
    roles = {
        "train": split.train,
        "validation": split.validation,
        "test": split.test,
    }
    if geometry == "topology":
        return {
            role: [_topology_graph(records[int(i)], int(i), mean, std) for i in indices]
            for role, indices in roles.items()
        }, {"geometry": "topology", "failed": 0}
    if geometry == "dft":
        return {
            role: [_dft_graph(records[int(i)], int(i), mean, std) for i in indices]
            for role, indices in roles.items()
        }, {"geometry": "dft", "failed": 0}
    if geometry != "etkdg":
        raise ValueError(f"Unsupported geometry: {geometry}")
    graphs, report = build_etkdg_cache(
        records, split.all_indices, mean, std, cache_dir=cache_dir, seed=seed
    )
    return {
        role: [graphs[int(i)] for i in indices if int(i) in graphs]
        for role, indices in roles.items()
    }, {"geometry": "etkdg", **report}


def make_encoder(candidate: str, in_channels: int = 11, edge_dim: int = 4):
    config = dict(ENCODER_CONFIGS[candidate])
    config.pop("batch_size")
    kind = config.pop("kind")
    if candidate == "gine6":
        return GINEWrapper(in_channels=in_channels, edge_dim=edge_dim, **config), kind
    if candidate.startswith("gps"):
        return GPSWrapper(in_channels=in_channels, edge_dim=edge_dim, **config), kind
    if candidate == "schnet":
        return SchNetWrapper(**config, use_charges=False), kind
    if candidate == "tensornet":
        return TensorNetWrapper(**config, use_charges=False), kind
    if candidate == "egnn":
        return EGNNWrapper(**config), kind
    raise ValueError(candidate)


def _forward(kind: str, model, batch):
    if kind == "topology":
        return model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
    return model(batch.z, batch.pos, batch.batch)


def _encode(kind: str, model, batch):
    if kind == "topology":
        return model.encode(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
    return model.encode(batch.z, batch.pos, batch.batch)


def _metrics(prediction: np.ndarray, target: np.ndarray) -> dict:
    errors = np.abs(prediction - target)
    result = {
        name: {"mae": float(errors[:, i].mean())}
        for i, name in enumerate(TARGET_NAMES)
    }
    result["average"] = {"mae": float(errors.mean())}
    return result


@torch.no_grad()
def evaluate_encoder(kind, model, graphs, batch_size, device, mean, std):
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    predictions, targets, embeddings, source_indices = [], [], [], []
    for batch in loader:
        batch = batch.to(device)
        embedding = _encode(kind, model, batch)
        normalized = model.head(embedding)
        predictions.append((normalized * std.to(device) + mean.to(device)).float().cpu())
        targets.append(batch.y_eV.view(-1, 3).float().cpu())
        embeddings.append(embedding.float().cpu())
        source_indices.append(batch.source_idx.view(-1).cpu())
    return {
        "predictions": torch.cat(predictions),
        "targets": torch.cat(targets),
        "embeddings": torch.cat(embeddings),
        "source_idx": torch.cat(source_indices),
    }


def train_encoder(
    *,
    candidate: str,
    geometry: str,
    train_size: int,
    validation_size: int,
    test_size: int,
    epochs: int,
    seed: int = 42,
    split_seed: int = 42,
    learning_rate: float = 4e-4,
    weight_decay: float = 1e-5,
    patience: int = 8,
    resume: bool = False,
    cache_dir: Path = DEFAULT_CACHE,
    results_dir: Path = DEFAULT_RESULTS,
    models_dir: Path = DEFAULT_MODELS,
) -> dict:
    expected = ENCODER_CONFIGS[candidate]["kind"]
    if expected == "topology" and geometry != "topology":
        raise ValueError(f"{candidate} requires --geometry topology")
    if expected == "geometry" and geometry not in {"dft", "etkdg"}:
        raise ValueError(f"{candidate} requires --geometry dft or etkdg")
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = load_qm9_records(cache_dir)
    split = fixed_split(len(records), train_size, validation_size, test_size, split_seed)
    mean, std = target_stats(records, split.train)
    graph_splits, geometry_report = make_graph_splits(
        records, split, geometry, mean, std, cache_dir, seed
    )
    model, kind = make_encoder(candidate)
    model = model.to(device)
    batch_size = int(ENCODER_CONFIGS[candidate]["batch_size"])
    train_loader = DataLoader(
        graph_splits["train"], batch_size=batch_size, shuffle=True, num_workers=0
    )
    validation_loader = DataLoader(
        graph_splits["validation"], batch_size=batch_size, shuffle=False, num_workers=0
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1), eta_min=1e-6
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    criterion = nn.L1Loss()
    run_name = (
        f"n{train_size}_{validation_size}_{test_size}/"
        f"{candidate}_{geometry}/seed{seed}"
    )
    result_path = results_dir / run_name / "metrics.json"
    embedding_path = cache_dir / "embeddings" / run_name / "payload.pt"
    model_path = models_dir / run_name / "model.pt"
    checkpoint_path = models_dir / run_name / "checkpoint.pt"
    best_mae = float("inf")
    best_state = None
    best_epoch = -1
    wait = 0
    log = []
    start_epoch = 0
    if resume and checkpoint_path.exists():
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        best_state = checkpoint["best_state"]
        best_mae = float(checkpoint["best_mae"])
        best_epoch = int(checkpoint["best_epoch"])
        wait = int(checkpoint["wait"])
        log = list(checkpoint["log"])
        start_epoch = int(checkpoint["epoch"]) + 1
    for epoch in range(start_epoch, epochs):
        started = time.perf_counter()
        model.train()
        total = 0.0
        count = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                loss = criterion(_forward(kind, model, batch), batch.y.view(-1, 3))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach()) * batch.num_graphs
            count += batch.num_graphs
        scheduler.step()
        validation = evaluate_encoder(
            kind, model, graph_splits["validation"], batch_size, device, mean, std
        )
        metrics = _metrics(
            validation["predictions"].numpy(), validation["targets"].numpy()
        )
        val_mae = metrics["average"]["mae"]
        improved = val_mae < best_mae
        if improved:
            best_mae = val_mae
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            wait = 0
        else:
            wait += 1
        row = {
            "epoch": epoch,
            "train_normalized_l1": total / max(count, 1),
            "validation_average_mae_eV": val_mae,
            "elapsed_s": time.perf_counter() - started,
            "selected": improved,
        }
        log.append(row)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_tmp = checkpoint_path.with_suffix(".tmp")
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_state": best_state,
            "best_mae": best_mae,
            "best_epoch": best_epoch,
            "wait": wait,
            "log": log,
        }, checkpoint_tmp)
        os.replace(checkpoint_tmp, checkpoint_path)
        print(
            f"{candidate}/{geometry} ep{epoch:02d} "
            f"train={row['train_normalized_l1']:.5f} "
            f"val={val_mae:.5f}eV {row['elapsed_s']:.1f}s"
            f"{' *' if improved else ''}",
            flush=True,
        )
        if wait >= patience:
            break
    if best_state is None:
        raise RuntimeError("Training produced no checkpoint")
    model.load_state_dict(best_state)

    role_payloads = {}
    role_metrics = {}
    for role, graphs in graph_splits.items():
        payload = evaluate_encoder(kind, model, graphs, batch_size, device, mean, std)
        role_payloads[role] = payload
        role_metrics[role] = _metrics(
            payload["predictions"].numpy(), payload["targets"].numpy()
        )

    embedding_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(role_payloads, embedding_path)
    torch.save(best_state, model_path)
    result = {
        "experiment": "qm9_architecture_screen",
        "candidate": candidate,
        "geometry": geometry,
        "seed": seed,
        "split_seed": split_seed,
        "split_fingerprint": split.fingerprint,
        "split_rows": {role: len(graphs) for role, graphs in graph_splits.items()},
        "requested_rows": {
            "train": train_size,
            "validation": validation_size,
            "test": test_size,
        },
        "target_names": list(TARGET_NAMES),
        "target_units": "eV",
        "target_mean": mean.tolist(),
        "target_std": std.tolist(),
        "model_config": ENCODER_CONFIGS[candidate],
        "n_params": sum(parameter.numel() for parameter in model.parameters()),
        "best_epoch": best_epoch,
        "best_validation_average_mae_eV": best_mae,
        "metrics": role_metrics,
        "geometry_report": geometry_report,
        "log": log,
        "artifacts": {
            "embeddings": str(embedding_path),
            "model": str(model_path),
            "checkpoint": str(checkpoint_path),
        },
    }
    _atomic_json(result_path, result)
    return result


def evaluate_encoder_on_geometry(
    *,
    candidate: str,
    geometry: str,
    checkpoint: Path,
    output: Path,
    embedding_output: Path,
    train_size: int,
    validation_size: int,
    test_size: int,
    split_seed: int,
    geometry_seed: int,
    cache_dir: Path = DEFAULT_CACHE,
) -> dict:
    if ENCODER_CONFIGS[candidate]["kind"] != "geometry":
        raise ValueError(f"{candidate} is not a geometry encoder")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = load_qm9_records(cache_dir)
    split = fixed_split(
        len(records), train_size, validation_size, test_size, split_seed
    )
    mean, std = target_stats(records, split.train)
    graph_splits, geometry_report = make_graph_splits(
        records,
        split,
        geometry,
        mean,
        std,
        cache_dir,
        geometry_seed,
    )
    model, kind = make_encoder(candidate)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model = model.to(device)
    batch_size = int(ENCODER_CONFIGS[candidate]["batch_size"])
    role_payloads = {
        role: evaluate_encoder(
            kind, model, graphs, batch_size, device, mean, std
        )
        for role, graphs in graph_splits.items()
    }
    role_metrics = {
        role: _metrics(
            payload["predictions"].numpy(), payload["targets"].numpy()
        )
        for role, payload in role_payloads.items()
    }
    _atomic_torch_save(embedding_output, role_payloads)
    result = {
        "experiment": "qm9_geometry_transfer_eval",
        "candidate": candidate,
        "geometry": geometry,
        "geometry_seed": geometry_seed,
        "split_seed": split_seed,
        "split_fingerprint": split.fingerprint,
        "checkpoint": str(checkpoint),
        "embedding_output": str(embedding_output),
        "split_rows": {
            role: len(graphs) for role, graphs in graph_splits.items()
        },
        "metrics": role_metrics,
        "geometry_report": geometry_report,
    }
    _atomic_json(output, result)
    return result


@torch.no_grad()
def export_gps_multiscale_embeddings(
    *,
    checkpoint: Path,
    output: Path,
    embedding_output: Path,
    train_size: int,
    validation_size: int,
    test_size: int,
    split_seed: int,
    layers: tuple[int, ...] = (2, 4, -1),
    cache_dir: Path = DEFAULT_CACHE,
) -> dict:
    """Export several GPS9 layers without adding another encoder forward pass."""
    if -1 not in layers:
        raise ValueError("GPS multiscale export requires -1 for final predictions")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = load_qm9_records(cache_dir)
    split = fixed_split(
        len(records), train_size, validation_size, test_size, split_seed
    )
    mean, std = target_stats(records, split.train)
    graph_splits, _ = make_graph_splits(
        records, split, "topology", mean, std, cache_dir, seed=split_seed
    )
    model, _ = make_encoder("gps9")
    model.load_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=True)
    )
    model = model.to(device)
    model.eval()
    batch_size = int(ENCODER_CONFIGS["gps9"]["batch_size"])
    hidden = int(ENCODER_CONFIGS["gps9"]["hidden_channels"])
    role_payloads = {}
    for role, graphs in graph_splits.items():
        loader = DataLoader(
            graphs, batch_size=batch_size, shuffle=False, num_workers=0
        )
        predictions, targets, embeddings, source_indices = [], [], [], []
        for batch in loader:
            batch = batch.to(device)
            embedding = model.encode_layers(
                batch.x,
                batch.edge_index,
                batch.edge_attr,
                batch.batch,
                layers=layers,
            )
            normalized = model.head(embedding[:, -hidden:])
            predictions.append(
                (normalized * std.to(device) + mean.to(device)).float().cpu()
            )
            targets.append(batch.y_eV.view(-1, 3).float().cpu())
            embeddings.append(embedding.float().cpu())
            source_indices.append(batch.source_idx.view(-1).cpu())
        role_payloads[role] = {
            "predictions": torch.cat(predictions),
            "targets": torch.cat(targets),
            "embeddings": torch.cat(embeddings),
            "source_idx": torch.cat(source_indices),
        }
    _atomic_torch_save(embedding_output, role_payloads)
    result = {
        "experiment": "qm9_gps_multiscale_export",
        "candidate": "gps9",
        "layers": list(layers),
        "split_seed": split_seed,
        "split_fingerprint": split.fingerprint,
        "checkpoint": str(checkpoint),
        "embedding_output": str(embedding_output),
        "embedding_dim": int(role_payloads["train"]["embeddings"].shape[1]),
        "split_rows": {
            role: len(payload["source_idx"])
            for role, payload in role_payloads.items()
        },
        "metrics": {
            role: _metrics(
                payload["predictions"].numpy(), payload["targets"].numpy()
            )
            for role, payload in role_payloads.items()
        },
    }
    _atomic_json(output, result)
    return result


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _atomic_torch_save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)
