"""Durable direct training for one PubChemQC 2D architecture candidate."""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, r2_score
from torch_geometric.loader import DataLoader

from .edge_state_gps import DynamicEdgeGPSWrapper

TARGETS = ("HOMO", "LUMO", "Gap")


def _atomic_json(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_torch_save(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _read_split(split_csv: Path, graphs: list) -> tuple[dict[str, list], dict]:
    with split_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not {"source_idx", "split"}.issubset(rows[0]):
        raise ValueError("split CSV needs source_idx and split columns")
    assignments = [(int(row["source_idx"]), row["split"].strip().lower()) for row in rows]
    indices = [source_idx for source_idx, _ in assignments]
    if len(indices) != len(set(indices)):
        raise ValueError("split CSV contains duplicate source_idx values")
    if set(role for _, role in assignments) - {"train", "validation", "test"}:
        raise ValueError("split CSV contains unsupported split roles")
    graph_map = {int(graph.source_idx.view(-1)[0]): graph for graph in graphs}
    if len(graph_map) != len(graphs):
        raise ValueError("graph cache contains duplicate source_idx values")
    missing = sorted(set(indices) - set(graph_map))
    if missing:
        raise ValueError(f"split CSV references {len(missing)} unavailable graphs")
    split_sets = {
        role: [graph_map[source_idx] for source_idx, assigned in assignments if assigned == role]
        for role in ("train", "validation", "test")
    }
    if any(not values for values in split_sets.values()):
        raise ValueError("split must contain train, validation, and test rows")
    return split_sets, {
        "path": str(split_csv),
        "sha256": _sha256(split_csv),
        "rows": {role: len(values) for role, values in split_sets.items()},
    }


def _metrics(prediction: np.ndarray, target: np.ndarray) -> dict:
    result = {}
    for index, name in enumerate(TARGETS):
        result[name] = {
            "mae": float(mean_absolute_error(target[:, index], prediction[:, index])),
            "r2": float(r2_score(target[:, index], prediction[:, index])),
        }
    result["average"] = {
        "mae": float(np.mean([result[name]["mae"] for name in TARGETS])),
        "r2": float(np.mean([result[name]["r2"] for name in TARGETS])),
    }
    return result


@torch.no_grad()
def _evaluate(model, graphs: list, device: torch.device, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False, num_workers=0)
    predictions, targets = [], []
    model.eval()
    for batch in loader:
        batch = batch.to(device)
        predictions.append(model(batch.x, batch.edge_index, batch.edge_attr, batch.batch).float().cpu().numpy())
        targets.append(batch.y.float().cpu().numpy())
    return np.concatenate(predictions), np.concatenate(targets)


@torch.no_grad()
def _extract_embeddings(model, graphs: list, device: torch.device, batch_size: int) -> dict:
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False, num_workers=0)
    embeddings, source_indices = [], []
    model.eval()
    for batch in loader:
        batch = batch.to(device)
        embeddings.append(
            model.encode(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            .float()
            .cpu()
        )
        source_indices.append(batch.source_idx.view(-1).long().cpu())
    return {"embeddings": torch.cat(embeddings), "source_idx": torch.cat(source_indices)}


def train(
    *,
    graphs_path: Path,
    split_csv: Path,
    output_dir: Path,
    model_config: dict,
    epochs: int = 40,
    patience: int = 10,
    learning_rate: float = 4e-4,
    weight_decay: float = 1e-5,
    batch_size: int = 256,
    seed: int = 42,
    split_seed: int = 42,
    resume_from: Path | None = None,
    write_embeddings: bool = True,
) -> dict:
    if seed != 42 or split_seed != 42:
        raise ValueError("this fixed architecture comparison requires seed=42 and split_seed=42")
    _set_seed(seed)
    graphs = torch.load(graphs_path, weights_only=False)
    graphs_sha256 = _sha256(graphs_path)
    split_sets, split_contract = _read_split(split_csv, graphs)
    training_params = {
        "lr": learning_rate,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "scheduler": "cosine",
        "loss": "L1",
        "gradient_clip": 10.0,
        "amp": bool(torch.cuda.is_available()),
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DynamicEdgeGPSWrapper(**model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1), eta_min=1e-6
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    criterion = nn.L1Loss()
    loader = DataLoader(split_sets["train"], batch_size=batch_size, shuffle=True, num_workers=0)
    validation_loader = DataLoader(
        split_sets["validation"], batch_size=batch_size, shuffle=False, num_workers=0
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "training_state.pt"
    metrics_path = output_dir / "metrics.json"
    model_path = output_dir / "model.pt"
    start_epoch = 0
    best_val = float("inf")
    best_epoch = -1
    best_state = None
    wait = 0
    log_rows = []
    if resume_from is not None:
        checkpoint = torch.load(resume_from, weights_only=False, map_location=device)
        expected = {
            "kind": "dynamic_edge_gps",
            "graphs_path": str(graphs_path),
            "graphs_sha256": graphs_sha256,
            "n_graphs": len(graphs),
            "seed": seed,
            "split_seed": split_seed,
            "split_contract": split_contract,
            "model_config": model_config,
            "training_params": training_params,
        }
        for key, value in expected.items():
            if checkpoint.get(key) != value:
                raise ValueError(f"resume checkpoint contract mismatch: {key}")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        scaler.load_state_dict(checkpoint["scaler_state"])
        start_epoch = int(checkpoint["next_epoch"])
        best_val = float(checkpoint["best_val"])
        best_epoch = int(checkpoint["best_epoch"])
        best_state = checkpoint["best_state"]
        wait = int(checkpoint["wait"])
        log_rows = list(checkpoint["log"])

    for epoch in range(start_epoch, epochs):
        started = time.time()
        model.train()
        train_total = 0.0
        train_count = 0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                prediction = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                loss = criterion(prediction, batch.y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            scaler.step(optimizer)
            scaler.update()
            train_total += float(loss.detach()) * batch.num_graphs
            train_count += batch.num_graphs

        model.eval()
        validation_total = 0.0
        validation_count = 0
        with torch.no_grad():
            for batch in validation_loader:
                batch = batch.to(device)
                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    loss = criterion(
                        model(batch.x, batch.edge_index, batch.edge_attr, batch.batch),
                        batch.y,
                    )
                validation_total += float(loss) * batch.num_graphs
                validation_count += batch.num_graphs
        val_mae = validation_total / max(validation_count, 1)
        scheduler.step()
        improved = val_mae < best_val
        if improved:
            best_val = val_mae
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        row = {
            "epoch": epoch,
            "train_loss": train_total / max(train_count, 1),
            "val_mae": val_mae,
            "lr": optimizer.param_groups[0]["lr"],
            "time_s": time.time() - started,
        }
        log_rows.append(row)
        checkpoint = {
            "kind": "dynamic_edge_gps",
            "graphs_path": str(graphs_path),
            "graphs_sha256": graphs_sha256,
            "n_graphs": len(graphs),
            "seed": seed,
            "split_seed": split_seed,
            "split_contract": split_contract,
            "model_config": model_config,
            "training_params": training_params,
            "next_epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "best_val": best_val,
            "best_epoch": best_epoch,
            "best_state": best_state,
            "wait": wait,
            "log": log_rows,
        }
        _atomic_torch_save(checkpoint, checkpoint_path)
        _atomic_json(
            {
                "complete": False,
                "kind": "dynamic_edge_gps",
                "next_epoch": epoch + 1,
                "best_val_mae": best_val,
                "best_epoch": best_epoch,
                "log": log_rows,
            },
            metrics_path,
        )
        print(
            f"ep{epoch:03d} train={row['train_loss']:.4f} val={val_mae:.4f} "
            f"best={best_val:.4f}@{best_epoch} lr={row['lr']:.2e} "
            f"{row['time_s']:.1f}s{' *' if improved else ''}",
            flush=True,
        )
        if wait >= patience:
            print(f"Early stop at epoch {epoch}", flush=True)
            break

    if best_state is None:
        raise RuntimeError("training produced no finite best state")
    model.load_state_dict(best_state)
    _atomic_torch_save(best_state, model_path)
    prediction, target = _evaluate(model, split_sets["test"], device, batch_size)
    result = {
        "kind": "dynamic_edge_gps",
        "graphs_path": str(graphs_path),
        "graphs_sha256": graphs_sha256,
        "n_graphs": len(graphs),
        "n_params": int(sum(parameter.numel() for parameter in model.parameters())),
        "seed": seed,
        "split_seed": split_seed,
        "split_contract": split_contract,
        "model_params": model_config,
        "params": training_params,
        "best_val_mae": best_val,
        "best_epoch": best_epoch,
        "training_time_s": float(sum(row["time_s"] for row in log_rows)),
        "test_metrics": _metrics(prediction, target),
        "log": log_rows,
    }
    # Publish the scientific result before the optional, larger embedding
    # artifact so a late export failure cannot hide a complete evaluation.
    _atomic_json(result, metrics_path)
    if write_embeddings:
        embedding_payload = _extract_embeddings(model, graphs, device, batch_size)
        _atomic_torch_save(embedding_payload, output_dir / "embeddings.pt")
        result["embedding_shape"] = list(embedding_payload["embeddings"].shape)
    _atomic_json(result, metrics_path)
    return result
