"""PubChemQC scaffold-screen SchNet training with one or two ETKDG views."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from .schnet import SchNetWrapper

TARGETS = ("HOMO", "LUMO", "Gap")
MODEL_CONFIG = {
    "hidden_channels": 176,
    "num_filters": 160,
    "num_interactions": 6,
    "num_gaussians": 50,
    "cutoff": 10.0,
    "dropout": 0.05,
}


def atomic_json(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_torch_save(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_split(path: Path) -> dict[int, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not {"source_idx", "split"}.issubset(rows[0]):
        raise ValueError("split CSV needs source_idx and split columns")
    result = {int(row["source_idx"]): row["split"].strip().lower() for row in rows}
    if len(result) != len(rows):
        raise ValueError("split CSV contains duplicate source_idx values")
    if set(result.values()) - {"train", "validation", "test"}:
        raise ValueError("split CSV contains unsupported roles")
    return result


def source_map(graphs, accepted: set[int] | None = None) -> dict[int, object]:
    result = {}
    for graph in graphs:
        index = int(graph.source_idx.view(-1)[0])
        if accepted is not None and index not in accepted:
            continue
        if index in result:
            raise ValueError(f"duplicate graph source_idx {index}")
        result[index] = graph
    return result


def align_graph_views(primary, secondary, split: dict[int, str]) -> dict[str, tuple[list, list]]:
    accepted = set(split)
    first = source_map(primary, accepted)
    second = source_map(secondary, accepted)
    common = sorted(set(first).intersection(second))
    if not common:
        raise ValueError("primary and secondary graph caches have no aligned rows")
    roles: dict[str, tuple[list, list]] = {}
    for role in ("train", "validation", "test"):
        indices = [index for index in common if split[index] == role]
        left = [first[index] for index in indices]
        right = [second[index] for index in indices]
        for a, b in zip(left, right):
            if not torch.allclose(a.y.view(-1), b.y.view(-1), atol=1e-6, rtol=0.0):
                raise ValueError(
                    f"target mismatch for source_idx {int(a.source_idx.view(-1)[0])}"
                )
        roles[role] = (left, right)
    if any(not pair[0] for pair in roles.values()):
        raise ValueError("one or more aligned split roles are empty")
    return roles


def _forward(model, batch):
    charges = batch.charges if hasattr(batch, "charges") else None
    return model(batch.z, batch.pos, batch.batch, charges=charges)


def _encode(model, batch):
    charges = batch.charges if hasattr(batch, "charges") else None
    return model.encode(batch.z, batch.pos, batch.batch, charges=charges)


def _metrics(predictions: torch.Tensor, targets: torch.Tensor) -> dict:
    error = (predictions - targets).abs()
    result = {
        target: {"mae_eV": float(error[:, index].mean())}
        for index, target in enumerate(TARGETS)
    }
    result["average"] = {"mae_eV": float(error.mean())}
    return result


@torch.no_grad()
def evaluate_view(model, graphs, batch_size, device, mean, std) -> dict:
    model.eval()
    predictions = []
    embeddings = []
    targets = []
    source_indices = []
    for batch in DataLoader(graphs, batch_size=batch_size, shuffle=False, num_workers=0):
        batch = batch.to(device)
        embedding = _encode(model, batch)
        prediction = model.head(embedding) * std + mean
        predictions.append(prediction.float().cpu())
        embeddings.append(embedding.float().cpu())
        targets.append(batch.y.view(-1, 3).float().cpu())
        source_indices.append(batch.source_idx.view(-1).long().cpu())
    return {
        "predictions": torch.cat(predictions),
        "embeddings": torch.cat(embeddings),
        "targets": torch.cat(targets),
        "source_idx": torch.cat(source_indices),
    }


def average_views(first: dict, second: dict) -> dict:
    if not torch.equal(first["source_idx"], second["source_idx"]):
        raise ValueError("conformer source_idx alignment failed")
    if not torch.allclose(first["targets"], second["targets"], atol=1e-6, rtol=0.0):
        raise ValueError("conformer target alignment failed")
    return {
        "predictions": 0.5 * (first["predictions"] + second["predictions"]),
        "embeddings": 0.5 * (first["embeddings"] + second["embeddings"]),
        "targets": first["targets"],
        "source_idx": first["source_idx"],
    }


def train(
    *,
    variant: str,
    primary_graphs: list,
    secondary_graphs: list,
    split_csv: Path,
    output_dir: Path,
    epochs: int = 30,
    patience: int = 8,
    batch_size: int = 128,
    learning_rate: float = 4e-4,
    weight_decay: float = 1e-5,
    seed: int = 42,
) -> dict:
    if variant not in {"primary", "augmented"}:
        raise ValueError("variant must be primary or augmented")
    set_seed(seed)
    split = read_split(split_csv)
    roles = align_graph_views(primary_graphs, secondary_graphs, split)
    train_targets = torch.cat([graph.y.view(1, 3) for graph in roles["train"][0]])
    mean = train_targets.mean(dim=0)
    std = train_targets.std(dim=0).clamp_min(1e-6)

    training_graphs = list(roles["train"][0])
    if variant == "augmented":
        training_graphs = [
            graph
            for pair in zip(roles["train"][0], roles["train"][1])
            for graph in pair
        ]
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        training_graphs,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SchNetWrapper(**MODEL_CONFIG, use_charges=True).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1), eta_min=1e-6
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    criterion = nn.L1Loss()
    mean_device = mean.to(device)
    std_device = std.to(device)

    output_dir.mkdir(parents=True, exist_ok=True)
    last_path = output_dir / "last.pt"
    best_path = output_dir / "best.pt"
    start_epoch = 0
    best_epoch = -1
    best_mae = float("inf")
    best_state = None
    wait = 0
    log = []
    if last_path.exists():
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        if checkpoint["variant"] != variant or checkpoint["model_config"] != MODEL_CONFIG:
            raise RuntimeError("resume checkpoint contract mismatch")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        best_state = checkpoint["best_state"]
        best_epoch = int(checkpoint["best_epoch"])
        best_mae = float(checkpoint["best_mae"])
        wait = int(checkpoint["wait"])
        log = list(checkpoint["log"])
        start_epoch = int(checkpoint["epoch"]) + 1

    for epoch in range(start_epoch, epochs):
        started = time.perf_counter()
        model.train()
        total = 0.0
        count = 0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                target = (batch.y.view(-1, 3) - mean_device) / std_device
                loss = criterion(_forward(model, batch), target)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach()) * batch.num_graphs
            count += batch.num_graphs
        scheduler.step()
        first = evaluate_view(
            model, roles["validation"][0], batch_size, device, mean_device, std_device
        )
        if variant == "augmented":
            second = evaluate_view(
                model,
                roles["validation"][1],
                batch_size,
                device,
                mean_device,
                std_device,
            )
            selected_payload = average_views(first, second)
        else:
            selected_payload = first
        value = _metrics(
            selected_payload["predictions"], selected_payload["targets"]
        )["average"]["mae_eV"]
        improved = value < best_mae
        if improved:
            best_mae = value
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
        row = {
            "epoch": epoch,
            "train_normalized_l1": total / max(count, 1),
            "validation_average_mae_eV": value,
            "elapsed_s": time.perf_counter() - started,
            "selected": improved,
        }
        log.append(row)
        checkpoint = {
            "variant": variant,
            "model_config": MODEL_CONFIG,
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_state": best_state,
            "best_epoch": best_epoch,
            "best_mae": best_mae,
            "wait": wait,
            "log": log,
            "target_mean": mean,
            "target_std": std,
        }
        atomic_torch_save(checkpoint, last_path)
        atomic_json(
            {
                "status": "running",
                "variant": variant,
                "epoch": epoch,
                "best_epoch": best_epoch,
                "best_validation_average_mae_eV": best_mae,
            },
            output_dir / "progress.json",
        )
        print(
            f"{variant} ep{epoch:02d} train={row['train_normalized_l1']:.5f} "
            f"val={value:.5f}eV {row['elapsed_s']:.1f}s"
            f"{' *' if improved else ''}",
            flush=True,
        )
        if wait >= patience:
            break

    if best_state is None:
        raise RuntimeError("training produced no finite best checkpoint")
    model.load_state_dict(best_state)
    atomic_torch_save(
        {
            "variant": variant,
            "model_config": MODEL_CONFIG,
            "model": best_state,
            "target_mean": mean,
            "target_std": std,
            "best_epoch": best_epoch,
            "best_validation_average_mae_eV": best_mae,
        },
        best_path,
    )

    payloads = {}
    metrics = {}
    for role, (primary, secondary) in roles.items():
        first = evaluate_view(
            model, primary, batch_size, device, mean_device, std_device
        )
        if variant == "augmented":
            second = evaluate_view(
                model, secondary, batch_size, device, mean_device, std_device
            )
            selected = average_views(first, second)
            metrics[role] = {
                "primary": _metrics(first["predictions"], first["targets"]),
                "secondary": _metrics(second["predictions"], second["targets"]),
                "average": _metrics(
                    selected["predictions"], selected["targets"]
                ),
            }
        else:
            selected = first
            metrics[role] = {
                "primary": _metrics(first["predictions"], first["targets"])
            }
        payloads[role] = selected
    atomic_torch_save(payloads, output_dir / "embeddings.pt")
    result = {
        "experiment": "pubchemqc100k_light_schnet",
        "variant": variant,
        "model_config": MODEL_CONFIG,
        "n_params": sum(parameter.numel() for parameter in model.parameters()),
        "aligned_rows": {
            role: len(pair[0]) for role, pair in roles.items()
        },
        "training_graphs_per_epoch": len(training_graphs),
        "best_epoch": best_epoch,
        "best_validation_average_mae_eV": best_mae,
        "metrics": metrics,
        "log": log,
        "artifacts": {
            "best": str(best_path),
            "last": str(last_path),
            "embeddings": str(output_dir / "embeddings.pt"),
        },
    }
    atomic_json(result, output_dir / "metrics.json")
    atomic_json(
        {
            "status": "complete",
            "variant": variant,
            "best_epoch": best_epoch,
            "best_validation_average_mae_eV": best_mae,
            "artifacts": {
                path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
                for path in (best_path, last_path, output_dir / "embeddings.pt")
            },
        },
        output_dir / "completion_manifest.json",
    )
    return result
