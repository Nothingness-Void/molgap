"""Streaming PCQM training for the validated TGT/topology hybrid.

The PCQM cache contains a GPS 2D topology view and a primary ETKDG view with
different node counts.  This adapter keeps those views separate and passes
their explicit batch metadata to :class:`TGTLiteHybridV2Wrapper`.
"""
from __future__ import annotations

import copy
import gc
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from .pcqm_tgt_lite import (
    PackedGraphDataset,
    _shards,
    atomic_json,
    atomic_torch,
    set_seed,
    sha256_file,
    target_stats,
    verify_graph_manifest,
)
from .tgt_hybrid_v2 import TGTLiteHybridV2Wrapper


@dataclass(frozen=True)
class PCQMHybridV3Config:
    name: str = "tgt_hybrid_v3_pcqm"
    hidden_channels: int = 192
    pair_channels: int = 48
    num_layers: int = 8
    num_heads: int = 4
    num_rbf: int = 32
    cutoff: float = 12.0
    topology_layers: int = 6
    dropout: float = 0.05
    batch_size: int = 16
    learning_rate: float = 2.0e-4
    weight_decay: float = 1.0e-5
    max_epochs: int = 30
    patience: int = 6
    seed: int = 42
    training_budget_s: float = 9.0 * 3600.0


def _model(config: PCQMHybridV3Config) -> TGTLiteHybridV2Wrapper:
    return TGTLiteHybridV2Wrapper(
        in_channels=18,
        edge_dim=4,
        hidden_channels=config.hidden_channels,
        pair_channels=config.pair_channels,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        num_rbf=config.num_rbf,
        cutoff=config.cutoff,
        dropout=config.dropout,
        topology_layers=config.topology_layers,
        n_targets=1,
    )


def _view_kwargs(batch) -> dict[str, torch.Tensor]:
    names = (
        "topology_x",
        "topology_edges",
        "topology_edge_attr",
        "topology_node_count",
        "topology_edge_count",
        "geometry_node_count",
    )
    values = {name: getattr(batch, name, None) for name in names}
    missing = [name for name, value in values.items() if value is None]
    if missing:
        raise RuntimeError(
            "PCQM hybrid cache is missing explicit view fields: "
            + ", ".join(missing)
        )
    return values


def _forward(model: TGTLiteHybridV2Wrapper, batch) -> torch.Tensor:
    return model(
        batch.x,
        batch.edge_index,
        batch.edge_attr,
        batch.z,
        batch.pos,
        batch.batch,
        **_view_kwargs(batch),
    ).view(-1)


def _encode(model: TGTLiteHybridV2Wrapper, batch) -> torch.Tensor:
    return model.encode(
        batch.x,
        batch.edge_index,
        batch.edge_attr,
        batch.z,
        batch.pos,
        batch.batch,
        **_view_kwargs(batch),
    )


@torch.no_grad()
def evaluate_dev(
    model: TGTLiteHybridV2Wrapper,
    root: Path,
    device: torch.device,
    batch_size: int,
    mean: float,
    std: float,
) -> float:
    model.eval()
    absolute = 0.0
    count = 0
    for path in _shards(root, "dev"):
        dataset = PackedGraphDataset(path)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        for batch in loader:
            batch = batch.to(device, non_blocking=True)
            prediction = _forward(model, batch) * std + mean
            target = batch.y.view(-1).float()
            absolute += float((prediction - target).abs().sum())
            count += int(target.numel())
        del loader, dataset
        gc.collect()
    return absolute / max(count, 1)


@torch.no_grad()
def export_embeddings(
    model: TGTLiteHybridV2Wrapper,
    root: Path,
    output_dir: Path,
    device: torch.device,
    batch_size: int,
    mean: float,
    std: float,
) -> dict:
    model.eval()
    manifest = {
        "format": "molgap-pcqm-tgt-hybrid-v3-embedding-parts-v1",
        "status": "exporting",
        "embedding_dim": model.fusion[1].out_features,
        "parts": [],
    }
    atomic_json(output_dir / "manifest.json", manifest)
    for role in ("train", "dev"):
        for part_index, path in enumerate(_shards(root, role)):
            dataset = PackedGraphDataset(path)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            embeddings, predictions, targets, indices = [], [], [], []
            for batch in loader:
                batch = batch.to(device, non_blocking=True)
                embedding = _encode(model, batch)
                prediction = model.head(embedding).view(-1) * std + mean
                embeddings.append(embedding.float().cpu())
                predictions.append(prediction.float().cpu())
                targets.append(batch.y.view(-1).float().cpu())
                indices.append(batch.source_idx.view(-1).long().cpu())
            payload = {
                "role": role,
                "source_idx": torch.cat(indices),
                "embedding": torch.cat(embeddings),
                "prediction_eV": torch.cat(predictions),
                "target_eV": torch.cat(targets),
                "source_shard": path.name,
            }
            part_path = output_dir / role / f"part_{part_index:04d}.pt"
            atomic_torch(part_path, payload)
            manifest["parts"].append(
                {
                    "role": role,
                    "source_shard": path.name,
                    "path": part_path.relative_to(output_dir).as_posix(),
                    "rows": int(payload["source_idx"].numel()),
                    "sha256": sha256_file(part_path),
                }
            )
            atomic_json(output_dir / "manifest.json", manifest)
            del loader, dataset
            gc.collect()
    manifest["status"] = "complete"
    atomic_json(output_dir / "manifest.json", manifest)
    return manifest


def train_tgt_hybrid_v3(
    *,
    root: Path,
    output_dir: Path,
    config: PCQMHybridV3Config = PCQMHybridV3Config(),
) -> dict:
    started = time.monotonic()
    set_seed(config.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("PCQM TGT-hybrid v3 training requires a CUDA GPU")
    device = torch.device("cuda")
    input_report = verify_graph_manifest(root)
    mean, std = target_stats(root)
    model = _model(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.max_epochs, eta_min=1.0e-6
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    best_state = None
    best_mae = float("inf")
    best_epoch = -1
    wait = 0
    start_epoch = 0
    log = []
    last_path = output_dir / "last.pt"
    if last_path.exists():
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        if checkpoint.get("config") != asdict(config):
            raise RuntimeError("TGT-hybrid v3 resume config changed")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        best_state = checkpoint["best_state"]
        best_mae = float(checkpoint["best_dev_gap_mae_eV"])
        best_epoch = int(checkpoint["best_epoch"])
        wait = int(checkpoint["wait"])
        start_epoch = int(checkpoint["epoch"]) + 1
        log = list(checkpoint["log"])

    criterion = nn.L1Loss()
    for epoch in range(start_epoch, config.max_epochs):
        if time.monotonic() - started + 1200.0 > config.training_budget_s:
            break
        model.train()
        total_loss = 0.0
        total_rows = 0
        epoch_started = time.monotonic()
        for path in _shards(root, "train"):
            dataset = PackedGraphDataset(path)
            loader = DataLoader(
                dataset,
                batch_size=config.batch_size,
                shuffle=True,
                pin_memory=True,
            )
            for batch in loader:
                batch = batch.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=True):
                    prediction = _forward(model, batch)
                    target = (batch.y.view(-1) - mean) / std
                    loss = criterion(prediction, target)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                total_loss += float(loss.detach()) * batch.num_graphs
                total_rows += batch.num_graphs
            del loader, dataset
            gc.collect()
        scheduler.step()
        dev_mae = evaluate_dev(model, root, device, config.batch_size, mean, std)
        improved = np.isfinite(dev_mae) and dev_mae < best_mae
        if improved:
            best_state = copy.deepcopy(model.state_dict())
            best_mae = dev_mae
            best_epoch = epoch
            wait = 0
        else:
            wait += 1
        row = {
            "epoch": epoch,
            "train_gap_l1_normalized": total_loss / max(total_rows, 1),
            "train_rows": total_rows,
            "dev_gap_mae_eV": dev_mae,
            "elapsed_s": time.monotonic() - epoch_started,
            "selected": bool(improved),
        }
        log.append(row)
        atomic_torch(
            last_path,
            {
                "format": "molgap-pcqm-tgt-hybrid-v3-checkpoint-v1",
                "config": asdict(config),
                "mean_gap": mean,
                "std_gap": std,
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "best_state": best_state,
                "best_dev_gap_mae_eV": best_mae,
                "best_epoch": best_epoch,
                "wait": wait,
                "log": log,
                "input_report": input_report,
            },
        )
        atomic_json(
            output_dir / "progress.json",
            {
                "status": "training",
                "epoch": epoch,
                "best_epoch": best_epoch,
                "best_dev_gap_mae_eV": best_mae,
                "elapsed_s": time.monotonic() - started,
            },
        )
        print(
            f"{config.name} ep{epoch:02d} "
            f"train={row['train_gap_l1_normalized']:.6f} "
            f"dev={dev_mae:.6f}eV {row['elapsed_s']:.1f}s"
            f"{' *' if improved else ''}",
            flush=True,
        )
        if wait >= config.patience:
            break
    if best_state is None:
        raise RuntimeError("TGT-hybrid v3 PCQM training produced no finite checkpoint")
    model.load_state_dict(best_state, strict=True)
    atomic_torch(
        output_dir / "best.pt",
        {
            "format": "molgap-pcqm-tgt-hybrid-v3-best-v1",
            "config": asdict(config),
            "model": best_state,
            "mean_gap": mean,
            "std_gap": std,
            "best_epoch": best_epoch,
            "best_dev_gap_mae_eV": best_mae,
            "input_report": input_report,
        },
    )
    embedding_manifest = export_embeddings(
        model,
        root,
        output_dir / "embeddings",
        device,
        config.batch_size,
        mean,
        std,
    )
    metrics = {
        "status": "complete",
        "config": asdict(config),
        "best_epoch": best_epoch,
        "best_dev_gap_mae_eV": best_mae,
        "train_log": log,
        "target_mean_gap": mean,
        "target_std_gap": std,
        "input_report": input_report,
        "embedding_manifest": embedding_manifest,
        "official_valid_metric_read": False,
        "official_test_used": False,
        "sealed_20k_used": False,
        "production_registry_changed": False,
        "runtime_s": time.monotonic() - started,
    }
    atomic_json(output_dir / "metrics.json", metrics)
    artifacts = {
        path.relative_to(output_dir).as_posix(): {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(output_dir.rglob("*"))
        if path.is_file() and path.name != "completion_manifest.json"
    }
    atomic_json(
        output_dir / "completion_manifest.json",
        {
            "status": "complete",
            "name": config.name,
            "artifacts": artifacts,
        },
    )
    return metrics
