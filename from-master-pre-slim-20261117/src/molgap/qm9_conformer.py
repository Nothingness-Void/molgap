"""Conformer-robust QM9 training experiments."""
from __future__ import annotations

import copy
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from .qm9_screen import (
    DEFAULT_CACHE,
    DEFAULT_MODELS,
    DEFAULT_RESULTS,
    ENCODER_CONFIGS,
    _atomic_json,
    _atomic_torch_save,
    _forward,
    _metrics,
    evaluate_encoder,
    fixed_split,
    load_qm9_records,
    make_encoder,
    make_graph_splits,
    set_seed,
    target_stats,
)


def _source_map(graphs):
    return {int(graph.source_idx.view(-1)[0]): graph for graph in graphs}


def _paired_views(first, second):
    first_map = _source_map(first)
    second_map = _source_map(second)
    common = [index for index in first_map if index in second_map]
    return (
        [first_map[index] for index in common],
        [second_map[index] for index in common],
    )


@torch.no_grad()
def _evaluate_views(model, first, second, batch_size, device, mean, std):
    first_payload = evaluate_encoder(
        "geometry", model, first, batch_size, device, mean, std
    )
    second_payload = evaluate_encoder(
        "geometry", model, second, batch_size, device, mean, std
    )
    if not torch.equal(
        first_payload["source_idx"], second_payload["source_idx"]
    ):
        raise ValueError("Conformer views are not source-index aligned")
    if not torch.equal(first_payload["targets"], second_payload["targets"]):
        raise ValueError("Conformer views have mismatched targets")
    average = {
        "predictions": 0.5
        * (first_payload["predictions"] + second_payload["predictions"]),
        "embeddings": 0.5
        * (first_payload["embeddings"] + second_payload["embeddings"]),
        "targets": first_payload["targets"],
        "source_idx": first_payload["source_idx"],
    }
    return first_payload, second_payload, average


def train_schnet_conformer_augmented(
    *,
    train_size: int = 100_000,
    validation_size: int = 10_000,
    test_size: int = 10_000,
    epochs: int = 30,
    seed: int = 42,
    split_seed: int = 42,
    geometry_seeds: tuple[int, int] = (42, 43),
    learning_rate: float = 4e-4,
    weight_decay: float = 1e-5,
    patience: int = 8,
    resume: bool = False,
    cache_dir: Path = DEFAULT_CACHE,
    results_dir: Path = DEFAULT_RESULTS,
    models_dir: Path = DEFAULT_MODELS,
) -> dict:
    """Train one SchNet on two aligned ETKDG views per molecule."""
    if len(geometry_seeds) != 2:
        raise ValueError("Exactly two geometry seeds are required")
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = load_qm9_records(cache_dir)
    split = fixed_split(
        len(records), train_size, validation_size, test_size, split_seed
    )
    mean, std = target_stats(records, split.train)
    view_splits = []
    geometry_reports = []
    loaded_views = {}
    for geometry_seed in geometry_seeds:
        if geometry_seed not in loaded_views:
            loaded_views[geometry_seed] = make_graph_splits(
                records,
                split,
                "etkdg",
                mean,
                std,
                cache_dir,
                geometry_seed,
            )
        graphs, report = loaded_views[geometry_seed]
        view_splits.append(graphs)
        geometry_reports.append(report)
    paired = {
        role: _paired_views(view_splits[0][role], view_splits[1][role])
        for role in ("train", "validation", "test")
    }

    model, _ = make_encoder("schnet")
    model = model.to(device)
    config = ENCODER_CONFIGS["schnet"]
    batch_size = int(config["batch_size"])
    train_graphs = [
        graph
        for pair in zip(*paired["train"])
        for graph in pair
    ]
    train_loader = DataLoader(
        train_graphs,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1), eta_min=1e-6
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    criterion = nn.L1Loss()

    variant = "aug2" if geometry_seeds[0] != geometry_seeds[1] else "dup2"
    run_name = (
        f"n{train_size}_{validation_size}_{test_size}/"
        f"schnet_etkdg_{variant}/seed{seed}"
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
                loss = criterion(
                    _forward("geometry", model, batch),
                    batch.y.view(-1, 3),
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach()) * batch.num_graphs
            count += batch.num_graphs
        scheduler.step()
        _, _, validation = _evaluate_views(
            model,
            *paired["validation"],
            batch_size,
            device,
            mean,
            std,
        )
        metrics = _metrics(
            validation["predictions"].numpy(),
            validation["targets"].numpy(),
        )
        value = metrics["average"]["mae"]
        improved = value < best_mae
        if improved:
            best_mae = value
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
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
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = checkpoint_path.with_suffix(".tmp")
        torch.save(
            {
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
            },
            temporary,
        )
        os.replace(temporary, checkpoint_path)
        print(
            f"schnet/etkdg_{variant} ep{epoch:02d} "
            f"train={row['train_normalized_l1']:.5f} "
            f"val={value:.5f}eV {row['elapsed_s']:.1f}s"
            f"{' *' if improved else ''}",
            flush=True,
        )
        if wait >= patience:
            break
    if best_state is None:
        raise RuntimeError("Conformer-augmented training produced no checkpoint")
    model.load_state_dict(best_state)

    average_payloads = {}
    role_metrics = {}
    for role in ("train", "validation", "test"):
        first, second, average = _evaluate_views(
            model, *paired[role], batch_size, device, mean, std
        )
        average_payloads[role] = average
        role_metrics[role] = {
            "first": _metrics(
                first["predictions"].numpy(), first["targets"].numpy()
            ),
            "second": _metrics(
                second["predictions"].numpy(), second["targets"].numpy()
            ),
            "average": _metrics(
                average["predictions"].numpy(), average["targets"].numpy()
            ),
        }
    _atomic_torch_save(embedding_path, average_payloads)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, model_path)
    result = {
        "experiment": "qm9_schnet_two_view_training",
        "candidate": f"schnet_etkdg_{variant}",
        "seed": seed,
        "split_seed": split_seed,
        "split_fingerprint": split.fingerprint,
        "geometry_seeds": list(geometry_seeds),
        "model_config": config,
        "n_params": sum(parameter.numel() for parameter in model.parameters()),
        "paired_rows": {
            role: len(views[0]) for role, views in paired.items()
        },
        "training_graphs_per_epoch": len(train_graphs),
        "best_epoch": best_epoch,
        "best_validation_average_mae_eV": best_mae,
        "metrics": role_metrics,
        "geometry_reports": [
            {key: value for key, value in report.items() if key != "failure_indices"}
            for report in geometry_reports
        ],
        "log": log,
        "artifacts": {
            "embeddings": str(embedding_path),
            "model": str(model_path),
            "checkpoint": str(checkpoint_path),
        },
    }
    _atomic_json(result_path, result)
    return result
