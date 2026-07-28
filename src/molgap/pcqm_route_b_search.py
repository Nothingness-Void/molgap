"""Nested, resumable hyperparameter search for PCQM Route B encoders."""

from __future__ import annotations

import copy
import gc
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader

from .gps import GPSWrapper
from .pcqm_route_b_training import (
    CONFIGS,
    EncoderConfig,
    PackedGraphDataset,
    _eV_prediction,
    _forward,
    atomic_json,
    atomic_torch,
    expand_gps_input_state,
    set_seed,
    sha256_file,
)
from .schnet import SchNetWrapper


SEARCH_FORMAT = "molgap-pcqm-route-b-hparam-search-v1"
SUBSET_FORMAT = "molgap-pcqm-route-b-nested-subsets-v1"
SCHNET_CUTOFF_ANGSTROM = 6.0
SUBSET_SEED = 20260728
TRAIN_CORE_ROWS = 50_000
TRAIN_EXTENSION_ROWS = 50_000
DEV_ROWS = 10_000
SHARD_ROWS = 5_000


@dataclass(frozen=True)
class SearchTrial:
    trial_id: str
    learning_rate: float
    weight_decay: float
    dropout: float
    batch_size: int
    warmup_ratio: float
    grad_clip: float


def _trial(
    index: int,
    learning_rate: float,
    weight_decay: float,
    dropout: float,
    batch_size: int,
    warmup_ratio: float,
    grad_clip: float = 1.0,
) -> SearchTrial:
    return SearchTrial(
        trial_id=f"trial_{index:02d}",
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        dropout=dropout,
        batch_size=batch_size,
        warmup_ratio=warmup_ratio,
        grad_clip=grad_clip,
    )


GPS_TRIALS = (
    _trial(0, 8e-5, 1e-5, 0.05, 256, 0.00),
    _trial(1, 4e-5, 1e-5, 0.05, 256, 0.05),
    _trial(2, 1.6e-4, 1e-5, 0.05, 256, 0.05),
    _trial(3, 8e-5, 1e-6, 0.05, 256, 0.05),
    _trial(4, 8e-5, 1e-4, 0.05, 256, 0.05),
    _trial(5, 8e-5, 1e-5, 0.00, 256, 0.05),
    _trial(6, 8e-5, 1e-5, 0.10, 256, 0.05),
    _trial(7, 8e-5, 1e-5, 0.15, 256, 0.05),
    _trial(8, 4e-5, 1e-4, 0.10, 128, 0.10),
    _trial(9, 1.6e-4, 1e-6, 0.00, 384, 0.10, 0.5),
    _trial(10, 6e-5, 3e-5, 0.10, 384, 0.05),
    _trial(11, 1.2e-4, 3e-6, 0.05, 128, 0.10, 2.0),
)

SCHNET_TRIALS = (
    _trial(0, 4e-4, 1e-5, 0.05, 128, 0.00),
    _trial(1, 1e-4, 1e-5, 0.05, 128, 0.05),
    _trial(2, 2e-4, 1e-5, 0.05, 128, 0.05),
    _trial(3, 8e-4, 1e-5, 0.05, 128, 0.05),
    _trial(4, 4e-4, 1e-6, 0.05, 128, 0.05),
    _trial(5, 4e-4, 1e-4, 0.05, 128, 0.05),
    _trial(6, 4e-4, 1e-5, 0.00, 128, 0.05),
    _trial(7, 4e-4, 1e-5, 0.10, 128, 0.05),
    _trial(8, 2e-4, 1e-4, 0.10, 64, 0.10),
    _trial(9, 8e-4, 1e-6, 0.00, 192, 0.10, 0.5),
    _trial(10, 3e-4, 3e-5, 0.10, 192, 0.05),
    _trial(11, 6e-4, 3e-6, 0.05, 64, 0.10, 2.0),
)


def search_trials(encoder_name: str) -> tuple[SearchTrial, ...]:
    config = CONFIGS[encoder_name]
    return GPS_TRIALS if config.kind == "gps" else SCHNET_TRIALS


def _splitmix64(values: np.ndarray, seed: int) -> np.ndarray:
    values = values.astype(np.uint64, copy=True) + np.uint64(seed)
    values = (values ^ (values >> np.uint64(30))) * np.uint64(
        0xBF58476D1CE4E5B9
    )
    values = (values ^ (values >> np.uint64(27))) * np.uint64(
        0x94D049BB133111EB
    )
    return values ^ (values >> np.uint64(31))


def _collect_source_indices(paths: list[Path]) -> np.ndarray:
    values = []
    for path in paths:
        dataset = PackedGraphDataset(path)
        values.append(dataset._data.source_idx.view(-1).numpy().astype(np.int64))
        del dataset
    result = np.concatenate(values)
    if len(np.unique(result)) != len(result):
        raise RuntimeError("Duplicate source_idx in Route B source split")
    return result


def _ranked_subset(values: np.ndarray, rows: int, seed: int) -> np.ndarray:
    if len(values) < rows:
        raise ValueError(f"Requested {rows} rows from only {len(values)}")
    hashes = _splitmix64(values, seed)
    order = np.lexsort((values, hashes))
    return values[order[:rows]]


def _pack_graphs(path: Path, graphs: list, *, source_indices: list[int]) -> dict:
    if not graphs:
        raise RuntimeError(f"Refusing to write empty graph shard: {path}")
    data, slices = InMemoryDataset.collate(graphs)
    atomic_torch(path, (data, slices))
    return {
        "path": path.as_posix(),
        "rows": len(graphs),
        "source_idx_min": min(source_indices),
        "source_idx_max": max(source_indices),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _extract_selected(
    *,
    paths: list[Path],
    selected: dict[int, str],
    output_dir: Path,
    modality: str,
) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    buffers: dict[str, list] = {"core": [], "extension": [], "dev": []}
    indices: dict[str, list[int]] = {"core": [], "extension": [], "dev": []}
    counters = {name: 0 for name in buffers}
    reports = []
    observed: set[int] = set()

    # Atomic shards are the resume boundary. Load completed shards first, then
    # scan the immutable source while skipping their accepted identities.
    for role in buffers:
        existing = sorted(output_dir.glob(f"{role}_shard_*.pt"))
        counters[role] = len(existing)
        for path in existing:
            dataset = PackedGraphDataset(path)
            source_indices = dataset._data.source_idx.view(-1).tolist()
            del dataset
            for source_idx in source_indices:
                source_idx = int(source_idx)
                if selected.get(source_idx) != role:
                    raise RuntimeError(
                        f"Resume shard role differs: {path} source_idx={source_idx}"
                    )
                if source_idx in observed:
                    raise RuntimeError(
                        f"Duplicate source_idx across resume shards: {source_idx}"
                    )
                observed.add(source_idx)
            reports.append(
                {
                    "path": path.as_posix(),
                    "rows": len(source_indices),
                    "source_idx_min": min(source_indices),
                    "source_idx_max": max(source_indices),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "role": role,
                    "modality": modality,
                    "resumed": True,
                }
            )
    resumed_indices = set(observed)

    def flush(role: str) -> None:
        if not buffers[role]:
            return
        shard_path = output_dir / f"{role}_shard_{counters[role]:03d}.pt"
        report = _pack_graphs(
            shard_path,
            buffers[role],
            source_indices=indices[role],
        )
        report.update({"role": role, "modality": modality})
        reports.append(report)
        counters[role] += 1
        buffers[role] = []
        indices[role] = []

    for path in paths:
        dataset = PackedGraphDataset(path)
        for graph in dataset:
            source_idx = int(graph.source_idx.view(-1)[0])
            role = selected.get(source_idx)
            if role is None:
                continue
            if source_idx in observed:
                if source_idx in resumed_indices:
                    continue
                raise RuntimeError(f"Duplicate selected source_idx: {source_idx}")
            observed.add(source_idx)
            buffers[role].append(graph)
            indices[role].append(source_idx)
            if len(buffers[role]) >= SHARD_ROWS:
                flush(role)
        del dataset
        gc.collect()
    for role in buffers:
        flush(role)
    if observed != set(selected):
        missing = sorted(set(selected) - observed)
        raise RuntimeError(
            f"{modality} subset extraction missed {len(missing)} rows"
        )
    return reports


def build_nested_subsets(
    *,
    source_root: Path,
    output_root: Path,
) -> dict:
    """Build aligned 50K/100K train and fixed 10K dev graph shards."""
    manifest_path = output_root / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("status") == "complete":
            for artifact in existing["artifacts"]:
                path = output_root / artifact["relative_path"]
                if not path.exists() or sha256_file(path) != artifact["sha256"]:
                    raise RuntimeError(f"Nested subset artifact changed: {path}")
            return existing

    gps_train = sorted((source_root / "gps").glob("train_shard_*.pt"))
    gps_dev = sorted((source_root / "gps").glob("dev_shard_*.pt"))
    if not gps_train or not gps_dev:
        raise FileNotFoundError("Accepted Route B GPS train/dev shards are missing")
    train_indices = _collect_source_indices(gps_train)
    dev_indices = _collect_source_indices(gps_dev)
    ranked_train = _ranked_subset(
        train_indices,
        TRAIN_CORE_ROWS + TRAIN_EXTENSION_ROWS,
        SUBSET_SEED,
    )
    ranked_dev = _ranked_subset(dev_indices, DEV_ROWS, SUBSET_SEED + 1)
    selected_train = {
        int(value): "core" if index < TRAIN_CORE_ROWS else "extension"
        for index, value in enumerate(ranked_train)
    }
    selected_dev = {int(value): "dev" for value in ranked_dev}
    selected = selected_train | selected_dev

    reports = []
    for modality in ("gps", "primary", "secondary"):
        paths = sorted((source_root / modality).glob("train_shard_*.pt"))
        paths += sorted((source_root / modality).glob("dev_shard_*.pt"))
        reports.extend(
            _extract_selected(
                paths=paths,
                selected=selected,
                output_dir=output_root / modality,
                modality=modality,
            )
        )

    role_counts = {}
    role_indices = {}
    for modality in ("gps", "primary", "secondary"):
        role_counts[modality] = {}
        role_indices[modality] = {}
        for role in ("core", "extension", "dev"):
            indices = []
            for path in sorted((output_root / modality).glob(f"{role}_shard_*.pt")):
                dataset = PackedGraphDataset(path)
                indices.extend(dataset._data.source_idx.view(-1).tolist())
                del dataset
            role_counts[modality][role] = len(indices)
            role_indices[modality][role] = indices
    reference = role_indices["gps"]
    for modality in ("primary", "secondary"):
        for role in reference:
            if set(role_indices[modality][role]) != set(reference[role]):
                raise RuntimeError(f"{modality}/{role} alignment differs")

    artifacts = []
    for report in reports:
        path = Path(report["path"])
        artifacts.append(
            {
                "relative_path": path.relative_to(output_root).as_posix(),
                "rows": report["rows"],
                "bytes": report["bytes"],
                "sha256": report["sha256"],
            }
        )
    manifest = {
        "format": SUBSET_FORMAT,
        "status": "complete",
        "source_root": str(source_root),
        "selection": {
            "seed": SUBSET_SEED,
            "method": "splitmix64 source_idx rank",
            "nested": "core 50K is contained in core+extension 100K",
            "dev_policy": "fixed 10K sample from scaffold-development only",
        },
        "counts": role_counts,
        "source_idx_sha256": {
            role: _indices_sha256(values) for role, values in reference.items()
        },
        "artifacts": sorted(artifacts, key=lambda item: item["relative_path"]),
        "official_valid_read": False,
        "official_test_used": False,
        "sealed_20k_used": False,
    }
    atomic_json(manifest_path, manifest)
    return manifest


def _indices_sha256(values: list[int]) -> str:
    array = np.asarray(sorted(values), dtype="<i8")
    import hashlib

    return hashlib.sha256(array.tobytes()).hexdigest()


def verify_nested_subsets(root: Path) -> dict:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != SUBSET_FORMAT or manifest.get("status") != "complete":
        raise RuntimeError("Route B nested subset manifest is not complete")
    for artifact in manifest["artifacts"]:
        path = root / artifact["relative_path"]
        if (
            not path.exists()
            or path.stat().st_size != artifact["bytes"]
            or sha256_file(path) != artifact["sha256"]
        ):
            raise RuntimeError(f"Nested subset artifact differs: {path}")
    expected = {
        "core": TRAIN_CORE_ROWS,
        "extension": TRAIN_EXTENSION_ROWS,
        "dev": DEV_ROWS,
    }
    for modality, counts in manifest["counts"].items():
        if counts != expected:
            raise RuntimeError(f"Nested subset counts differ for {modality}: {counts}")
    return manifest


def _search_model(
    config: EncoderConfig,
    trial: SearchTrial,
    warm_start: Path,
    device: torch.device,
) -> tuple[nn.Module, dict, float, float]:
    payload = torch.load(warm_start, map_location="cpu", weights_only=False)
    source = payload.get("model", payload)
    if config.kind == "gps":
        model = GPSWrapper(
            in_channels=18,
            edge_dim=4,
            hidden_channels=config.hidden_channels,
            num_layers=config.num_layers,
            num_heads=4,
            dropout=trial.dropout,
            n_targets=3,
        )
        state, report = expand_gps_input_state(source, model.state_dict())
        model.load_state_dict(state, strict=True)
        return model.to(device), report, 0.0, 1.0

    source_config = payload.get("model_config", {})
    required = {
        "hidden_channels": 176,
        "num_filters": 160,
        "num_interactions": 6,
        "num_gaussians": 50,
    }
    if any(source_config.get(key) != value for key, value in required.items()):
        raise RuntimeError(f"SchNet warm-start architecture differs: {source_config}")
    target_config = {
        **required,
        "cutoff": SCHNET_CUTOFF_ANGSTROM,
        "dropout": trial.dropout,
    }
    model = SchNetWrapper(**target_config, use_charges=True, n_targets=3)
    target_state = model.state_dict()
    migrated = dict(source)
    offset_key = "schnet.distance_expansion.offset"
    if offset_key not in migrated or offset_key not in target_state:
        raise RuntimeError("SchNet distance basis is missing from state_dict")
    migrated[offset_key] = target_state[offset_key]
    model.load_state_dict(migrated, strict=True)
    mean = torch.as_tensor(payload["target_mean"]).view(-1)
    std = torch.as_tensor(payload["target_std"]).view(-1)
    report = {
        "strict_except_distance_basis": True,
        "source_model_config": source_config,
        "target_model_config": target_config,
        "distance_basis_reinitialized": True,
        "cutoff_fixed_by_protocol_A": SCHNET_CUTOFF_ANGSTROM,
    }
    return model.to(device), report, float(mean[2]), float(std[2])


def _paths_for_stage(
    subset_root: Path,
    modality: str,
    stage: str,
) -> list[Path]:
    roles = ["core"] if stage == "50k" else ["core", "extension"]
    paths = []
    for role in roles:
        paths.extend(sorted((subset_root / modality).glob(f"{role}_shard_*.pt")))
    if not paths:
        raise FileNotFoundError(f"No {modality} training shards for {stage}")
    return paths


@torch.no_grad()
def _evaluate_search_dev(
    *,
    config: EncoderConfig,
    model: nn.Module,
    subset_root: Path,
    device: torch.device,
    batch_size: int,
    target_mean: float,
    target_std: float,
) -> float:
    model.eval()
    predictions, targets, indices = [], [], []
    for path in sorted(
        (subset_root / config.modality).glob("dev_shard_*.pt")
    ):
        dataset = PackedGraphDataset(path)
        for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
            batch = batch.to(device)
            with torch.amp.autocast("cuda", enabled=True):
                prediction = _forward(config, model, batch)
            predictions.append(
                _eV_prediction(
                    config, prediction.float(), target_mean, target_std
                ).cpu()
            )
            targets.append(batch.y.view(-1).float().cpu())
            indices.append(batch.source_idx.view(-1).long().cpu())
        del dataset
    source_idx = torch.cat(indices)
    if len(torch.unique(source_idx)) != DEV_ROWS:
        raise RuntimeError("Search dev source_idx coverage differs")
    order = torch.argsort(source_idx)
    return float(
        (torch.cat(predictions)[order] - torch.cat(targets)[order]).abs().mean()
    )


def _scheduler_lambda(
    epoch: int,
    *,
    epochs: int,
    warmup_ratio: float,
) -> float:
    warmup_epochs = int(round(epochs * warmup_ratio))
    if warmup_epochs and epoch < warmup_epochs:
        return float(epoch + 1) / warmup_epochs
    progress = (epoch - warmup_epochs) / max(epochs - warmup_epochs - 1, 1)
    return 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * progress))


def preflight_search(
    *,
    subset_root: Path,
    warm_starts: dict[str, Path],
    output_path: Path,
) -> dict:
    """Run one real CUDA optimization step for every search encoder."""
    verify_nested_subsets(subset_root)
    if not torch.cuda.is_available():
        raise RuntimeError("Route B search preflight requires CUDA")
    device = torch.device("cuda")
    reports = {}
    for encoder_name in sorted(CONFIGS):
        config = CONFIGS[encoder_name]
        trial = search_trials(encoder_name)[0]
        set_seed(config.seed)
        model, warm_report, target_mean, target_std = _search_model(
            config,
            trial,
            warm_starts[encoder_name],
            device,
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=trial.learning_rate,
            weight_decay=trial.weight_decay,
        )
        scaler = torch.amp.GradScaler("cuda", enabled=True)
        modalities = [config.modality]
        if config.augmented:
            modalities.append("secondary")
        losses = []
        torch.cuda.reset_peak_memory_stats()
        for modality in modalities:
            path = sorted(
                (subset_root / modality).glob("core_shard_*.pt")
            )[0]
            dataset = PackedGraphDataset(path)
            batch = next(
                iter(
                    DataLoader(
                        dataset,
                        batch_size=min(8, trial.batch_size),
                        shuffle=False,
                    )
                )
            ).to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=True):
                prediction = _forward(config, model, batch)
                target = batch.y.view(-1)
                if config.kind == "schnet":
                    target = (target - target_mean) / target_std
                loss = nn.functional.l1_loss(prediction, target)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), trial.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
            del batch, dataset
        torch.cuda.synchronize()
        if not all(np.isfinite(loss) for loss in losses):
            raise RuntimeError(f"Non-finite search preflight: {encoder_name}")
        reports[encoder_name] = {
            "trial": asdict(trial),
            "losses": losses,
            "warm_start_report": warm_report,
            "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
            "device_name": torch.cuda.get_device_name(0),
            "schnet_cutoff_A": (
                float(model.schnet.cutoff) if config.kind == "schnet" else None
            ),
            "distance_basis_max_A": (
                float(model.schnet.distance_expansion.offset[-1])
                if config.kind == "schnet"
                else None
            ),
        }
        del model, optimizer, scaler
        torch.cuda.empty_cache()
    report = {
        "format": "molgap-pcqm-route-b-hparam-preflight-v1",
        "status": "complete",
        "encoders": reports,
        "subset_manifest_sha256": sha256_file(subset_root / "manifest.json"),
        "official_valid_metric_read": False,
        "official_test_used": False,
        "sealed_20k_used": False,
    }
    atomic_json(output_path, report)
    return report


def train_search_trial(
    *,
    encoder_name: str,
    trial: SearchTrial,
    stage: str,
    subset_root: Path,
    warm_start: Path,
    output_dir: Path,
    seed: int = 42,
) -> dict:
    if stage not in {"50k", "100k", "100k_confirm"}:
        raise ValueError(f"Unsupported Route B search stage: {stage}")
    verify_nested_subsets(subset_root)
    completion_path = output_dir / "completion_manifest.json"
    if completion_path.exists():
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        if completion.get("status") == "complete":
            return json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))

    config = replace(
        CONFIGS[encoder_name],
        batch_size=trial.batch_size,
        learning_rate=trial.learning_rate,
        weight_decay=trial.weight_decay,
        seed=seed,
    )
    epochs = 20 if stage == "50k" else 24
    patience = 4 if stage == "50k" else 5
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Route B hyperparameter search requires CUDA")
    model, warm_report, target_mean, target_std = _search_model(
        config, trial, warm_start, device
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=trial.learning_rate,
        weight_decay=trial.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda epoch: _scheduler_lambda(
            epoch, epochs=epochs, warmup_ratio=trial.warmup_ratio
        ),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    criterion = nn.L1Loss()
    output_dir.mkdir(parents=True, exist_ok=True)
    last_path = output_dir / "last.pt"
    best_state = None
    best_mae = float("inf")
    best_epoch = -1
    wait = 0
    log = []
    start_epoch = 0
    contract = {
        "format": SEARCH_FORMAT,
        "encoder": encoder_name,
        "stage": stage,
        "trial": asdict(trial),
        "seed": seed,
        "epochs": epochs,
        "patience": patience,
        "subset_manifest_sha256": sha256_file(subset_root / "manifest.json"),
        "warm_start_sha256": sha256_file(warm_start),
        "schnet_cutoff_A": (
            SCHNET_CUTOFF_ANGSTROM if config.kind == "schnet" else None
        ),
    }
    if last_path.exists():
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        if checkpoint.get("contract") != contract:
            raise RuntimeError("Route B search resume contract changed")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        best_state = checkpoint["best_state"]
        best_mae = float(checkpoint["best_dev_gap_mae_eV"])
        best_epoch = int(checkpoint["best_epoch"])
        wait = int(checkpoint["wait"])
        log = list(checkpoint["log"])
        start_epoch = int(checkpoint["epoch"]) + 1

    data_stage = "50k" if stage == "50k" else "100k"
    train_paths = _paths_for_stage(subset_root, config.modality, data_stage)
    if config.augmented:
        train_paths += _paths_for_stage(subset_root, "secondary", data_stage)
    for epoch in range(start_epoch, epochs):
        epoch_started = time.monotonic()
        model.train()
        total_loss = 0.0
        total_rows = 0
        paths = list(train_paths)
        random.Random(seed + epoch).shuffle(paths)
        for path in paths:
            dataset = PackedGraphDataset(path)
            loader = DataLoader(
                dataset,
                batch_size=trial.batch_size,
                shuffle=True,
                generator=torch.Generator().manual_seed(
                    seed * 100_000 + epoch * 1_000 + total_rows
                ),
            )
            for batch in loader:
                batch = batch.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=True):
                    prediction = _forward(config, model, batch)
                    target = batch.y.view(-1)
                    if config.kind == "schnet":
                        target = (target - target_mean) / target_std
                    loss = criterion(prediction, target)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), trial.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                total_loss += float(loss.detach()) * batch.num_graphs
                total_rows += batch.num_graphs
            del loader, dataset
            gc.collect()
        scheduler.step()
        dev_mae = _evaluate_search_dev(
            config=config,
            model=model,
            subset_root=subset_root,
            device=device,
            batch_size=trial.batch_size,
            target_mean=target_mean,
            target_std=target_std,
        )
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
            "train_gap_l1": total_loss / max(total_rows, 1),
            "train_draw_rows": total_rows,
            "dev_gap_mae_eV": dev_mae,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "elapsed_s": time.monotonic() - epoch_started,
            "selected": bool(improved),
        }
        log.append(row)
        atomic_torch(
            last_path,
            {
                "contract": contract,
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "best_state": best_state,
                "best_epoch": best_epoch,
                "best_dev_gap_mae_eV": best_mae,
                "wait": wait,
                "log": log,
            },
        )
        atomic_json(
            output_dir / "progress.json",
            {
                "status": "training",
                **contract,
                "epoch": epoch,
                "best_epoch": best_epoch,
                "best_dev_gap_mae_eV": best_mae,
            },
        )
        print(
            f"{encoder_name}/{stage}/{trial.trial_id} ep{epoch:02d} "
            f"train={row['train_gap_l1']:.6f} dev={dev_mae:.6f}eV "
            f"{row['elapsed_s']:.1f}s{' *' if improved else ''}",
            flush=True,
        )
        if wait >= patience:
            break
    if best_state is None:
        raise RuntimeError("Route B search produced no finite checkpoint")
    atomic_torch(
        output_dir / "best.pt",
        {
            "contract": contract,
            "model": best_state,
            "best_epoch": best_epoch,
            "best_dev_gap_mae_eV": best_mae,
            "target_mean_gap": target_mean,
            "target_std_gap": target_std,
            "warm_start_report": warm_report,
        },
    )
    metrics = {
        "status": "complete",
        **contract,
        "best_epoch": best_epoch,
        "best_dev_gap_mae_eV": best_mae,
        "train_log": log,
        "warm_start_report": warm_report,
        "official_valid_metric_read": False,
        "official_test_used": False,
        "sealed_20k_used": False,
        "production_registry_changed": False,
    }
    atomic_json(output_dir / "metrics.json", metrics)
    atomic_json(
        completion_path,
        {
            "status": "complete",
            "metrics_sha256": sha256_file(output_dir / "metrics.json"),
            "best_sha256": sha256_file(output_dir / "best.pt"),
            "last_sha256": sha256_file(last_path),
        },
    )
    return metrics


def run_search_stage(
    *,
    encoder_name: str,
    stage: str,
    subset_root: Path,
    warm_start: Path,
    output_root: Path,
    top_k: int = 4,
) -> dict:
    trials = list(search_trials(encoder_name))
    if stage in {"100k", "100k_confirm"}:
        prior_stage = "50k" if stage == "100k" else "100k"
        prior = json.loads(
            (
                output_root.parent
                / prior_stage
                / encoder_name
                / "summary.json"
            ).read_text(encoding="utf-8")
        )
        promotion_count = top_k if stage == "100k" else 2
        promoted = {
            item["trial_id"] for item in prior["ranking"][:promotion_count]
        }
        trials = [trial for trial in trials if trial.trial_id in promoted]
    results = []
    for trial in trials:
        seeds = (42, 43, 44) if stage == "100k_confirm" else (42,)
        for seed in seeds:
            trial_output = output_root / encoder_name / trial.trial_id
            if stage == "100k_confirm":
                trial_output = trial_output / f"seed_{seed}"
            result = train_search_trial(
                encoder_name=encoder_name,
                trial=trial,
                stage=stage,
                subset_root=subset_root,
                warm_start=warm_start,
                output_dir=trial_output,
                seed=seed,
            )
            results.append(result)
    if stage == "100k_confirm":
        grouped = {}
        for result in results:
            grouped.setdefault(result["trial"]["trial_id"], []).append(result)
        ranking = [
            {
                "trial_id": trial_id,
                "mean_best_dev_gap_mae_eV": float(
                    np.mean(
                        [item["best_dev_gap_mae_eV"] for item in trial_results]
                    )
                ),
                "std_best_dev_gap_mae_eV": float(
                    np.std(
                        [item["best_dev_gap_mae_eV"] for item in trial_results]
                    )
                ),
                "seeds": {
                    str(item["seed"]): item["best_dev_gap_mae_eV"]
                    for item in trial_results
                },
                "trial": trial_results[0]["trial"],
            }
            for trial_id, trial_results in grouped.items()
        ]
        ranking.sort(key=lambda item: item["mean_best_dev_gap_mae_eV"])
    else:
        ranking = sorted(
            (
                {
                    "trial_id": result["trial"]["trial_id"],
                    "best_dev_gap_mae_eV": result["best_dev_gap_mae_eV"],
                    "best_epoch": result["best_epoch"],
                    "trial": result["trial"],
                }
                for result in results
            ),
            key=lambda item: item["best_dev_gap_mae_eV"],
        )
    summary = {
        "format": SEARCH_FORMAT,
        "status": "complete",
        "encoder": encoder_name,
        "stage": stage,
        "ranking": ranking,
        "promotion_count": min(top_k, len(ranking)),
        "official_valid_metric_read": False,
        "official_test_used": False,
        "sealed_20k_used": False,
    }
    atomic_json(output_root / encoder_name / "summary.json", summary)
    return summary
