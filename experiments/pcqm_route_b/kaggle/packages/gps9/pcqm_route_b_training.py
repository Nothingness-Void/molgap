"""Streaming PCQM Route B encoder continuation over accepted packed shards."""

from __future__ import annotations

import copy
import gc
import hashlib
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader

from .gps import GPSWrapper
from .schnet import SchNetWrapper

EXPECTED_MANIFEST_SHA256 = (
    "561c64074102ed9244ade72b9e60f5a5a9f6c81173f6ce33b59a5c517d4f9aae"
)
SCHNET_CONFIG = {
    "hidden_channels": 176,
    "num_filters": 160,
    "num_interactions": 6,
    "num_gaussians": 50,
    "cutoff": 10.0,
    "dropout": 0.05,
}


@dataclass(frozen=True)
class EncoderConfig:
    name: str
    kind: str
    modality: str
    hidden_channels: int
    num_layers: int
    batch_size: int
    learning_rate: float
    weight_decay: float = 1.0e-5
    max_epochs: int = 40
    patience: int = 6
    seed: int = 42
    augmented: bool = False
    training_budget_s: float = 9 * 3600
    hard_budget_s: float = 12 * 3600


CONFIGS = {
    "gps9": EncoderConfig(
        name="gps9",
        kind="gps",
        modality="gps",
        hidden_channels=192,
        num_layers=9,
        batch_size=256,
        learning_rate=8.0e-5,
    ),
    "gps11_160": EncoderConfig(
        name="gps11_160",
        kind="gps",
        modality="gps",
        hidden_channels=160,
        num_layers=11,
        batch_size=256,
        learning_rate=8.0e-5,
    ),
    "primary_schnet": EncoderConfig(
        name="primary_schnet",
        kind="schnet",
        modality="primary",
        hidden_channels=176,
        num_layers=6,
        batch_size=128,
        learning_rate=4.0e-4,
    ),
    "augmented_schnet": EncoderConfig(
        name="augmented_schnet",
        kind="schnet",
        modality="primary",
        hidden_channels=176,
        num_layers=6,
        batch_size=128,
        learning_rate=4.0e-4,
        augmented=True,
    ),
}


class PackedGraphDataset(InMemoryDataset):
    """Read one tensor-packed graph shard."""

    def __init__(self, path: Path):
        super().__init__(root=None)
        self.data, self.slices = torch.load(
            path, map_location="cpu", weights_only=False
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def verify_graph_view(root: Path, modality: str) -> dict:
    """Validate one mounted modality against the accepted shared manifest."""
    manifest_path = root / "manifest.json"
    if sha256_file(manifest_path) != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError(f"Route B manifest identity changed: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise RuntimeError("Route B graph manifest is not complete")
    expected = {}
    prefix = f"{modality}/"
    for shard in manifest["shards"]:
        expected.update(
            {
                name: digest
                for name, digest in shard["files"].items()
                if name.startswith(prefix)
            }
        )
    actual = sorted((root / modality).glob("*_shard_*.pt"))
    if len(actual) != len(expected):
        raise RuntimeError(
            f"{modality} shard count differs: {len(actual)} != {len(expected)}"
        )
    for path in actual:
        name = f"{modality}/{path.name}"
        if name not in expected or sha256_file(path) != expected[name]:
            raise RuntimeError(f"{modality} shard hash differs: {path.name}")
    return {
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "modality": modality,
        "files": len(actual),
        "split_counts": manifest["split_counts"],
    }


def expand_gps_input_state(
    source: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict]:
    """Map the old C/N/O/F/S/Cl + 3 input into the accepted 15 + 3 input."""
    if tuple(source["node_emb.weight"].shape[1:]) != (9,):
        raise RuntimeError("GPS warm start is not the expected 9-wide model")
    expanded = dict(source)
    weight = target["node_emb.weight"].clone()
    weight[:, 6:15].zero_()
    weight[:, :6] = source["node_emb.weight"][:, :6]
    weight[:, 15:18] = source["node_emb.weight"][:, 6:9]
    expanded["node_emb.weight"] = weight
    mismatched = {
        key
        for key, value in expanded.items()
        if key not in target or target[key].shape != value.shape
    }
    missing = set(target) - set(expanded)
    if mismatched or missing:
        raise RuntimeError(
            f"GPS warm-start contract differs: mismatched={sorted(mismatched)} "
            f"missing={sorted(missing)}"
        )
    return expanded, {
        "source_input_dim": 9,
        "target_input_dim": 18,
        "copied_old_element_columns": [0, 1, 2, 3, 4, 5],
        "copied_old_scalar_columns_to": [15, 16, 17],
        "zero_initialized_new_element_columns": list(range(6, 15)),
    }


def make_model(
    config: EncoderConfig,
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
            dropout=0.05,
            n_targets=3,
        )
        state, report = expand_gps_input_state(source, model.state_dict())
        model.load_state_dict(state, strict=True)
        target_mean, target_std = 0.0, 1.0
    else:
        if payload.get("model_config") != SCHNET_CONFIG:
            raise RuntimeError("SchNet warm start is not the 176/160/6 contract")
        model = SchNetWrapper(**SCHNET_CONFIG, use_charges=True, n_targets=3)
        model.load_state_dict(source, strict=True)
        mean = torch.as_tensor(payload["target_mean"]).view(-1)
        std = torch.as_tensor(payload["target_std"]).view(-1)
        target_mean, target_std = float(mean[2]), float(std[2])
        report = {
            "strict": True,
            "model_config": SCHNET_CONFIG,
            "target_mean_gap": target_mean,
            "target_std_gap": target_std,
        }
    return model.to(device), report, target_mean, target_std


def _forward(config: EncoderConfig, model: nn.Module, batch) -> torch.Tensor:
    if config.kind == "gps":
        output = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
    else:
        charges = batch.charges if hasattr(batch, "charges") else None
        output = model(batch.z, batch.pos, batch.batch, charges=charges)
    return output[:, 2]


def _encode(config: EncoderConfig, model: nn.Module, batch) -> torch.Tensor:
    if config.kind == "gps":
        return model.encode(
            batch.x, batch.edge_index, batch.edge_attr, batch.batch
        )
    charges = batch.charges if hasattr(batch, "charges") else None
    return model.encode(batch.z, batch.pos, batch.batch, charges=charges)


def _shards(root: Path, modality: str, role: str) -> list[Path]:
    paths = sorted((root / modality).glob(f"{role}_shard_*.pt"))
    if not paths:
        raise FileNotFoundError(f"No {modality}/{role} shards in {root}")
    return paths


def _train_paths(
    config: EncoderConfig,
    roots: dict[str, Path],
    epoch: int,
) -> list[Path]:
    paths = _shards(roots[config.modality], config.modality, "train")
    if config.augmented:
        paths += _shards(roots["secondary"], "secondary", "train")
    generator = random.Random(config.seed + epoch)
    generator.shuffle(paths)
    return paths


def _eV_prediction(
    config: EncoderConfig,
    prediction: torch.Tensor,
    target_mean: float,
    target_std: float,
) -> torch.Tensor:
    if config.kind == "schnet":
        return prediction * target_std + target_mean
    return prediction


@torch.no_grad()
def evaluate_dev(
    config: EncoderConfig,
    model: nn.Module,
    roots: dict[str, Path],
    device: torch.device,
    target_mean: float,
    target_std: float,
) -> float:
    model.eval()
    # The augmented branch trains on both conformers but production inference
    # uses the primary view, so epoch selection must use that same view.
    views = [config.modality]
    payloads = []
    for modality in views:
        predictions, targets, indices = [], [], []
        for path in _shards(roots[modality], modality, "dev"):
            dataset = PackedGraphDataset(path)
            for batch in DataLoader(
                dataset, batch_size=config.batch_size, shuffle=False
            ):
                batch = batch.to(device)
                with torch.amp.autocast(
                    "cuda", enabled=device.type == "cuda"
                ):
                    prediction = _forward(config, model, batch)
                predictions.append(
                    _eV_prediction(
                        config, prediction.float(), target_mean, target_std
                    ).cpu()
                )
                targets.append(batch.y.view(-1).float().cpu())
                indices.append(batch.source_idx.view(-1).long().cpu())
            del dataset
        order = torch.argsort(torch.cat(indices))
        payloads.append(
            (
                torch.cat(indices)[order],
                torch.cat(predictions)[order],
                torch.cat(targets)[order],
            )
        )
    source_idx, prediction, target = payloads[0]
    return float((prediction - target).abs().mean())


def _rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def train_encoder(
    *,
    config: EncoderConfig,
    roots: dict[str, Path],
    warm_start: Path,
    output_dir: Path,
) -> dict:
    """Continue one encoder and export durable primary-view embeddings."""
    started = time.monotonic()
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Route B encoder continuation requires a GPU")
    required = {config.modality}
    if config.augmented:
        required.add("secondary")
    input_reports = {
        modality: verify_graph_view(roots[modality], modality)
        for modality in sorted(required)
    }
    model, warm_report, target_mean, target_std = make_model(
        config, warm_start, device
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.max_epochs, eta_min=1.0e-6
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    criterion = nn.L1Loss()
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best.pt"
    last_path = output_dir / "last.pt"
    log = []
    best_state = None
    best_mae = float("inf")
    best_epoch = -1
    wait = 0
    start_epoch = 0
    if last_path.exists():
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        if checkpoint.get("config") != asdict(config):
            raise RuntimeError("Route B resume config changed")
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

    for epoch in range(start_epoch, config.max_epochs):
        elapsed = time.monotonic() - started
        previous_epoch_s = log[-1]["elapsed_s"] if log else 0.0
        if elapsed + max(previous_epoch_s, 600.0) > config.training_budget_s:
            print("training budget reached; reserving time for embeddings", flush=True)
            break
        epoch_started = time.monotonic()
        model.train()
        total_loss = 0.0
        total_rows = 0
        for path in _train_paths(config, roots, epoch):
            dataset = PackedGraphDataset(path)
            loader = DataLoader(
                dataset,
                batch_size=config.batch_size,
                shuffle=True,
                generator=torch.Generator().manual_seed(
                    config.seed * 100_000 + epoch * 1_000 + total_rows
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                total_loss += float(loss.detach()) * batch.num_graphs
                total_rows += batch.num_graphs
            del loader, dataset
            gc.collect()
        scheduler.step()
        dev_mae = evaluate_dev(
            config, model, roots, device, target_mean, target_std
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
            "elapsed_s": time.monotonic() - epoch_started,
            "selected": bool(improved),
        }
        log.append(row)
        checkpoint = {
            "format": "molgap-pcqm-route-b-encoder-checkpoint-v1",
            "config": asdict(config),
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
            "rng_state": _rng_state(),
            "target_mean_gap": target_mean,
            "target_std_gap": target_std,
            "input_reports": input_reports,
            "warm_start_sha256": sha256_file(warm_start),
            "warm_start_report": warm_report,
        }
        atomic_torch(last_path, checkpoint)
        atomic_json(
            output_dir / "progress.json",
            {
                "status": "training",
                "name": config.name,
                "epoch": epoch,
                "best_epoch": best_epoch,
                "best_dev_gap_mae_eV": best_mae,
                "elapsed_s": time.monotonic() - started,
            },
        )
        print(
            f"{config.name} ep{epoch:02d} train={row['train_gap_l1']:.6f} "
            f"dev={dev_mae:.6f}eV {row['elapsed_s']:.1f}s"
            f"{' *' if improved else ''}",
            flush=True,
        )
        if wait >= config.patience:
            break
    if best_state is None:
        raise RuntimeError("Route B training produced no finite checkpoint")
    model.load_state_dict(best_state, strict=True)
    atomic_torch(
        best_path,
        {
            "format": "molgap-pcqm-route-b-encoder-best-v1",
            "config": asdict(config),
            "model": best_state,
            "best_epoch": best_epoch,
            "best_dev_gap_mae_eV": best_mae,
            "target_mean_gap": target_mean,
            "target_std_gap": target_std,
            "input_reports": input_reports,
            "warm_start_sha256": sha256_file(warm_start),
            "warm_start_report": warm_report,
        },
    )
    embedding_manifest = export_embeddings(
        config=config,
        model=model,
        root=roots[config.modality],
        output_dir=output_dir / "embeddings",
        device=device,
        started=started,
    )
    metrics = {
        "status": "complete",
        "config": asdict(config),
        "best_epoch": best_epoch,
        "best_dev_gap_mae_eV": best_mae,
        "train_log": log,
        "input_reports": input_reports,
        "warm_start_report": warm_report,
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


@torch.no_grad()
def export_embeddings(
    *,
    config: EncoderConfig,
    model: nn.Module,
    root: Path,
    output_dir: Path,
    device: torch.device,
    started: float,
) -> dict:
    """Export independently retrievable primary-view embedding parts."""
    model.eval()
    manifest_path = output_dir / "manifest.json"
    manifest = {
        "format": "molgap-pcqm-route-b-embedding-parts-v1",
        "status": "exporting",
        "name": config.name,
        "embedding_dim": config.hidden_channels,
        "parts": [],
        "rows": {"train": 0, "dev": 0, "official": 0},
        "official_valid_metric_read": False,
    }
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            existing.get("name") == config.name
            and existing.get("embedding_dim") == config.hidden_channels
        ):
            manifest = existing
    completed = {part["source_shard"] for part in manifest["parts"]}
    for role in ("train", "dev", "official"):
        for path in _shards(root, config.modality, role):
            source_name = f"{config.modality}/{path.name}"
            if source_name in completed:
                continue
            if time.monotonic() - started > config.hard_budget_s:
                raise RuntimeError("12-hour encoder budget exhausted during embedding export")
            dataset = PackedGraphDataset(path)
            embeddings, indices, targets = [], [], []
            for batch in DataLoader(
                dataset, batch_size=config.batch_size, shuffle=False
            ):
                batch = batch.to(device)
                with torch.amp.autocast("cuda", enabled=True):
                    embedding = _encode(config, model, batch)
                embeddings.append(embedding.to(torch.float16).cpu())
                indices.append(batch.source_idx.view(-1).long().cpu())
                if role != "official":
                    targets.append(batch.y.view(-1).float().cpu())
            payload = {
                "format": "molgap-pcqm-route-b-embedding-part-v1",
                "name": config.name,
                "role": role,
                "source_shard": source_name,
                "source_idx": torch.cat(indices),
                "embeddings": torch.cat(embeddings),
            }
            if targets:
                payload["targets"] = torch.cat(targets)
            part_path = output_dir / role / f"{path.stem}_embeddings.pt"
            atomic_torch(part_path, payload)
            rows = len(payload["source_idx"])
            manifest["parts"].append(
                {
                    "role": role,
                    "source_shard": source_name,
                    "path": part_path.relative_to(output_dir).as_posix(),
                    "rows": rows,
                    "bytes": part_path.stat().st_size,
                    "sha256": sha256_file(part_path),
                }
            )
            manifest["rows"][role] += rows
            atomic_json(manifest_path, manifest)
            del dataset, payload
            gc.collect()
    manifest["status"] = "complete"
    atomic_json(manifest_path, manifest)
    return manifest
