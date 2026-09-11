"""Durable fixed-500K training and fusion for geometry-transfer candidates."""
from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .pcqm_distance_angle_scale import (
    DEVELOPMENT_ROWS,
    SUBSET_FORMAT,
    TRAIN_ROWS,
    _load_graphs,
    _role_shards,
    _target_stats,
    _torch_load,
    atomic_json,
    atomic_torch_save,
    set_seed,
    sha256_file,
    state_sha256,
)
from .pcqm_geometry_transfer import (
    GEOMETRY_GPTRANS_T,
    GEOMETRY_NEURAL_ATOM_K1,
    MODEL_IDS,
    forward_geometry_transfer,
    make_geometry_transfer_model,
)


RUN_FORMAT = "molgap-pcqm-geometry-transfer-500k-run-v1"
FUSION_FORMAT = "molgap-pcqm-geometry-transfer-500k-fusion-v1"
EXPECTED_CACHE_SHA256 = (
    "676a506c808402bc16a4437cc02286168239a8dd5de4451e992131eddb4f1b20"
)
EXPECTED_PARAMETERS = {
    GEOMETRY_GPTRANS_T: 5_251_425,
    GEOMETRY_NEURAL_ATOM_K1: 3_778_801,
}


@dataclass(frozen=True)
class GeometryTransferConfig:
    model_id: str
    batch_size: int = 128
    loader_workers: int = 4
    prefetch_factor: int = 4
    gradient_clip: float = 1.0
    seed: int = 42
    precision: str = "fp32"
    drop_last: bool = True

    @property
    def epochs(self) -> int:
        return 60 if self.model_id == GEOMETRY_GPTRANS_T else 40

    @property
    def learning_rate(self) -> float:
        return 1.0e-3 if self.model_id == GEOMETRY_GPTRANS_T else 4.0e-4

    @property
    def minimum_learning_rate(self) -> float:
        return 1.0e-6

    @property
    def weight_decay(self) -> float:
        return 0.05 if self.model_id == GEOMETRY_GPTRANS_T else 1.0e-5

    @property
    def warmup_epochs(self) -> int:
        return 4 if self.model_id == GEOMETRY_GPTRANS_T else 0

    @property
    def ema_decay(self) -> float | None:
        return 0.9999 if self.model_id == GEOMETRY_GPTRANS_T else None

    def validate(self) -> None:
        if self.model_id not in MODEL_IDS:
            raise ValueError(f"unsupported model_id: {self.model_id}")
        if self.batch_size != 128 or not self.drop_last:
            raise ValueError("the v4 screen requires physical BS128 and drop_last")
        if self.precision != "fp32":
            raise ValueError("the frozen comparison requires FP32")

    def fingerprint(self) -> dict:
        payload = asdict(self)
        payload.update(
            epochs=self.epochs,
            learning_rate=self.learning_rate,
            minimum_learning_rate=self.minimum_learning_rate,
            weight_decay=self.weight_decay,
            warmup_epochs=self.warmup_epochs,
            ema_decay=self.ema_decay,
        )
        return payload


class ExponentialMovingAverage:
    def __init__(self, model, decay: float) -> None:
        self.decay = float(decay)
        self.state = {
            name: value.detach().clone() for name, value in model.state_dict().items()
        }

    def update(self, model) -> None:
        for name, value in model.state_dict().items():
            target = self.state[name]
            if value.is_floating_point():
                target.mul_(self.decay).add_(value.detach(), alpha=1.0 - self.decay)
            else:
                target.copy_(value)

    def state_dict(self):
        return self.state

    def load_state_dict(self, state) -> None:
        if state.keys() != self.state.keys():
            raise RuntimeError("EMA state keys changed")
        self.state = {name: value.clone() for name, value in state.items()}


def _loader(graphs, config: GeometryTransferConfig, *, shuffle: bool, seed: int, train: bool):
    import torch
    from torch_geometric.loader import DataLoader

    options = {}
    if config.loader_workers:
        options.update(
            num_workers=config.loader_workers,
            prefetch_factor=config.prefetch_factor,
            persistent_workers=False,
        )
    return DataLoader(
        graphs,
        batch_size=config.batch_size,
        shuffle=shuffle,
        drop_last=config.drop_last if train else False,
        generator=torch.Generator().manual_seed(seed),
        pin_memory=True,
        **options,
    )


def _iter_train_batches(cache_root, manifest, config, epoch):
    paths = _role_shards(cache_root, manifest, "train")
    order = list(range(len(paths)))
    random.Random(config.seed + 10_000 * epoch).shuffle(order)
    for position, shard_id in enumerate(order):
        graphs = _load_graphs(paths[shard_id])
        loader = _loader(
            graphs,
            config,
            shuffle=True,
            seed=config.seed + epoch * 101 + position,
            train=True,
        )
        yield from loader
        del loader, graphs


def _learning_rate(config: GeometryTransferConfig, epoch: int) -> float:
    if epoch < config.warmup_epochs:
        return config.learning_rate * float(epoch + 1) / config.warmup_epochs
    progress = float(epoch - config.warmup_epochs) / max(
        1, config.epochs - config.warmup_epochs - 1
    )
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.minimum_learning_rate + (
        config.learning_rate - config.minimum_learning_rate
    ) * cosine


def _set_lr(optimizer, value: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = value


def _accepted_cache(cache_root: Path) -> tuple[dict, dict]:
    manifest = json.loads((cache_root / "manifest.json").read_text(encoding="utf-8"))
    acceptance = json.loads(
        (cache_root / "acceptance.json").read_text(encoding="utf-8")
    )
    if manifest.get("format") != SUBSET_FORMAT or acceptance.get("accepted") is not True:
        raise RuntimeError("the fixed 500K cache is not accepted")
    if manifest.get("aggregate_sha256") != EXPECTED_CACHE_SHA256:
        raise RuntimeError("the fixed 500K cache identity changed")
    if acceptance.get("aggregate_sha256") != EXPECTED_CACHE_SHA256:
        raise RuntimeError("cache acceptance identity changed")
    if manifest.get("train_rows") != TRAIN_ROWS or manifest.get(
        "development_rows"
    ) != DEVELOPMENT_ROWS:
        raise RuntimeError("fixed role counts changed")
    return manifest, acceptance


def _largest_training_graphs(cache_root, manifest, count: int):
    import torch

    selected = []
    for path in _role_shards(cache_root, manifest, "train"):
        graphs = _load_graphs(path)
        if hasattr(graphs, "slices") and "x" in graphs.slices:
            sizes = graphs.slices["x"][1:] - graphs.slices["x"][:-1]
            indices = torch.topk(sizes, k=min(count, len(graphs))).indices.tolist()
        else:
            indices = sorted(
                range(len(graphs)),
                key=lambda index: int(graphs[index].num_nodes),
                reverse=True,
            )[:count]
        selected.extend(graphs[index] for index in indices)
        selected.sort(key=lambda graph: int(graph.num_nodes), reverse=True)
        del selected[count:]
        del graphs
    if len(selected) != count:
        raise RuntimeError(f"expected {count} large graphs, found {len(selected)}")
    return selected


def _evaluate(model, ema, cache_root, manifest, config, mean, std, device):
    import torch

    live_state = None
    if ema is not None:
        live_state = {
            name: value.detach().clone() for name, value in model.state_dict().items()
        }
        model.load_state_dict(ema.state_dict(), strict=True)
    model.eval()
    predictions = []
    targets = []
    source_indices = []
    with torch.no_grad():
        for path in _role_shards(cache_root, manifest, "development"):
            graphs = _load_graphs(path)
            for batch in _loader(
                graphs, config, shuffle=False, seed=config.seed, train=False
            ):
                batch = batch.to(device, non_blocking=True)
                prediction = forward_geometry_transfer(model, batch) * std + mean
                predictions.append(prediction.float().cpu())
                targets.append(batch.y.view(-1).float().cpu())
                source_idx = getattr(
                    batch, "source_idx", getattr(batch, "row_index", None)
                )
                if source_idx is None:
                    raise RuntimeError("development graph has no source identity")
                source_indices.append(source_idx.view(-1).long().cpu())
            del graphs
    if live_state is not None:
        model.load_state_dict(live_state, strict=True)
    prediction = torch.cat(predictions)
    target = torch.cat(targets)
    source_idx = torch.cat(source_indices)
    return {
        "mae_eV": float((prediction - target).abs().mean()),
        "prediction": prediction,
        "target": target,
        "source_idx": source_idx,
    }


def _checkpoint_payload(
    config,
    epoch,
    model,
    ema,
    optimizer,
    trace,
    initial_hash,
    source_commit,
):
    import torch

    return {
        "format": RUN_FORMAT,
        "config": config.fingerprint(),
        "epoch": epoch,
        "model": model.state_dict(),
        "ema": None if ema is None else ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "trace": trace,
        "initial_model_sha256": initial_hash,
        "source_commit": source_commit,
        "cache_aggregate_sha256": EXPECTED_CACHE_SHA256,
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "accelerator_rng_state": torch.cuda.get_rng_state_all(),
    }


def _resume(path, config, model, ema, optimizer, *, source_commit):
    import torch

    if not path.is_file():
        return 0, []
    payload = _torch_load(path)
    if payload.get("format") != RUN_FORMAT or payload.get(
        "config"
    ) != config.fingerprint():
        raise RuntimeError("resume contract changed")
    if payload.get("source_commit") != source_commit or payload.get(
        "cache_aggregate_sha256"
    ) != EXPECTED_CACHE_SHA256:
        raise RuntimeError("resume source or cache identity changed")
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    if ema is not None:
        if payload.get("ema") is None:
            raise RuntimeError("resume checkpoint lost EMA state")
        ema.load_state_dict(payload["ema"])
    random.setstate(payload["python_rng_state"])
    np.random.set_state(payload["numpy_rng_state"])
    torch.set_rng_state(payload["torch_rng_state"])
    torch.cuda.set_rng_state_all(payload["accelerator_rng_state"])
    return int(payload["epoch"]) + 1, list(payload["trace"])


def run_preflight(cache_root: Path, output: Path, config: GeometryTransferConfig) -> dict:
    import torch
    import torch.nn.functional as functional

    config.validate()
    manifest, acceptance = _accepted_cache(cache_root)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("preflight requires exactly one visible accelerator")
    set_seed(config.seed)
    device = torch.device("cuda:0")
    model = make_geometry_transfer_model(config.model_id).to(device)
    parameters = sum(value.numel() for value in model.parameters())
    if parameters != EXPECTED_PARAMETERS[config.model_id]:
        raise RuntimeError(f"parameter count changed: {parameters}")
    graphs = _largest_training_graphs(cache_root, manifest, config.batch_size)
    batch = next(
        iter(_loader(graphs, config, shuffle=False, seed=config.seed, train=True))
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    torch.cuda.reset_peak_memory_stats(device)
    prediction = forward_geometry_transfer(model, batch)
    loss = functional.l1_loss(prediction, batch.y.view(-1).float())
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("preflight loss is non-finite")
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
    optimizer.step()
    peak = int(torch.cuda.max_memory_allocated(device))
    total = int(torch.cuda.get_device_properties(device).total_memory)
    reserve = 1.0 - peak / total
    if reserve < 0.15:
        raise RuntimeError(f"preflight memory reserve is only {reserve:.1%}")
    payload = {
        "format": "molgap-pcqm-geometry-transfer-500k-preflight-v1",
        "accepted": True,
        "model_id": config.model_id,
        "config": config.fingerprint(),
        "parameter_count": parameters,
        "loss": float(loss.detach()),
        "prediction_count": int(prediction.numel()),
        "peak_memory_bytes": peak,
        "total_memory_bytes": total,
        "memory_reserve_fraction": reserve,
        "accelerator": torch.cuda.get_device_name(0),
        "cache_acceptance": acceptance,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
    }
    atomic_json(output / "preflight.json", payload)
    return payload


def run_training(
    cache_root: Path,
    output: Path,
    config: GeometryTransferConfig,
    *,
    source_commit: str,
) -> dict:
    import torch
    import torch.nn.functional as functional

    config.validate()
    manifest, _ = _accepted_cache(cache_root)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("training requires exactly one visible accelerator")
    output.mkdir(parents=True, exist_ok=True)
    completion = output / "completion_manifest.json"
    if completion.is_file():
        payload = json.loads(completion.read_text(encoding="utf-8"))
        if payload.get("complete") is True and payload.get(
            "config"
        ) == config.fingerprint():
            return payload
        raise RuntimeError("incompatible completion manifest already exists")

    set_seed(config.seed)
    device = torch.device("cuda:0")
    model = make_geometry_transfer_model(config.model_id).to(device)
    parameters = sum(value.numel() for value in model.parameters())
    if parameters != EXPECTED_PARAMETERS[config.model_id]:
        raise RuntimeError(f"parameter count changed: {parameters}")
    initial_hash = state_sha256(model)
    ema = (
        ExponentialMovingAverage(model, config.ema_decay)
        if config.ema_decay is not None
        else None
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    checkpoint = output / "last_checkpoint.pt"
    start_epoch, trace = _resume(
        checkpoint, config, model, ema, optimizer, source_commit=source_commit
    )
    mean, std = _target_stats(cache_root, manifest)
    mean_tensor = torch.tensor(mean, device=device)
    std_tensor = torch.tensor(std, device=device)
    best = min((row["development_mae_eV"] for row in trace), default=float("inf"))
    best_epoch = next(
        (row["epoch"] for row in trace if row["development_mae_eV"] == best), -1
    )
    started = time.perf_counter()

    for epoch in range(start_epoch, config.epochs):
        learning_rate = _learning_rate(config, epoch)
        _set_lr(optimizer, learning_rate)
        model.train()
        absolute_error = 0.0
        rows = 0
        optimizer_steps = 0
        epoch_started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats(device)
        for batch in _iter_train_batches(cache_root, manifest, config, epoch):
            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prediction = forward_geometry_transfer(model, batch)
            target = (batch.y.view(-1).float() - mean_tensor) / std_tensor
            loss = functional.l1_loss(prediction, target)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"non-finite loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()
            if ema is not None:
                ema.update(model)
            absolute_error += float(
                (prediction.detach() * std_tensor + mean_tensor - batch.y.view(-1))
                .abs()
                .sum()
            )
            rows += int(batch.y.numel())
            optimizer_steps += 1
        expected_steps = 10 * (50_000 // config.batch_size)
        expected_rows = expected_steps * config.batch_size
        if optimizer_steps != expected_steps or rows != expected_rows:
            raise RuntimeError(
                f"optimizer exposure changed: steps={optimizer_steps}, rows={rows}"
            )
        development = _evaluate(
            model, ema, cache_root, manifest, config, mean_tensor, std_tensor, device
        )
        elapsed = time.perf_counter() - epoch_started
        improved = development["mae_eV"] < best
        if improved:
            best = development["mae_eV"]
            best_epoch = epoch
            atomic_torch_save(
                output / "best_model.pt",
                {
                    "format": RUN_FORMAT,
                    "model_id": config.model_id,
                    "config": config.fingerprint(),
                    "epoch": epoch,
                    "model": (
                        model.state_dict() if ema is None else ema.state_dict()
                    ),
                    "target_mean_eV": mean,
                    "target_std_eV": std,
                    "source_commit": source_commit,
                    "cache_aggregate_sha256": EXPECTED_CACHE_SHA256,
                },
            )
            atomic_torch_save(
                output / "best_validation_payload.pt",
                {
                    "source_idx": development["source_idx"],
                    "target_eV": development["target"],
                    "prediction_eV": development["prediction"],
                },
            )
        row = {
            "epoch": epoch,
            "train_mae_eV": absolute_error / rows,
            "development_mae_eV": development["mae_eV"],
            "optimizer_steps": optimizer_steps,
            "presented_rows": rows,
            "seconds": elapsed,
            "learning_rate": learning_rate,
            "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
            "improved": improved,
        }
        trace.append(row)
        atomic_torch_save(
            checkpoint,
            _checkpoint_payload(
                config,
                epoch,
                model,
                ema,
                optimizer,
                trace,
                initial_hash,
                source_commit,
            ),
        )
        atomic_json(output / "trace.json", {"epochs": trace})
        atomic_json(
            output / "progress.json",
            {
                "complete": False,
                "model_id": config.model_id,
                "epoch_completed": epoch,
                "epochs": config.epochs,
                "best_epoch": best_epoch,
                "best_development_mae_eV": best,
                "checkpoint_sha256": sha256_file(checkpoint),
            },
        )
        print(
            f"{config.model_id} ep{epoch:02d} train={row['train_mae_eV']:.6f} "
            f"dev={row['development_mae_eV']:.6f}eV {elapsed:.1f}s"
            f"{' *' if improved else ''}",
            flush=True,
        )

    result = {
        "format": RUN_FORMAT,
        "complete": True,
        "model_id": config.model_id,
        "config": config.fingerprint(),
        "source_commit": source_commit,
        "cache_aggregate_sha256": EXPECTED_CACHE_SHA256,
        "parameter_count": parameters,
        "initial_model_sha256": initial_hash,
        "best_epoch": best_epoch,
        "best_development_mae_eV": best,
        "epochs_completed": len(trace),
        "optimizer_steps_per_epoch": trace[-1]["optimizer_steps"],
        "presented_rows_per_epoch": trace[-1]["presented_rows"],
        "wall_seconds": time.perf_counter() - started,
        "mean_epoch_seconds": float(np.mean([row["seconds"] for row in trace])),
        "accelerator": torch.cuda.get_device_name(0),
        "artifacts": {
            name: sha256_file(output / name)
            for name in (
                "best_model.pt",
                "best_validation_payload.pt",
                "last_checkpoint.pt",
                "trace.json",
            )
        },
        "official_validation_role_read": False,
        "test_dev_role_read": False,
    }
    atomic_json(output / "metrics.json", result)
    atomic_json(output / "completion_manifest.json", result)
    return result


def _convex_weight(reference, candidate, target) -> tuple[float, float]:
    weights = np.linspace(0.0, 1.0, 1001, dtype=np.float64)
    best_weight = 0.0
    best_mae = float("inf")
    for weight in weights:
        prediction = weight * reference + (1.0 - weight) * candidate
        mae = float(np.mean(np.abs(prediction - target)))
        if mae < best_mae:
            best_weight = float(weight)
            best_mae = mae
    return best_weight, best_mae


def run_fusion(output_root: Path) -> dict:
    import torch

    payloads = {}
    metrics = {}
    for model_id in MODEL_IDS:
        root = output_root / model_id
        metrics[model_id] = json.loads(
            (root / "completion_manifest.json").read_text(encoding="utf-8")
        )
        if metrics[model_id].get("complete") is not True:
            raise RuntimeError(f"upstream model is incomplete: {model_id}")
        payloads[model_id] = _torch_load(root / "best_validation_payload.pt")
    first, second = MODEL_IDS
    for key in ("source_idx", "target_eV"):
        if not torch.equal(payloads[first][key], payloads[second][key]):
            raise RuntimeError(f"upstream validation {key} is not aligned")
    source_idx = payloads[first]["source_idx"].long()
    expected = torch.arange(TRAIN_ROWS, TRAIN_ROWS + DEVELOPMENT_ROWS)
    if not torch.equal(source_idx, expected):
        raise RuntimeError("development source indices changed")
    target = payloads[first]["target_eV"].double().numpy()
    predictions = {
        model_id: payloads[model_id]["prediction_eV"].double().numpy()
        for model_id in MODEL_IDS
    }
    if not all(np.isfinite(value).all() for value in [target, *predictions.values()]):
        raise RuntimeError("fusion inputs contain non-finite values")

    equal = 0.5 * (predictions[first] + predictions[second])
    oracle = np.where(
        np.abs(predictions[first] - target)
        <= np.abs(predictions[second] - target),
        predictions[first],
        predictions[second],
    )
    folds = source_idx.numpy() % 5
    oof = np.empty_like(target)
    fold_weights = []
    for fold in range(5):
        train = folds != fold
        held_out = folds == fold
        weight, _ = _convex_weight(
            predictions[first][train], predictions[second][train], target[train]
        )
        oof[held_out] = (
            weight * predictions[first][held_out]
            + (1.0 - weight) * predictions[second][held_out]
        )
        fold_weights.append(weight)
    mae = lambda value: float(np.mean(np.abs(value - target)))
    result = {
        "format": FUSION_FORMAT,
        "complete": True,
        "source_idx_start": int(source_idx[0]),
        "source_idx_stop": int(source_idx[-1]) + 1,
        "rows": int(source_idx.numel()),
        "component_metrics": {
            model_id: metrics[model_id]["best_development_mae_eV"]
            for model_id in MODEL_IDS
        },
        "development_mae_eV": {
            first: mae(predictions[first]),
            second: mae(predictions[second]),
            "equal_blend": mae(equal),
            "five_fold_oof_convex_blend": mae(oof),
            "oracle_upper_bound": mae(oracle),
        },
        "five_fold_reference_weights": fold_weights,
        "oracle_only_for_headroom": True,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
        "component_artifact_sha256": {
            model_id: metrics[model_id]["artifacts"]["best_validation_payload.pt"]
            for model_id in MODEL_IDS
        },
    }
    atomic_json(output_root / "fusion_metrics.json", result)
    return result
