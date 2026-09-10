"""Durable 500K direct-Gap screen for the GPTrans-T core."""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .pcqm_edgestate_scale import (
    DEVELOPMENT_ROWS,
    SUBSET_FORMAT,
    TRAIN_ROWS,
    _iter_train_batches,
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


RUN_FORMAT = "molgap-pcqm-gptrans-t-core-500k-run-v1"


@dataclass(frozen=True)
class GPTransScaleConfig:
    node_channels: int = 256
    pair_channels: int = 32
    num_layers: int = 12
    num_heads: int = 8
    shortest_path_cap: int = 20
    dropout: float = 0.1
    drop_path: float = 0.1
    layer_scale: float = 1.0
    batch_size: int = 128
    loader_workers: int = 4
    prefetch_factor: int = 4
    learning_rate: float = 1.0e-3
    minimum_learning_rate: float = 1.0e-6
    weight_decay: float = 0.05
    gradient_clip: float = 1.0
    epochs: int = 60
    warmup_epochs: int = 4
    ema_decay: float = 0.9999
    seed: int = 42

    def validate(self) -> None:
        if self.node_channels % self.num_heads:
            raise ValueError("node_channels must be divisible by num_heads")
        if self.batch_size != 128:
            raise ValueError("the frozen screen requires physical batch 128")
        if self.num_layers != 12 or self.pair_channels != 32:
            raise ValueError("GPTrans-T depth and pair width are frozen")
        if not 0.0 < self.ema_decay < 1.0:
            raise ValueError("ema_decay must fall between zero and one")


def make_model(config: GPTransScaleConfig):
    from .gptrans import OGBGPTransTiny

    return OGBGPTransTiny(
        node_channels=config.node_channels,
        pair_channels=config.pair_channels,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        shortest_path_cap=config.shortest_path_cap,
        dropout=config.dropout,
        drop_path=config.drop_path,
        layer_scale=config.layer_scale,
        n_targets=1,
    )


def parameter_count(config: GPTransScaleConfig) -> int:
    return sum(parameter.numel() for parameter in make_model(config).parameters())


def _forward(model, batch):
    return model(
        batch.x,
        batch.edge_index,
        batch.edge_attr,
        batch.batch,
    ).view(-1)


def _largest_training_graphs(cache_root, manifest, count: int):
    """Materialize only the largest graphs for a conservative memory gate."""
    import torch

    candidates = []
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
        candidates.extend(graphs[index] for index in indices)
        candidates.sort(key=lambda graph: int(graph.num_nodes), reverse=True)
        del candidates[count:]
        del graphs
    if len(candidates) != count:
        raise RuntimeError(f"expected {count} largest graphs, found {len(candidates)}")
    return candidates


def _learning_rate(config: GPTransScaleConfig, epoch: int) -> float:
    if epoch < config.warmup_epochs:
        return config.learning_rate * float(epoch + 1) / config.warmup_epochs
    progress = float(epoch - config.warmup_epochs) / max(
        1, config.epochs - config.warmup_epochs - 1
    )
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.minimum_learning_rate + (
        config.learning_rate - config.minimum_learning_rate
    ) * cosine


def _set_lr(optimizer, learning_rate: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = learning_rate


class ExponentialMovingAverage:
    def __init__(self, model, decay: float) -> None:
        self.decay = float(decay)
        self.state = {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
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


def _evaluate(model, ema, cache_root, manifest, config, mean, std, device):
    import torch

    live_state = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
    }
    model.load_state_dict(ema.state_dict(), strict=True)
    model.eval()
    absolute_error = 0.0
    count = 0
    prediction_parts = []
    target_parts = []
    source_idx_parts = []
    with torch.no_grad():
        for path in _role_shards(cache_root, manifest, "development"):
            graphs = _load_graphs(path)
            from .pcqm_edgestate_scale import _loader

            for batch in _loader(
                graphs, config, shuffle=False, seed=config.seed
            ):
                batch = batch.to(device, non_blocking=True)
                prediction = _forward(model, batch) * std + mean
                target = batch.y.view(-1).float()
                absolute_error += float((prediction - target).abs().sum())
                count += int(target.numel())
                prediction_parts.append(prediction.cpu())
                target_parts.append(target.cpu())
                source_idx = getattr(
                    batch, "source_idx", getattr(batch, "row_index", None)
                )
                if source_idx is None:
                    raise RuntimeError("development graph has no source identity")
                source_idx_parts.append(source_idx.view(-1).cpu())
            del graphs
    model.load_state_dict(live_state, strict=True)
    return {
        "mae_eV": absolute_error / count,
        "count": count,
        "prediction": torch.cat(prediction_parts),
        "target": torch.cat(target_parts),
        "source_idx": torch.cat(source_idx_parts),
    }


def _checkpoint_payload(
    *, config, epoch, model, ema, optimizer, trace, initial_hash, cache_sha256
):
    import numpy as np
    import random
    import torch

    return {
        "format": RUN_FORMAT,
        "config": asdict(config),
        "epoch": epoch,
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "trace": trace,
        "initial_encoder_sha256": initial_hash,
        "cache_aggregate_sha256": cache_sha256,
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "accelerator_rng_state": torch.cuda.get_rng_state_all(),
    }


def _load_resume(path, config, model, ema, optimizer):
    import numpy as np
    import random
    import torch

    if not path.is_file():
        return 0, []
    state = _torch_load(path)
    if state.get("format") != RUN_FORMAT or state.get("config") != asdict(config):
        raise RuntimeError("resume contract changed")
    model.load_state_dict(state["model"], strict=True)
    ema.load_state_dict(state["ema"])
    optimizer.load_state_dict(state["optimizer"])
    random.setstate(state["python_rng_state"])
    np.random.set_state(state["numpy_rng_state"])
    torch.set_rng_state(state["torch_rng_state"])
    torch.cuda.set_rng_state_all(state["accelerator_rng_state"])
    return int(state["epoch"]) + 1, list(state["trace"])


def run_preflight(
    cache_root: Path, output: Path, config: GPTransScaleConfig
) -> dict:
    import torch
    from .pcqm_edgestate_scale import _loader

    config.validate()
    acceptance_path = cache_root / "acceptance.json"
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    if acceptance.get("accepted") is not True:
        raise RuntimeError("the immutable 500K subset has not passed CPU acceptance")
    manifest = json.loads((cache_root / "manifest.json").read_text(encoding="utf-8"))
    if acceptance.get("aggregate_sha256") != manifest.get("aggregate_sha256"):
        raise RuntimeError("cache manifest and acceptance identities differ")
    set_seed(config.seed)
    device = torch.device("cuda:0")
    model = make_model(config).to(device)
    graphs = _largest_training_graphs(cache_root, manifest, config.batch_size)
    batch = next(iter(_loader(graphs, config, shuffle=False, seed=config.seed)))
    batch = batch.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    torch.cuda.reset_peak_memory_stats(device)
    prediction = _forward(model, batch)
    loss = torch.nn.functional.l1_loss(prediction, batch.y.view(-1).float())
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("preflight loss is non-finite")
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
    optimizer.step()
    payload = {
        "format": "molgap-pcqm-gptrans-t-core-500k-preflight-v1",
        "accepted": True,
        "config": asdict(config),
        "parameter_count": parameter_count(config),
        "loss": float(loss.detach()),
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "prediction_count": int(prediction.numel()),
        "largest_batch_max_nodes": max(int(graph.num_nodes) for graph in graphs),
        "largest_batch_min_nodes": min(int(graph.num_nodes) for graph in graphs),
        "cache_acceptance": acceptance,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
    }
    atomic_json(output / "preflight.json", payload)
    return payload


def run_worker(
    cache_root: Path,
    output: Path,
    config: GPTransScaleConfig,
    *,
    source_commit: str,
) -> dict:
    import torch
    import torch.nn.functional as functional

    config.validate()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("SCNet worker requires exactly one visible DCU")
    output.mkdir(parents=True, exist_ok=True)
    completion = output / "completion_manifest.json"
    if completion.is_file():
        payload = json.loads(completion.read_text(encoding="utf-8"))
        if payload.get("complete") is True and payload.get("config") == asdict(config):
            return payload
        raise RuntimeError("incompatible completion manifest already exists")
    manifest = json.loads((cache_root / "manifest.json").read_text(encoding="utf-8"))
    acceptance = json.loads((cache_root / "acceptance.json").read_text(encoding="utf-8"))
    if manifest.get("format") != SUBSET_FORMAT or acceptance.get("accepted") is not True:
        raise RuntimeError("500K subset has not passed acceptance")

    set_seed(config.seed)
    device = torch.device("cuda:0")
    model = make_model(config).to(device)
    initial_hash = state_sha256(model)
    ema = ExponentialMovingAverage(model, config.ema_decay)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    checkpoint = output / "last_checkpoint.pt"
    start_epoch, trace = _load_resume(checkpoint, config, model, ema, optimizer)
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
        count = 0
        epoch_started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats(device)
        for batch in _iter_train_batches(cache_root, manifest, config, epoch):
            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prediction = _forward(model, batch)
            target = (batch.y.view(-1).float() - mean_tensor) / std_tensor
            loss = functional.l1_loss(prediction, target)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"non-finite loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()
            ema.update(model)
            absolute_error += float(
                (prediction.detach() * std_tensor + mean_tensor - batch.y.view(-1))
                .abs()
                .sum()
            )
            count += int(batch.y.numel())
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
                    "config": asdict(config),
                    "epoch": epoch,
                    "model": ema.state_dict(),
                    "target_mean": mean,
                    "target_std": std,
                    "cache_aggregate_sha256": manifest["aggregate_sha256"],
                },
            )
            atomic_torch_save(
                output / "development_predictions.pt",
                {
                    "prediction_eV": development["prediction"],
                    "target_eV": development["target"],
                    "source_idx": development["source_idx"],
                    "official_validation_role_read": False,
                    "test_dev_role_read": False,
                },
            )
        row = {
            "epoch": epoch,
            "train_mae_eV": absolute_error / count,
            "development_mae_eV": development["mae_eV"],
            "elapsed_s": elapsed,
            "graphs_per_s": TRAIN_ROWS / elapsed,
            "learning_rate": learning_rate,
            "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
            "improved": improved,
        }
        trace.append(row)
        atomic_json(output / "trace.json", {"epochs": trace})
        atomic_torch_save(
            checkpoint,
            _checkpoint_payload(
                config=config,
                epoch=epoch,
                model=model,
                ema=ema,
                optimizer=optimizer,
                trace=trace,
                initial_hash=initial_hash,
                cache_sha256=manifest["aggregate_sha256"],
            ),
        )
        print(
            f"gptrans_t_core ep{epoch:02d} train={row['train_mae_eV']:.6f} "
            f"dev={row['development_mae_eV']:.6f}eV {elapsed:.1f}s "
            f"peak={row['peak_memory_bytes'] / 2**30:.2f}GiB"
            f"{' *' if improved else ''}",
            flush=True,
        )
    payload = {
        "format": RUN_FORMAT,
        "complete": True,
        "source_commit": source_commit,
        "config": asdict(config),
        "architecture": "GPTrans-T core transfer",
        "parameter_count": parameter_count(config),
        "initial_encoder_sha256": initial_hash,
        "cache_aggregate_sha256": manifest["aggregate_sha256"],
        "train_rows": TRAIN_ROWS,
        "development_rows": DEVELOPMENT_ROWS,
        "best_epoch": best_epoch,
        "best_development_mae_eV": best,
        "best_sha256": sha256_file(output / "best_model.pt"),
        "last_sha256": sha256_file(checkpoint),
        "trace": trace,
        "elapsed_s": time.perf_counter() - started,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
    }
    atomic_json(output / "metrics.json", payload)
    atomic_json(completion, payload)
    return payload
