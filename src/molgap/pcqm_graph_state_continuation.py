"""Resumable convergence continuation for the completed GraphState full run."""
from __future__ import annotations

import copy
import gc
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

from .pcqm_geometry_audit import ACCEPTANCE_SHA, checked_acceptance
from .pcqm_geometry_scratch import (
    _device,
    _load,
    _records,
    capture_rng,
    evaluate,
    forward,
    make_matched_model,
    restore_rng,
    seed_all,
)
from .pcqm_official_edge_state import atomic_json, atomic_torch, sha256_file


@dataclass(frozen=True)
class GraphStateContinuationConfig:
    source_epochs: int = 12
    max_total_epochs: int = 80
    batch_size: int = 192
    learning_rate: float = 4e-5
    minimum_learning_rate: float = 5e-7
    weight_decay: float = 1e-6
    plateau_factor: float = 0.5
    scheduler_patience: int = 2
    convergence_patience: int = 8
    convergence_delta_eV: float = 5e-5
    gradient_clip: float = 1.0
    segment_seconds: float = 12_600
    precision: str = "fp32_tf32_off"


def _safe_artifact(root: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise RuntimeError(f"Unsafe source artifact path: {relative}")
    return root / path


def accept_source(source_dir: Path, acceptance_path: Path) -> dict:
    """Validate the immutable completed 12-epoch source and return its states."""
    source_dir, acceptance_path = Path(source_dir), Path(acceptance_path)
    acceptance = checked_acceptance(acceptance_path)
    completion_path = source_dir / "completion_manifest.json"
    metrics_path = source_dir / "metrics.json"
    if not completion_path.is_file() or not metrics_path.is_file():
        raise FileNotFoundError("Completed GraphState source artifacts are missing")
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if completion.get("status") != "complete" or metrics.get("status") != "complete":
        raise RuntimeError("GraphState source is not complete")
    for relative, digest in completion.get("hashes", {}).items():
        path = _safe_artifact(source_dir, relative)
        if not path.is_file() or sha256_file(path) != digest:
            raise RuntimeError(f"GraphState source artifact changed: {relative}")
    last_path, best_path = source_dir / "last.pt", source_dir / "best.pt"
    last = torch.load(last_path, map_location="cpu", weights_only=False)
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    identity = metrics.get("identity")
    if (
        identity is None
        or identity.get("arm") != "graphstate"
        or last.get("identity") != identity
        or best.get("identity") != identity
        or identity.get("acceptance_sha256") != ACCEPTANCE_SHA
    ):
        raise RuntimeError("GraphState source identity differs")
    source_epochs = int(identity["config"]["epochs"])
    if (
        int(last.get("epoch", -1)) != source_epochs
        or int(last.get("next_shard", -1)) != 0
        or int(metrics.get("best_epoch", -1)) != source_epochs - 1
        or not math.isfinite(float(metrics.get("best_mae_eV", float("nan"))))
    ):
        raise RuntimeError("GraphState source did not finish its frozen schedule")
    return {
        "acceptance": acceptance,
        "identity": identity,
        "last": last,
        "best": best,
        "metrics": metrics,
        "hashes": {
            "acceptance": sha256_file(acceptance_path),
            "completion": sha256_file(completion_path),
            "metrics": sha256_file(metrics_path),
            "last": sha256_file(last_path),
            "best": sha256_file(best_path),
        },
    }


def convergence_update(anchor: float, wait: int, value: float, delta: float) -> tuple[float, int]:
    if value < anchor - delta:
        return value, 0
    return anchor, wait + 1


def _identity(source: dict, config: GraphStateContinuationConfig) -> dict:
    return {
        "format": "molgap-pcqm-graphstate-convergence-v1",
        "config": asdict(config),
        "source_identity": source["identity"],
        "source_hashes": source["hashes"],
        "code_sha256": sha256_file(Path(__file__)),
        "model_source_sha256": sha256_file(Path(__file__).with_name("pcqm_graph_state.py")),
    }


def preflight(
    graph_dir: Path,
    acceptance_path: Path,
    source_dir: Path,
    output_path: Path,
    config: GraphStateContinuationConfig = GraphStateContinuationConfig(),
) -> dict:
    source = accept_source(source_dir, acceptance_path)
    if config.source_epochs != int(source["last"]["epoch"]):
        raise RuntimeError("Continuation source epoch contract changed")
    device = _device()
    model = make_matched_model("graphstate", source["identity"]["config"]["seed"]).to(device)
    model.load_state_dict(source["last"]["model"], strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    optimizer.load_state_dict(source["last"]["optimizer"])
    for group in optimizer.param_groups:
        group["lr"] = config.learning_rate
        group["initial_lr"] = config.learning_rate
    record = _records(source["acceptance"], "train")[0]
    dataset = _load(Path(graph_dir), record)
    batch = next(iter(DataLoader(dataset, batch_size=config.batch_size, num_workers=0))).to(device)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    prediction = forward(model, batch, "graphstate")
    target = (batch.y.view(-1) - source["acceptance"]["target_mean_gap"]) / source["acceptance"]["target_std_gap"]
    loss = torch.nn.functional.l1_loss(prediction, target)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip, error_if_nonfinite=True)
    if not torch.isfinite(loss) or not torch.isfinite(grad_norm):
        raise RuntimeError("Non-finite continuation preflight")
    optimizer.step()
    if not all(torch.isfinite(parameter).all() for parameter in model.parameters()):
        raise RuntimeError("Continuation optimizer step produced non-finite parameters")
    report = {
        "status": "accepted",
        "identity": _identity(source, config),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "loss": float(loss.detach()),
        "gradient_norm": float(grad_norm),
        "memory_headroom": 1 - torch.cuda.max_memory_reserved() / torch.cuda.get_device_properties(0).total_memory,
    }
    if report["parameters"] != 3_665_809 or report["memory_headroom"] < 0.15:
        raise RuntimeError("Continuation preflight resource contract failed")
    atomic_json(Path(output_path), report)
    return report


def train(
    graph_dir: Path,
    acceptance_path: Path,
    source_dir: Path,
    preflight_path: Path,
    output_dir: Path,
    config: GraphStateContinuationConfig = GraphStateContinuationConfig(),
) -> dict:
    graph_dir, source_dir, output_dir = Path(graph_dir), Path(source_dir), Path(output_dir)
    source = accept_source(source_dir, acceptance_path)
    identity = _identity(source, config)
    report = json.loads(Path(preflight_path).read_text(encoding="utf-8"))
    if report.get("status") != "accepted" or report.get("identity") != identity:
        raise RuntimeError("Continuation preflight identity differs")
    output_dir.mkdir(parents=True, exist_ok=True)
    completion_path = output_dir / "completion_manifest.json"
    if completion_path.is_file():
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        if completion.get("identity") != identity:
            raise RuntimeError("Completed continuation identity differs")
        for relative, digest in completion["hashes"].items():
            if sha256_file(_safe_artifact(output_dir, relative)) != digest:
                raise RuntimeError(f"Completed continuation artifact changed: {relative}")
        return json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))

    device = _device()
    seed = int(source["identity"]["config"]["seed"])
    model = make_matched_model("graphstate", seed).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.plateau_factor,
        patience=config.scheduler_patience,
        min_lr=config.minimum_learning_rate,
    )
    start_epoch, cursor, loss_sum, rows = config.source_epochs, 0, 0.0, 0
    best = float(source["metrics"]["best_mae_eV"])
    best_epoch = int(source["metrics"]["best_epoch"])
    anchor, wait, log, elapsed_offset = best, 0, [], 0.0
    last_path, best_path = output_dir / "last.pt", output_dir / "best.pt"
    started = time.monotonic()

    if last_path.is_file():
        state = torch.load(last_path, map_location="cpu", weights_only=False)
        if state.get("identity") != identity:
            raise RuntimeError("Continuation checkpoint identity differs")
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        restore_rng(state["rng"])
        start_epoch, cursor = int(state["epoch"]), int(state["next_shard"])
        loss_sum, rows = float(state["loss_sum"]), int(state["rows"])
        best, best_epoch = float(state["best"]), int(state["best_epoch"])
        anchor, wait = float(state["convergence_anchor"]), int(state["wait"])
        log, elapsed_offset = list(state["log"]), float(state["elapsed_seconds"])
    else:
        model.load_state_dict(source["last"]["model"], strict=True)
        optimizer.load_state_dict(source["last"]["optimizer"])
        for group in optimizer.param_groups:
            group["lr"] = config.learning_rate
            group["initial_lr"] = config.learning_rate
        seed_all(seed + config.source_epochs)
        atomic_torch(output_dir / "initial.pt", {
            "identity": identity,
            "model": copy.deepcopy(model.state_dict()),
            "source_epoch": config.source_epochs - 1,
        })
        atomic_torch(best_path, {
            "identity": identity,
            "model": copy.deepcopy(source["best"]["model"]),
            "best_epoch": best_epoch,
            "mae_eV": best,
            "target_mean_gap": source["acceptance"]["target_mean_gap"],
            "target_std_gap": source["acceptance"]["target_std_gap"],
        })

    def save(epoch: int, next_shard: int) -> None:
        elapsed = elapsed_offset + time.monotonic() - started
        atomic_torch(last_path, {
            "identity": identity,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "rng": capture_rng(),
            "epoch": epoch,
            "next_shard": next_shard,
            "loss_sum": loss_sum,
            "rows": rows,
            "best": best,
            "best_epoch": best_epoch,
            "convergence_anchor": anchor,
            "wait": wait,
            "log": log,
            "elapsed_seconds": elapsed,
        })
        atomic_json(output_dir / "progress.json", {
            "status": "training",
            "epoch": epoch,
            "next_shard": next_shard,
            "rows": rows,
            "best_epoch": best_epoch,
            "best_mae_eV": best,
            "convergence_wait": wait,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "checkpoint": str(last_path),
        })

    records = _records(source["acceptance"], "train")
    stop_reason = None
    for epoch in range(start_epoch, config.max_total_epochs):
        epoch_started = time.monotonic()
        shuffled = list(records)
        random.Random(seed + epoch).shuffle(shuffled)
        model.train()
        for position in range(cursor, len(shuffled)):
            shard_seed = seed + epoch * 1000 + position
            seed_all(shard_seed)
            dataset = _load(graph_dir, shuffled[position])
            generator = torch.Generator().manual_seed(shard_seed)
            loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True,
                                generator=generator, num_workers=0)
            for batch in loader:
                batch = batch.to(device)
                optimizer.zero_grad(set_to_none=True)
                prediction = forward(model, batch, "graphstate")
                target = (batch.y.view(-1) - source["acceptance"]["target_mean_gap"]) / source["acceptance"]["target_std_gap"]
                loss = torch.nn.functional.l1_loss(prediction, target)
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite continuation training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip, error_if_nonfinite=True)
                optimizer.step()
                loss_sum += float(loss.detach()) * target.numel()
                rows += target.numel()
            del loader, dataset
            gc.collect()
            save(epoch, position + 1)
            print(f"graphstate-cont ep{epoch:02d} shard={position + 1}/{len(shuffled)} rows={rows}", flush=True)
            if time.monotonic() - started >= config.segment_seconds:
                atomic_json(output_dir / "progress.json", {
                    "status": "paused",
                    "reason": "segment_budget",
                    "epoch": epoch,
                    "next_shard": position + 1,
                    "checkpoint": str(last_path),
                    "identity": identity,
                })
                return {"status": "paused", "epoch": epoch, "next_shard": position + 1}
        if rows != source["acceptance"]["counts"]["train"]:
            raise RuntimeError("Continuation training row accounting differs")
        valid = evaluate(model, graph_dir, source["acceptance"], "graphstate",
                         output_dir / f"validation_ep{epoch:02d}", config.batch_size)
        selected = valid < best
        if selected:
            best, best_epoch = valid, epoch
            atomic_torch(best_path, {
                "identity": identity,
                "model": copy.deepcopy(model.state_dict()),
                "best_epoch": best_epoch,
                "mae_eV": best,
                "target_mean_gap": source["acceptance"]["target_mean_gap"],
                "target_std_gap": source["acceptance"]["target_std_gap"],
            })
        anchor, wait = convergence_update(anchor, wait, valid, config.convergence_delta_eV)
        scheduler.step(valid)
        log.append({
            "epoch": epoch,
            "train_l1_normalized": loss_sum / rows,
            "valid_mae_eV": valid,
            "selected": selected,
            "convergence_wait": wait,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "seconds": time.monotonic() - epoch_started,
        })
        loss_sum, rows, cursor = 0.0, 0, 0
        save(epoch + 1, 0)
        atomic_json(output_dir / "train_log.json", log)
        print(f"graphstate-cont ep{epoch:02d} valid={valid:.6f} best={best:.6f}@{best_epoch} wait={wait}", flush=True)
        if wait >= config.convergence_patience:
            stop_reason = "converged"
            break
    if stop_reason is None:
        stop_reason = "maximum_epoch_safety_ceiling"

    best_state = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best_state["model"], strict=True)
    accepted_best = evaluate(model, graph_dir, source["acceptance"], "graphstate",
                             output_dir / "validation_best", config.batch_size)
    if abs(accepted_best - best) > 1e-9:
        raise RuntimeError("Best continuation metric did not reproduce")
    result = {
        "status": "complete",
        "identity": identity,
        "stop_reason": stop_reason,
        "source_best_epoch": int(source["metrics"]["best_epoch"]),
        "source_best_mae_eV": float(source["metrics"]["best_mae_eV"]),
        "best_epoch": best_epoch,
        "best_mae_eV": best,
        "delta_vs_source_eV": best - float(source["metrics"]["best_mae_eV"]),
        "log": log,
        "runtime_seconds": elapsed_offset + time.monotonic() - started,
        "official_valid_used_for_convergence": True,
        "official_test_used": False,
        "pretrained_weights_used": False,
        "continued_from_completed_full_run": True,
    }
    atomic_json(output_dir / "metrics.json", result)
    atomic_json(output_dir / "progress.json", {
        "status": "complete",
        "epoch": log[-1]["epoch"] + 1,
        "stop_reason": stop_reason,
        "best_epoch": best_epoch,
        "best_mae_eV": best,
    })
    hashes = {
        relative: sha256_file(output_dir / relative)
        for relative in ("metrics.json", "best.pt", "last.pt", "initial.pt", "validation_best/manifest.json")
    }
    atomic_json(completion_path, {"status": "complete", "identity": identity, "hashes": hashes})
    return result
