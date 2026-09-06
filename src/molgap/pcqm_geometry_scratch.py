"""Matched full-data scratch training over the immutable geometry cache."""
from __future__ import annotations

import gc
import json
import math
import random
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from .pcqm_gap_architecture import make_pcqm_gap_encoder
from .pcqm_geometry_audit import ACCEPTANCE_SHA, checked_acceptance
from .pcqm_geometry_warmstart import CANDIDATE, _forward_geometry, load_pretrained_backbone
from .pcqm_official_edge_state import PackedGraphDataset, atomic_json, atomic_torch, sha256_file

ARMS = {"triangle": "ogb_sparse_triangle_edge_state_gps9", "geometry": CANDIDATE}
GRAPHSTATE = "ogb_distance_angle_triangle_edge_state_graph_state9"


@dataclass(frozen=True)
class ScratchConfig:
    seed: int = 42
    epochs: int = 12
    batch_size: int = 192
    learning_rate: float = 4e-4
    minimum_learning_rate: float = 1e-6
    weight_decay: float = 1e-5
    max_training_seconds: float = 43200
    job_seconds: float = 41400
    precision: str = "fp32_tf32_off"


def config_for(arm):
    config = ScratchConfig()
    if arm == "graphstate":
        return replace(config, learning_rate=1.6e-4, weight_decay=1e-6)
    return config


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([v.cpu() for v in state["cuda"]])


def make_matched_model(arm: str, seed: int):
    seed_all(seed)
    if arm == "graphstate":
        from .pcqm_graph_state import make_graph_state
        return make_graph_state()
    baseline = make_pcqm_gap_encoder(ARMS["triangle"])
    if arm == "triangle":
        return baseline
    if arm != "geometry":
        raise ValueError(arm)
    model = make_pcqm_gap_encoder(ARMS["geometry"])
    load_pretrained_backbone(model, baseline.state_dict())
    return model


def forward(model, batch, arm):
    if arm in {"geometry", "graphstate"}:
        return _forward_geometry(model, batch)
    return model(batch.x, batch.edge_index, batch.edge_attr, batch.batch,
                 batch.random_walk_pe, batch.wedge_edge_ids).view(-1)


def _device():
    if not torch.cuda.is_available() or "A100" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Frozen full-data contract requires a scheduled A100")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return torch.device("cuda")


def _records(acceptance, role):
    return sorted([r for r in acceptance["shards"] if r["role"] == role], key=lambda r: r["path"])


def _load(root, record):
    path = root / record["path"]
    if sha256_file(path) != record["sha256"]:
        raise RuntimeError(f"Immutable shard changed: {path}")
    dataset = PackedGraphDataset(path)
    if len(dataset) != record["rows"]:
        raise RuntimeError("Shard row count changed")
    return dataset


def require_audit(directory):
    report = json.loads((directory / "result.json").read_text())
    manifest = json.loads((directory / "completion_manifest.json").read_text())
    if (report["status"] != "accepted" or report["identity"]["acceptance_sha256"] != ACCEPTANCE_SHA
            or manifest["result_sha256"] != sha256_file(directory / "result.json")):
        raise RuntimeError("P0 audit did not pass immutable acceptance")
    return sha256_file(directory / "result.json")


@torch.no_grad()
def evaluate(model, root, acceptance, arm, output, batch_size):
    model.eval()
    mean, std = acceptance["target_mean_gap"], acceptance["target_std_gap"]
    parts, total, count = [], 0.0, 0
    for record in _records(acceptance, "valid"):
        dataset = _load(root, record)
        ids, values, targets = [], [], []
        for batch in DataLoader(dataset, batch_size=batch_size, num_workers=0):
            batch = batch.cuda()
            value = forward(model, batch, arm).float() * std + mean
            target = batch.y.view(-1)
            if not torch.isfinite(value).all():
                raise RuntimeError("Non-finite validation output")
            total += float((value.double() - target.double()).abs().sum())
            count += target.numel()
            ids.append(batch.source_idx.view(-1).cpu())
            values.append(value.cpu())
            targets.append(target.cpu())
        part = output / Path(record["path"]).name
        atomic_torch(part, {"source_idx": torch.cat(ids), "prediction_eV": torch.cat(values), "target_eV": torch.cat(targets)})
        parts.append({"path": part.name, "sha256": sha256_file(part), "rows": len(torch.cat(ids))})
        del dataset
        gc.collect()
    if count != acceptance["counts"]["valid"]:
        raise RuntimeError("Incomplete validation")
    atomic_json(output / "manifest.json", {"rows": count, "mae_eV": total / count, "parts": parts})
    return total / count


def preflight(root: Path, acceptance_path: Path, audit_dir: Path, output: Path, *, graphstate=False):
    config = config_for("graphstate" if graphstate else "triangle")
    acceptance = checked_acceptance(acceptance_path)
    audit_hash = require_audit(audit_dir)
    device = _device()
    record = _records(acceptance, "train")[0]
    dataset = _load(root, record)
    first = next(iter(DataLoader(dataset, batch_size=config.batch_size))).to(device)
    difference = None
    if not graphstate:
        a, b = make_matched_model("triangle", config.seed).to(device).eval(), make_matched_model("geometry", config.seed).to(device).eval()
        with torch.no_grad():
            difference = float((forward(a, first, "triangle") - forward(b, first, "geometry")).abs().max())
        del a, b
    del first
    gc.collect()
    torch.cuda.empty_cache()
    if difference is not None and difference > 2e-5:
        raise RuntimeError("Scratch initial function mismatch")
    reports = {}
    for arm in (["graphstate"] if graphstate else ARMS):
        model = make_matched_model(arm, config.seed).to(device).train()
        if graphstate and sum(p.numel() for p in model.parameters()) != 3665809:
            raise RuntimeError("Frozen GraphState parameter count changed")
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.monotonic()
        rows = 0
        for index, batch in enumerate(DataLoader(dataset, batch_size=config.batch_size, num_workers=0)):
            if index >= 64:
                break
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = forward(model, batch, arm)
            target = (batch.y.view(-1) - acceptance["target_mean_gap"]) / acceptance["target_std_gap"]
            loss = torch.nn.functional.l1_loss(pred, target)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite preflight loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            rows += batch.y.numel()
        torch.cuda.synchronize()
        seconds = time.monotonic() - started
        projected = seconds / rows * acceptance["counts"]["train"] * 1.20 * config.epochs
        headroom = 1 - torch.cuda.max_memory_reserved() / torch.cuda.get_device_properties(0).total_memory
        reports[arm] = {"sample_seconds": seconds, "sample_rows": rows,
                        "projected_training_seconds": projected, "memory_headroom": headroom,
                        "parameters": sum(p.numel() for p in model.parameters())}
        atomic_json(output.with_name("preflight_progress.json"), reports)
        print(f"preflight {arm}: projected={projected / 3600:.2f}h headroom={headroom:.3f}", flush=True)
        del model, optimizer, loss, pred, batch
        gc.collect()
        torch.cuda.empty_cache()
    passed = all(r["projected_training_seconds"] <= config.max_training_seconds and r["memory_headroom"] >= .15 for r in reports.values())
    result = {"status": "accepted" if passed else "rejected", "config": asdict(config),
              "audit_sha256": audit_hash, "code_sha256": sha256_file(Path(__file__)),
              "acceptance_sha256": ACCEPTANCE_SHA, "initial_difference": difference, "arms": reports}
    atomic_json(output, result)
    if not passed:
        raise RuntimeError("P1 timing/memory gate failed; scientific contract unchanged")
    return result


def validate_resume(state, identity):
    if state["identity"] != identity:
        raise RuntimeError("Scratch resume identity changed")


def train(root: Path, acceptance_path: Path, preflight_path: Path, output: Path, arm: str):
    config = config_for(arm)
    acceptance = checked_acceptance(acceptance_path)
    report = json.loads(preflight_path.read_text())
    if (report["status"] != "accepted" or report["config"] != asdict(config)
            or report["code_sha256"] != sha256_file(Path(__file__)) or arm not in report["arms"]):
        raise RuntimeError("Missing or changed paired scratch preflight")
    identity = {"arm": arm, "config": asdict(config), "preflight_sha256": sha256_file(preflight_path),
                "acceptance_sha256": ACCEPTANCE_SHA, "code_sha256": sha256_file(Path(__file__))}
    device = _device()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "completion_manifest.json").exists():
        completion = json.loads((output / "completion_manifest.json").read_text())
        if completion["identity"] != identity:
            raise RuntimeError("Completed identity differs")
        for filename, digest in completion["hashes"].items():
            if sha256_file(output / filename) != digest:
                raise RuntimeError("Completed artifact changed")
        return json.loads((output / "metrics.json").read_text())
    model = make_matched_model(arm, config.seed).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=config.minimum_learning_rate)
    start, cursor, loss_sum, rows, best, best_epoch, log, previous_seconds = 0, 0, 0., 0, float("inf"), -1, [], 0.
    started = time.monotonic()
    last = output / "last.pt"
    if last.exists():
        state = torch.load(last, map_location="cpu", weights_only=False)
        validate_resume(state, identity)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        restore_rng(state["rng"])
        start, cursor = state["epoch"], state["next_shard"]
        loss_sum, rows = state["loss_sum"], state["rows"]
        best, best_epoch, log = state["best"], state["best_epoch"], state["log"]
        previous_seconds = state["elapsed_seconds"]
        del state
    else:
        atomic_torch(output / "initial.pt", {"identity": identity, "model": model.state_dict(), "epoch": -1})
        if arm != "graphstate":
            baseline = evaluate(model, root, acceptance, arm, output / "validation_initial", config.batch_size)
            atomic_json(output / "initial_metrics.json", {"mae_eV": baseline, "epoch": -1})

    def save(epoch, next_shard):
        atomic_torch(last, {"identity": identity, "model": model.state_dict(),
                           "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                           "rng": capture_rng(), "epoch": epoch, "next_shard": next_shard,
                           "loss_sum": loss_sum, "rows": rows, "best": best,
                           "best_epoch": best_epoch, "log": log,
                           "elapsed_seconds": previous_seconds + time.monotonic() - started})
        atomic_json(output / "progress.json", {"status": "training", "epoch": epoch,
                                               "next_shard": next_shard, "rows": rows,
                                               "best_epoch": best_epoch, "best_mae_eV": best if math.isfinite(best) else None})

    for epoch in range(start, config.epochs):
        epoch_start = time.monotonic()
        records = _records(acceptance, "train")
        random.Random(config.seed + epoch).shuffle(records)
        model.train()
        for position in range(cursor, len(records)):
            # Per-shard seeds make restarts independent of loader reconstruction.
            shard_seed = config.seed + epoch * 1000 + position
            seed_all(shard_seed)
            dataset = _load(root, records[position])
            generator = torch.Generator().manual_seed(shard_seed)
            loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, generator=generator, num_workers=0)
            for batch in loader:
                batch = batch.to(device)
                optimizer.zero_grad(set_to_none=True)
                pred = forward(model, batch, arm)
                target = (batch.y.view(-1) - acceptance["target_mean_gap"]) / acceptance["target_std_gap"]
                loss = torch.nn.functional.l1_loss(pred, target)
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite scratch training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                loss_sum += float(loss.detach()) * target.numel()
                rows += target.numel()
            del loader, dataset
            gc.collect()
            save(epoch, position + 1)
            print(f"{arm} ep{epoch:02d} shard={position + 1}/{len(records)} rows={rows}", flush=True)
            if time.monotonic() - started >= config.job_seconds:
                raise TimeoutError("Durable shard checkpoint saved; explicit resume required")
            if previous_seconds + time.monotonic() - started >= config.max_training_seconds:
                raise TimeoutError("Cumulative 12-hour budget exhausted; checkpoint retained")
        if rows != acceptance["counts"]["train"]:
            raise RuntimeError("Training row accounting differs")
        mae = None
        if arm != "graphstate" or epoch == config.epochs - 1:
            mae = evaluate(model, root, acceptance, arm, output / f"validation_ep{epoch:02d}", config.batch_size)
        if mae is not None and mae < best:
            best, best_epoch = mae, epoch
            atomic_torch(output / "best.pt", {"identity": identity, "model": model.state_dict(),
                                             "best_epoch": best_epoch, "mae_eV": best,
                                             "target_mean_gap": acceptance["target_mean_gap"],
                                             "target_std_gap": acceptance["target_std_gap"]})
        log.append({"epoch": epoch, "mae_eV": mae, "train_l1_normalized": loss_sum / rows,
                    "seconds_this_session": time.monotonic() - epoch_start, "lr": optimizer.param_groups[0]["lr"]})
        scheduler.step()
        loss_sum, rows, cursor = 0., 0, 0
        save(epoch + 1, 0)
        atomic_json(output / "train_log.json", log)
        print(f"{arm} ep{epoch:02d} valid={mae} selected_epoch={best_epoch}", flush=True)
    result = {"status": "complete", "identity": identity, "best_epoch": best_epoch,
              "best_mae_eV": best, "final_mae_eV": log[-1]["mae_eV"], "log": log,
              "runtime_seconds": previous_seconds + time.monotonic() - started,
              "selection": "fixed_final_epoch" if arm == "graphstate" else "validation_best",
              "pretrained_weights_used": False, "official_test_used": False}
    atomic_json(output / "metrics.json", result)
    atomic_json(output / "progress.json", {"status": "complete", "epoch": config.epochs})
    atomic_json(output / "completion_manifest.json", {"status": "complete", "identity": identity,
                "hashes": {name: sha256_file(output / name) for name in ("metrics.json", "best.pt", "last.pt", "initial.pt", f"validation_ep{best_epoch:02d}/manifest.json")}})
    return result
