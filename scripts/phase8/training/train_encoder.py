"""
Train one Phase 8 encoder and extract full-cache embeddings.

This is a thin Phase 8 wrapper around the reusable model classes in src/molgap.
It never writes Phase 7 checkpoints.

Usage:
  .venv\\Scripts\\python.exe scripts/phase8/training/train_encoder.py --kind gps
  .venv\\Scripts\\python.exe scripts/phase8/training/train_encoder.py --kind schnet
  .venv\\Scripts\\python.exe scripts/phase8/training/train_encoder.py --kind gps --max-samples 2000 --epochs 2
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, r2_score
from torch_geometric.loader import DataLoader

from molgap.constants import MODELS_DIR, PARAMS_GPS_2D, PARAMS_SCHNET_300K, RESULTS_DIR, SEED
from molgap.gps import GPSWrapper
from molgap.retention import retention_loss
from molgap.schnet import SchNetWrapper
from molgap.utils import ensure_dirs

PHASE8_DIR = RESULTS_DIR / "phase8"
GRAPH_2D = PHASE8_DIR / "pyg_2d_graphs_bond_replacement_300k.pt"
GRAPH_3D = PHASE8_DIR / "pyg_3d_graphs_etkdg_replacement_300k.pt"

TRAIN_PARAMS = {
    "gps": {
        **PARAMS_GPS_2D,
        "lr": 0.0004754654349367296,
        "weight_decay": 1.3094136884618282e-05,
        "batch_size": 256,
        "scheduler": "cosine",
    },
    "schnet": {
        **PARAMS_SCHNET_300K,
        "lr": 0.00021150972021685588,
        "weight_decay": 1.4656553886225336e-05,
        "batch_size": 128,
        "scheduler": "cosine",
    },
}


def _model_params(kind: str, args) -> dict:
    if kind == "gps":
        params = dict(PARAMS_GPS_2D)
        overrides = {
            "hidden_channels": args.hidden_channels,
            "num_layers": args.num_layers,
            "num_heads": args.num_heads,
            "dropout": args.dropout,
            "pooling": args.pooling,
        }
    else:
        params = dict(PARAMS_SCHNET_300K)
        overrides = {
            "hidden_channels": args.hidden_channels,
            "num_filters": args.num_filters,
            "num_interactions": args.num_interactions,
            "num_gaussians": args.num_gaussians,
            "cutoff": args.cutoff,
            "dropout": args.dropout,
        }
    params.update({key: value for key, value in overrides.items() if value is not None})
    return params


def _make_model(kind: str, model_params: dict):
    if kind == "gps":
        return GPSWrapper(**model_params)
    if kind == "schnet":
        return SchNetWrapper(**model_params, use_charges=True)
    raise ValueError(kind)


def _load_compatible_state(model, checkpoint: Path, device) -> dict:
    source = torch.load(checkpoint, weights_only=True, map_location=device)
    target = model.state_dict()
    compatible = {
        key: value for key, value in source.items()
        if key in target and target[key].shape == value.shape
    }
    model.load_state_dict(compatible, strict=False)
    return {
        "loaded_tensors": len(compatible),
        "source_tensors": len(source),
        "target_tensors": len(target),
        "missing_tensors": sorted(set(target) - set(compatible)),
        "skipped_tensors": sorted(set(source) - set(compatible)),
    }


def _forward(kind: str, model, batch):
    if kind == "gps":
        return model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
    charges = batch.charges if hasattr(batch, "charges") else None
    return model(batch.z, batch.pos, batch.batch, charges=charges)


def _encode(kind: str, model, batch):
    if kind == "gps":
        return model.encode(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
    charges = batch.charges if hasattr(batch, "charges") else None
    return model.encode(batch.z, batch.pos, batch.batch, charges=charges)


def _make_train_loader(train_set, batch_size: int, replay_boundary: int | None,
                       replay_weight: float, seed: int):
    """Optionally oversample an older source-index prefix for replay experiments."""
    if replay_boundary is None:
        return DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0), None
    if replay_weight <= 0.0:
        raise ValueError("--replay-weight must be positive")
    old = np.asarray([
        int(graph.source_idx.view(-1)[0]) < replay_boundary
        for graph in train_set
    ])
    if not old.any() or old.all():
        raise ValueError("Replay boundary must split the training set into old and new rows")
    weights = torch.as_tensor(np.where(old, replay_weight, 1.0), dtype=torch.double)
    sampler = torch.utils.data.WeightedRandomSampler(
        weights,
        num_samples=len(train_set),
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    old_probability = float(weights[old].sum().item() / weights.sum().item())
    return (
        DataLoader(train_set, batch_size=batch_size, sampler=sampler, num_workers=0),
        {
            "source_idx_lt": int(replay_boundary),
            "old_train_rows": int(old.sum()),
            "new_train_rows": int((~old).sum()),
            "old_weight": float(replay_weight),
            "expected_old_draw_fraction": old_probability,
        },
    )


def _explicit_split(graphs, split_csv: Path):
    rows = []
    with split_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not {"source_idx", "split"}.issubset(reader.fieldnames or ()):
            raise ValueError("Explicit split CSV needs source_idx and split columns")
        rows = [
            (int(row["source_idx"]), row["split"].strip().lower())
            for row in reader
        ]
    if not rows:
        raise ValueError("Explicit split CSV is empty")
    source_indices = [value for value, _ in rows]
    if len(source_indices) != len(set(source_indices)):
        raise ValueError("Explicit split CSV contains duplicate source_idx values")
    allowed = {"train", "validation", "test"}
    unknown = sorted(set(role for _, role in rows) - allowed)
    if unknown:
        raise ValueError(f"Unknown explicit split roles: {unknown}")

    graph_map = {
        int(graph.source_idx.view(-1)[0]): graph
        for graph in graphs
    }
    if len(graph_map) != len(graphs):
        raise ValueError("Graph cache contains duplicate source_idx values")
    missing = [value for value in source_indices if value not in graph_map]
    if missing:
        raise ValueError(
            f"Explicit split references {len(missing)} unavailable graph rows"
        )
    split_sets = {
        role: [graph_map[value] for value, assigned in rows if assigned == role]
        for role in ("train", "validation", "test")
    }
    if any(not values for values in split_sets.values()):
        raise ValueError("Explicit split must contain train, validation, and test rows")
    return split_sets, {
        "path": str(split_csv),
        "sha256": _sha256(split_csv),
        "rows": {role: len(values) for role, values in split_sets.items()},
    }


@torch.no_grad()
def _evaluate(kind: str, model, loader, device):
    model.eval()
    pred, true = [], []
    for batch in loader:
        batch = batch.to(device)
        out = _forward(kind, model, batch)
        pred.append(out.float().cpu().numpy())
        true.append(batch.y.float().cpu().numpy())
    return np.concatenate(pred), np.concatenate(true)


def _metrics(pred, true):
    out = {}
    for i, name in enumerate(["HOMO", "LUMO", "Gap"]):
        out[name] = {
            "mae": float(mean_absolute_error(true[:, i], pred[:, i])),
            "r2": float(r2_score(true[:, i], pred[:, i])),
        }
    out["average"] = {
        "mae": float(np.mean([out[k]["mae"] for k in ["HOMO", "LUMO", "Gap"]])),
        "r2": float(np.mean([out[k]["r2"] for k in ["HOMO", "LUMO", "Gap"]])),
    }
    return out


def _extract_embeddings(kind: str, model, graphs, device, out_path: Path, batch_size: int):
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False, num_workers=0)
    n_batches = (len(graphs) + batch_size - 1) // batch_size
    embs, source_idx = [], []
    model.eval()
    t0 = time.time()
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            batch = batch.to(device)
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                emb = _encode(kind, model, batch)
            embs.append(emb.float().cpu())
            source_idx.append(batch.source_idx.view(-1).cpu())
            if bi % 20 == 0 or bi == n_batches - 1:
                print(f"  embed batch {bi + 1}/{n_batches} ({time.time() - t0:.0f}s)",
                      flush=True)
    payload = {
        "embeddings": torch.cat(embs, dim=0),
        "source_idx": torch.cat(source_idx, dim=0),
    }
    torch.save(payload, out_path)
    print(f"Embeddings -> {out_path} {tuple(payload['embeddings'].shape)}", flush=True)


def _atomic_torch_save(value: object, path: Path) -> None:
    ensure_dirs(path.parent)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _atomic_json_write(value: dict, path: Path) -> None:
    ensure_dirs(path.parent)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@torch.no_grad()
def _load_or_build_retention_targets(
    teacher,
    kind: str,
    graphs,
    *,
    boundary: int,
    teacher_sha256: str,
    cache_path: Path,
    batch_size: int,
    device,
) -> torch.Tensor:
    if cache_path.is_file():
        payload = torch.load(cache_path, weights_only=False, map_location="cpu")
        targets = payload.get("targets")
        if (
            payload.get("teacher_sha256") == teacher_sha256
            and int(payload.get("source_idx_lt", -1)) == boundary
            and isinstance(targets, torch.Tensor)
            and tuple(targets.shape) == (boundary, 3)
            and torch.isfinite(targets).all()
        ):
            print(f"Reused retention targets from {cache_path}", flush=True)
            return targets

    if boundary <= 0 or boundary > len(graphs):
        raise ValueError("Retention boundary must fall within the graph cache")
    targets = torch.empty((boundary, 3), dtype=torch.float32)
    loader = DataLoader(
        graphs[:boundary],
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    teacher.eval()
    for batch_number, batch in enumerate(loader):
        indices = batch.source_idx.view(-1).long()
        if int(indices.min()) < 0 or int(indices.max()) >= boundary:
            raise ValueError("Retention prefix contains out-of-range source_idx")
        batch = batch.to(device)
        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
            prediction = _forward(kind, teacher, batch).float().cpu()
        targets[indices] = prediction
        if batch_number % 200 == 0:
            print(f"  retention target batch {batch_number + 1}", flush=True)
    if not torch.isfinite(targets).all():
        raise ValueError("Retention target cache contains non-finite values")
    _atomic_torch_save(
        {
            "format": "molgap-retention-targets-v1",
            "teacher_sha256": teacher_sha256,
            "source_idx_lt": boundary,
            "targets": targets,
        },
        cache_path,
    )
    print(f"Retention targets -> {cache_path} {tuple(targets.shape)}", flush=True)
    return targets


def _load_graph_manifest(path: Path) -> tuple[dict, list[dict]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("format") != "molgap-pyg-shards-v1" or not manifest.get("complete"):
        raise ValueError(f"Incomplete or unsupported graph manifest: {path}")
    shards = list(manifest.get("shards", []))
    total = int(manifest.get("total_graphs", 0))
    expected_start = 0
    for entry in shards:
        start = int(entry["source_idx_start"])
        end = int(entry["source_idx_end"])
        count = int(entry["n_graphs"])
        if start != expected_start or end - start != count:
            raise ValueError(f"Non-contiguous shard entry: {entry}")
        if not Path(entry["path"]).is_file():
            raise FileNotFoundError(entry["path"])
        expected_start = end
    if not shards or expected_start != total:
        raise ValueError(f"Manifest covers {expected_start:,} of {total:,} graphs")
    return manifest, shards


def _split_roles(n_graphs: int, seed: int) -> np.ndarray:
    permutation = np.random.RandomState(seed).permutation(n_graphs)
    n_train, n_val = int(0.8 * n_graphs), int(0.1 * n_graphs)
    roles = np.empty(n_graphs, dtype=np.uint8)
    roles[permutation[:n_train]] = 0
    roles[permutation[n_train:n_train + n_val]] = 1
    roles[permutation[n_train + n_val:]] = 2
    return roles


def _load_manifest_shard(entry: dict):
    path = Path(entry["path"])
    graphs = torch.load(path, weights_only=False)
    count = int(entry["n_graphs"])
    start = int(entry["source_idx_start"])
    end = int(entry["source_idx_end"])
    if len(graphs) != count:
        raise ValueError(f"{path}: expected {count:,} graphs, found {len(graphs):,}")
    if count:
        first = int(graphs[0].source_idx.view(-1)[0])
        last = int(graphs[-1].source_idx.view(-1)[0])
        if first != start or last != end - 1:
            raise ValueError(f"{path}: source_idx endpoints {first}..{last}, expected {start}..{end - 1}")
    return graphs


def _local_split_indices(entry: dict, roles: np.ndarray, role: int) -> np.ndarray:
    start = int(entry["source_idx_start"])
    end = int(entry["source_idx_end"])
    return np.flatnonzero(roles[start:end] == role)


def _release(*values) -> None:
    for value in values:
        del value
    gc.collect()


@torch.no_grad()
def _evaluate_shards(kind: str, model, shards: list[dict], roles: np.ndarray,
                     role: int, batch_size: int, device, *, predictions: bool = False):
    model.eval()
    total, count = 0.0, 0
    pred, true = [], []
    criterion = nn.L1Loss()
    for shard_number, entry in enumerate(shards):
        graphs = _load_manifest_shard(entry)
        indices = _local_split_indices(entry, roles, role)
        subset = [graphs[int(i)] for i in indices]
        loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0)
        for batch in loader:
            batch = batch.to(device)
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                output = _forward(kind, model, batch)
                loss = criterion(output, batch.y)
            total += loss.item() * batch.num_graphs
            count += batch.num_graphs
            if predictions:
                pred.append(output.float().cpu().numpy())
                true.append(batch.y.float().cpu().numpy())
        print(f"  eval shard {shard_number + 1}/{len(shards)} rows={len(subset):,}", flush=True)
        del loader, subset, indices, graphs
        gc.collect()
    if predictions:
        return np.concatenate(pred), np.concatenate(true)
    return total / max(count, 1), count


def _extract_embeddings_sharded(kind: str, model, shards: list[dict], device,
                                out_path: Path, batch_size: int, model_sha256: str) -> None:
    parts_dir = out_path.parent / f"{out_path.stem}_parts"
    ensure_dirs(parts_dir)
    part_paths = []
    model.eval()
    for shard_number, entry in enumerate(shards):
        part_path = parts_dir / f"part-{shard_number:03d}.pt"
        expected_start = int(entry["source_idx_start"])
        expected_end = int(entry["source_idx_end"])
        if part_path.is_file():
            payload = torch.load(part_path, weights_only=False, map_location="cpu")
            valid = (
                payload.get("model_sha256") == model_sha256
                and len(payload.get("source_idx", [])) == expected_end - expected_start
                and int(payload["source_idx"][0]) == expected_start
                and int(payload["source_idx"][-1]) == expected_end - 1
            )
            if valid:
                print(f"Reused embedding part {part_path}", flush=True)
                part_paths.append(part_path)
                continue
        graphs = _load_manifest_shard(entry)
        loader = DataLoader(graphs, batch_size=batch_size, shuffle=False, num_workers=0)
        embeddings, source_indices = [], []
        started = time.time()
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                    embeddings.append(_encode(kind, model, batch).float().cpu())
                source_indices.append(batch.source_idx.view(-1).cpu())
        payload = {
            "embeddings": torch.cat(embeddings),
            "source_idx": torch.cat(source_indices),
            "model_sha256": model_sha256,
        }
        _atomic_torch_save(payload, part_path)
        print(
            f"Embedding part {shard_number + 1}/{len(shards)} -> {part_path} "
            f"{tuple(payload['embeddings'].shape)} ({time.time() - started:.0f}s)",
            flush=True,
        )
        part_paths.append(part_path)
        del payload, embeddings, source_indices, loader, graphs
        gc.collect()

    first = torch.load(part_paths[0], weights_only=False, map_location="cpu")
    total = int(shards[-1]["source_idx_end"])
    embeddings = torch.empty((total, first["embeddings"].shape[1]), dtype=first["embeddings"].dtype)
    source_indices = torch.arange(total, dtype=torch.long)
    del first
    for part_path in part_paths:
        payload = torch.load(part_path, weights_only=False, map_location="cpu")
        indices = payload["source_idx"].long()
        embeddings[indices] = payload["embeddings"]
        del payload, indices
    _atomic_torch_save({"embeddings": embeddings, "source_idx": source_indices}, out_path)
    print(f"Embeddings -> {out_path} {tuple(embeddings.shape)}", flush=True)


def _run_sharded(args, device, model_params: dict, training_params: dict,
                  model_out: Path, metrics_out: Path, embeddings_out: Path) -> None:
    if (
        args.max_samples is not None
        or args.replay_boundary is not None
        or args.retention_teacher is not None
    ):
        raise ValueError(
            "Sharded mode does not support --max-samples, replay sampling, "
            "or retention distillation"
        )
    manifest, shards = _load_graph_manifest(args.graph_manifest)
    n_graphs = int(manifest["total_graphs"])
    manifest_sha256 = _sha256(args.graph_manifest)
    roles = _split_roles(n_graphs, args.split_seed)
    split_counts = [int((roles == role).sum()) for role in range(3)]
    print(
        f"Loaded manifest {args.graph_manifest}: {len(shards)} shards, {n_graphs:,} graphs; "
        f"split={split_counts[0]:,}/{split_counts[1]:,}/{split_counts[2]:,}",
        flush=True,
    )
    model = _make_model(args.kind, model_params).to(device)
    eval_bs = args.eval_batch_size or int(training_params["batch_size"])
    embed_bs = args.embedding_batch_size or int(training_params["batch_size"])

    if args.extract_only or args.postprocess_only:
        model.load_state_dict(torch.load(model_out, weights_only=True, map_location=device))
        if args.postprocess_only:
            prediction, target = _evaluate_shards(
                args.kind, model, shards, roles, 2, eval_bs, device, predictions=True
            )
            _atomic_json_write({
                "kind": args.kind,
                "graph_manifest": str(args.graph_manifest),
                "n_graphs": n_graphs,
                "postprocess_only": True,
                "model_path": str(model_out),
                "test_metrics": _metrics(prediction, target),
            }, metrics_out)
        if not args.no_embeddings:
            _extract_embeddings_sharded(
                args.kind, model, shards, device, embeddings_out, embed_bs, _sha256(model_out)
            )
        return

    init_report = None
    if args.init_from is not None:
        if args.init_compatible:
            init_report = _load_compatible_state(model, args.init_from, device)
        else:
            model.load_state_dict(torch.load(args.init_from, weights_only=True, map_location=device))
        print(f"Warm-started from {args.init_from}", flush=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=training_params["lr"], weight_decay=training_params["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())
    criterion = nn.L1Loss()
    batch_size = int(training_params["batch_size"])
    best_val, best_state, best_epoch, wait = float("inf"), None, -1, 0
    log_rows, start_epoch = [], 0
    if args.resume_from is not None:
        resume = torch.load(args.resume_from, weights_only=False, map_location=device)
        if (
            resume.get("kind") != args.kind
            or resume.get("graph_manifest_sha256") != manifest_sha256
            or resume.get("n_graphs") != n_graphs
            or int(resume.get("seed", args.seed)) != args.seed
            or int(resume.get("split_seed", args.split_seed)) != args.split_seed
        ):
            raise ValueError(
                "Resume checkpoint does not match kind, graph manifest, or seeds"
            )
        model.load_state_dict(resume["model_state"])
        optimizer.load_state_dict(resume["optimizer_state"])
        scheduler.load_state_dict(resume["scheduler_state"])
        scaler.load_state_dict(resume["scaler_state"])
        best_val, best_state = float(resume["best_val"]), resume["best_state"]
        best_epoch, wait = int(resume["best_epoch"]), int(resume["wait"])
        log_rows, start_epoch = list(resume["log"]), int(resume["next_epoch"])
        print(f"Resuming from {args.resume_from}: epoch {start_epoch}", flush=True)

    for epoch in range(start_epoch, args.epochs):
        started = time.time()
        model.train()
        total, count = 0.0, 0
        shard_order = np.random.RandomState(args.seed + epoch).permutation(len(shards))
        for order_position, shard_number in enumerate(shard_order):
            entry = shards[int(shard_number)]
            graphs = _load_manifest_shard(entry)
            indices = _local_split_indices(entry, roles, 0)
            subset = [graphs[int(i)] for i in indices]
            generator = torch.Generator().manual_seed(
                args.seed + epoch * len(shards) + int(shard_number)
            )
            loader = DataLoader(
                subset, batch_size=batch_size, shuffle=True, num_workers=0, generator=generator
            )
            for batch in loader:
                batch = batch.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                    loss = criterion(_forward(args.kind, model, batch), batch.y)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                scaler.step(optimizer)
                scaler.update()
                total += loss.item() * batch.num_graphs
                count += batch.num_graphs
            print(
                f"  train shard {order_position + 1}/{len(shards)} rows={len(subset):,}",
                flush=True,
            )
            del loader, subset, indices, graphs
            gc.collect()
        train_loss = total / max(count, 1)
        val_loss, val_count = _evaluate_shards(
            args.kind, model, shards, roles, 1, eval_bs, device
        )
        scheduler.step()
        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_epoch, wait = epoch, 0
        else:
            wait += 1
        elapsed = time.time() - started
        log_rows.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_mae": val_loss,
            "lr": optimizer.param_groups[0]["lr"],
            "time_s": elapsed,
        })
        print(
            f"ep{epoch:03d} train={train_loss:.4f} val={val_loss:.4f} "
            f"best={best_val:.4f}@{best_epoch} lr={optimizer.param_groups[0]['lr']:.2e} "
            f"{elapsed:.1f}s{' *' if improved else ''}",
            flush=True,
        )
        if args.checkpoint_out is not None and (epoch + 1) % args.checkpoint_every == 0:
            checkpoint = {
                "kind": args.kind,
                "graph_manifest": str(args.graph_manifest),
                "graph_manifest_sha256": manifest_sha256,
                "n_graphs": n_graphs,
                "seed": int(args.seed),
                "split_seed": int(args.split_seed),
                "next_epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "scaler_state": scaler.state_dict(),
                "best_val": best_val,
                "best_state": best_state,
                "best_epoch": best_epoch,
                "wait": wait,
                "log": log_rows,
            }
            _atomic_torch_save(checkpoint, args.checkpoint_out)
            _atomic_json_write({
                "complete": False,
                "kind": args.kind,
                "graph_manifest": str(args.graph_manifest),
                "n_graphs": n_graphs,
                "seed": int(args.seed),
                "split_seed": int(args.split_seed),
                "next_epoch": epoch + 1,
                "best_val_mae": best_val,
                "best_epoch": best_epoch,
                "log": log_rows,
            }, metrics_out)
        if wait >= args.patience:
            print(f"Early stop at epoch {epoch}", flush=True)
            break

    if best_state is None:
        raise RuntimeError("No valid checkpoint state was produced")
    model.load_state_dict(best_state)
    _atomic_torch_save(best_state, model_out)
    prediction, target = _evaluate_shards(
        args.kind, model, shards, roles, 2, eval_bs, device, predictions=True
    )
    result = {
        "kind": args.kind,
        "graph_manifest": str(args.graph_manifest),
        "graph_manifest_sha256": manifest_sha256,
        "n_graphs": n_graphs,
        "shards": len(shards),
        "seed": int(args.seed),
        "split": {
            "seed": int(args.split_seed),
            "train": split_counts[0],
            "val": split_counts[1],
            "test": split_counts[2],
        },
        "best_val_mae": float(best_val),
        "best_epoch": int(best_epoch),
        "init_from": str(args.init_from) if args.init_from is not None else None,
        "init_compatible": bool(args.init_compatible),
        "init_report": init_report,
        "model_params": model_params,
        "params": training_params,
        "test_metrics": _metrics(prediction, target),
        "log": log_rows,
    }
    _atomic_json_write(result, metrics_out)
    print(f"Model -> {model_out}\nMetrics -> {metrics_out}", flush=True)
    if not args.no_embeddings:
        _extract_embeddings_sharded(
            args.kind, model, shards, device, embeddings_out, embed_bs, _sha256(model_out)
        )


def main():
    parser = argparse.ArgumentParser(description="Train a Phase 8 encoder")
    parser.add_argument("--kind", choices=["gps", "schnet"], required=True)
    parser.add_argument("--graphs", type=Path, default=None)
    parser.add_argument("--graph-manifest", type=Path, default=None,
                        help="stream a molgap-pyg-shards-v1 manifest instead of one graph cache")
    parser.add_argument("--model-out", type=Path, default=None)
    parser.add_argument("--metrics-out", type=Path, default=None)
    parser.add_argument("--embeddings-out", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--seed", type=int, default=SEED,
                        help="model, sampler, and training-order seed")
    parser.add_argument("--split-seed", type=int, default=SEED,
                        help="fixed train/validation/test partition seed")
    parser.add_argument(
        "--split-csv",
        type=Path,
        default=None,
        help="explicit source_idx,split CSV; overrides the random 80/10/10 split",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--no-embeddings", action="store_true")
    parser.add_argument("--extract-only", action="store_true",
                        help="load --model-out and only write --embeddings-out")
    parser.add_argument("--postprocess-only", action="store_true",
                        help="load --model-out, evaluate deterministic test split, and write embeddings")
    parser.add_argument("--init-from", type=Path, default=None,
                        help="optional same-architecture checkpoint for warm-starting")
    parser.add_argument("--init-compatible", action="store_true",
                        help="load only same-name, same-shape tensors (for architecture expansion)")
    parser.add_argument("--hidden-channels", type=int, default=None)
    parser.add_argument("--num-filters", type=int, default=None,
                        help="SchNet filter width override")
    parser.add_argument("--num-interactions", type=int, default=None,
                        help="SchNet interaction block count override")
    parser.add_argument("--num-gaussians", type=int, default=None,
                        help="SchNet radial basis count override")
    parser.add_argument("--cutoff", type=float, default=None,
                        help="SchNet distance cutoff override")
    parser.add_argument("--num-layers", type=int, default=None,
                        help="GPS layer count override")
    parser.add_argument("--num-heads", type=int, default=None,
                        help="GPS attention head count override")
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--pooling", choices=["mean", "mean_max"], default=None,
                        help="GPS molecule-level pooling override")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--embedding-batch-size", type=int, default=None)
    parser.add_argument("--replay-boundary", type=int, default=None,
                        help="source_idx below this value is the replay pool; disabled by default")
    parser.add_argument("--replay-weight", type=float, default=1.0,
                        help="relative sampling weight for rows below --replay-boundary")
    parser.add_argument("--retention-teacher", type=Path, default=None,
                        help="frozen same-architecture teacher for old-prefix retention")
    parser.add_argument("--retention-boundary", type=int, default=None,
                        help="apply retention distillation below this source_idx")
    parser.add_argument("--retention-weight", type=float, default=0.0,
                        help="additive weight for old-prefix teacher L1 loss")
    parser.add_argument("--retention-targets-cache", type=Path, default=None,
                        help="atomic cache for old-prefix teacher predictions")
    parser.add_argument("--checkpoint-out", type=Path, default=None,
                        help="atomic resumable training state")
    parser.add_argument("--checkpoint-every", type=int, default=1,
                        help="persist --checkpoint-out every N completed epochs")
    parser.add_argument("--resume-from", type=Path, default=None,
                        help="resume from a checkpoint created by --checkpoint-out")
    args = parser.parse_args()

    if args.graph_manifest is not None and args.graphs is not None:
        parser.error("--graphs and --graph-manifest are mutually exclusive")
    if args.graph_manifest is not None and args.split_csv is not None:
        parser.error("--split-csv is not supported with --graph-manifest")
    retention_values = (
        args.retention_teacher,
        args.retention_boundary,
        args.retention_weight > 0.0,
    )
    if any(retention_values) and not all(retention_values):
        parser.error(
            "--retention-teacher, --retention-boundary, and a positive "
            "--retention-weight must be provided together"
        )
    if args.retention_teacher is not None and args.kind != "gps":
        parser.error("retention distillation currently supports --kind gps only")
    if args.retention_targets_cache is not None and args.retention_teacher is None:
        parser.error("--retention-targets-cache requires --retention-teacher")

    ensure_dirs(PHASE8_DIR, MODELS_DIR)
    graph_path = args.graphs or (GRAPH_2D if args.kind == "gps" else GRAPH_3D)
    model_out = args.model_out or MODELS_DIR / f"phase8_{args.kind}_replacement_300k.pt"
    metrics_out = args.metrics_out or PHASE8_DIR / f"{args.kind}_replacement_300k_metrics.json"
    embeddings_out = args.embeddings_out or PHASE8_DIR / f"{args.kind}_replacement_300k_embeddings.pt"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | kind={args.kind}", flush=True)
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"{props.name} | {props.total_memory / 1e9:.1f} GB", flush=True)

    model_params = _model_params(args.kind, args)
    p = dict(TRAIN_PARAMS[args.kind])
    p.update(model_params)
    if args.lr is not None:
        p["lr"] = args.lr
    if args.weight_decay is not None:
        p["weight_decay"] = args.weight_decay
    if args.batch_size is not None:
        p["batch_size"] = args.batch_size
    print(f"Model params: {model_params}", flush=True)
    print(
        f"Training params: lr={p['lr']:.3g} weight_decay={p['weight_decay']:.3g} "
        f"batch_size={p['batch_size']}",
        flush=True,
    )
    eval_bs = args.eval_batch_size or int(p["batch_size"])
    embed_bs = args.embedding_batch_size or int(p["batch_size"])

    if args.graph_manifest is not None:
        _run_sharded(args, device, model_params, p, model_out, metrics_out, embeddings_out)
        return

    graphs = torch.load(graph_path, weights_only=False)
    if args.max_samples is not None:
        graphs = graphs[:args.max_samples]
    print(f"Loaded {len(graphs)} graphs from {graph_path}", flush=True)

    if args.extract_only:
        model = _make_model(args.kind, model_params).to(device)
        model.load_state_dict(torch.load(model_out, weights_only=False, map_location=device))
        _extract_embeddings(args.kind, model, graphs, device, embeddings_out, embed_bs)
        return

    if args.postprocess_only:
        if args.split_csv is not None:
            split_sets, split_contract = _explicit_split(graphs, args.split_csv)
            test_set = split_sets["test"]
        else:
            idx = np.random.RandomState(args.split_seed).permutation(len(graphs))
            n_train, n_val = int(0.8 * len(graphs)), int(0.1 * len(graphs))
            test_set = [graphs[i] for i in idx[n_train + n_val:]]
            split_contract = {
                "kind": "random_80_10_10",
                "seed": int(args.split_seed),
            }
        model = _make_model(args.kind, model_params).to(device)
        model.load_state_dict(torch.load(model_out, weights_only=False, map_location=device))
        test_loader = DataLoader(test_set, batch_size=eval_bs, shuffle=False, num_workers=0)
        pred, true = _evaluate(args.kind, model, test_loader, device)
        result = {
            "kind": args.kind,
            "graph_path": str(graph_path),
            "n_graphs": len(graphs),
            "postprocess_only": True,
            "model_path": str(model_out),
            "eval_batch_size": eval_bs,
            "embedding_batch_size": embed_bs,
            "split_contract": split_contract,
            "test_metrics": _metrics(pred, true),
        }
        _atomic_json_write(result, metrics_out)
        print(f"Metrics -> {metrics_out}", flush=True)
        if not args.no_embeddings:
            _extract_embeddings(args.kind, model, graphs, device, embeddings_out, embed_bs)
        return

    if args.split_csv is not None:
        split_sets, split_contract = _explicit_split(graphs, args.split_csv)
        train_set = split_sets["train"]
        val_set = split_sets["validation"]
        test_set = split_sets["test"]
    else:
        idx = np.random.RandomState(args.split_seed).permutation(len(graphs))
        n_train, n_val = int(0.8 * len(graphs)), int(0.1 * len(graphs))
        train_set = [graphs[i] for i in idx[:n_train]]
        val_set = [graphs[i] for i in idx[n_train:n_train + n_val]]
        test_set = [graphs[i] for i in idx[n_train + n_val:]]
        split_contract = {
            "kind": "random_80_10_10",
            "seed": int(args.split_seed),
        }
    print(f"Split: train={len(train_set)} val={len(val_set)} test={len(test_set)}", flush=True)

    model = _make_model(args.kind, model_params).to(device)
    retention_teacher = None
    retention_targets = None
    retention_config = None
    if args.retention_teacher is not None:
        retention_teacher = _make_model(args.kind, model_params).to(device)
        retention_teacher.load_state_dict(
            torch.load(args.retention_teacher, weights_only=True, map_location=device)
        )
        retention_teacher.eval()
        retention_teacher.requires_grad_(False)
        retention_config = {
            "teacher": str(args.retention_teacher),
            "teacher_sha256": _sha256(args.retention_teacher),
            "source_idx_lt": int(args.retention_boundary),
            "weight": float(args.retention_weight),
            "targets_cache": (
                str(args.retention_targets_cache)
                if args.retention_targets_cache is not None
                else None
            ),
        }
        print(f"Retention distillation: {retention_config}", flush=True)
    init_report = None
    if args.init_from is not None:
        if args.init_compatible:
            init_report = _load_compatible_state(model, args.init_from, device)
            print(
                f"Compatible warm-start from {args.init_from}: "
                f"loaded={init_report['loaded_tensors']}/{init_report['target_tensors']} "
                f"missing={len(init_report['missing_tensors'])} "
                f"skipped={len(init_report['skipped_tensors'])}",
                flush=True,
            )
        else:
            state = torch.load(args.init_from, weights_only=True, map_location=device)
            model.load_state_dict(state)
            print(f"Warm-started from {args.init_from}", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())
    criterion = nn.L1Loss()
    bs = int(p["batch_size"])
    train_loader, replay_report = _make_train_loader(
        train_set, bs, args.replay_boundary, args.replay_weight, args.seed,
    )
    if replay_report is not None:
        print(f"Replay sampling: {replay_report}", flush=True)
    val_loader = DataLoader(val_set, batch_size=eval_bs, shuffle=False, num_workers=0)
    if retention_teacher is not None and args.retention_targets_cache is not None:
        retention_targets = _load_or_build_retention_targets(
            retention_teacher,
            args.kind,
            graphs,
            boundary=int(args.retention_boundary),
            teacher_sha256=retention_config["teacher_sha256"],
            cache_path=args.retention_targets_cache,
            batch_size=eval_bs,
            device=device,
        )
        del retention_teacher
        retention_teacher = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    best_val, best_state, best_epoch, wait = float("inf"), None, -1, 0
    log_rows = []
    start_epoch = 0
    if args.resume_from is not None:
        resume = torch.load(args.resume_from, weights_only=False, map_location=device)
        if resume.get("kind") != args.kind or resume.get("graph_path") != str(graph_path) or resume.get("n_graphs") != len(graphs):
            raise ValueError("Resume checkpoint does not match kind, graph cache, or graph count")
        if (
            int(resume.get("seed", args.seed)) != args.seed
            or int(resume.get("split_seed", args.split_seed)) != args.split_seed
        ):
            raise ValueError("Resume checkpoint seed configuration differs")
        if resume.get("split_contract") != split_contract:
            raise ValueError("Resume checkpoint explicit split contract differs")
        if resume.get("replay_sampling") != replay_report:
            raise ValueError("Resume checkpoint replay configuration differs")
        if resume.get("retention_distillation") != retention_config:
            raise ValueError("Resume checkpoint retention configuration differs")
        model.load_state_dict(resume["model_state"])
        optimizer.load_state_dict(resume["optimizer_state"])
        scheduler.load_state_dict(resume["scheduler_state"])
        scaler.load_state_dict(resume["scaler_state"])
        best_val = float(resume["best_val"])
        best_state = resume["best_state"]
        best_epoch = int(resume["best_epoch"])
        wait = int(resume["wait"])
        log_rows = list(resume["log"])
        start_epoch = int(resume["next_epoch"])
        print(f"Resuming from {args.resume_from}: epoch {start_epoch}", flush=True)
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        model.train()
        total, label_total, distillation_total, retained_total, n = 0.0, 0.0, 0.0, 0, 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                prediction = _forward(args.kind, model, batch)
                if retention_teacher is None and retention_targets is None:
                    loss = criterion(prediction, batch.y)
                    label_loss = loss
                    distillation_loss = prediction.sum() * 0.0
                    retained_rows = 0
                else:
                    if retention_targets is None:
                        with torch.no_grad():
                            teacher_prediction = _forward(
                                args.kind, retention_teacher, batch
                            )
                    else:
                        source_idx = batch.source_idx.view(-1).long()
                        retained = source_idx < int(args.retention_boundary)
                        teacher_prediction = prediction.detach().clone()
                        if retained.any():
                            teacher_prediction[retained] = retention_targets[
                                source_idx[retained].cpu()
                            ].to(
                                device=device,
                                dtype=teacher_prediction.dtype,
                            )
                    losses = retention_loss(
                        prediction,
                        batch.y,
                        batch.source_idx.view(-1),
                        teacher_prediction,
                        boundary=int(args.retention_boundary),
                        distillation_weight=float(args.retention_weight),
                    )
                    loss = losses.total
                    label_loss = losses.label
                    distillation_loss = losses.distillation
                    retained_rows = losses.retained_rows
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            scaler.step(optimizer)
            scaler.update()
            total += loss.item() * batch.num_graphs
            label_total += label_loss.item() * batch.num_graphs
            distillation_total += distillation_loss.item() * retained_rows
            retained_total += retained_rows
            n += batch.num_graphs
        train_loss = total / max(n, 1)
        train_label_loss = label_total / max(n, 1)
        train_distillation_loss = distillation_total / max(retained_total, 1)

        model.eval()
        vtotal, vn = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                    loss = criterion(_forward(args.kind, model, batch), batch.y)
                vtotal += loss.item() * batch.num_graphs
                vn += batch.num_graphs
        val_loss = vtotal / max(vn, 1)
        scheduler.step()

        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            wait = 0
        else:
            wait += 1
        elapsed = time.time() - t0
        log_rows.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_label_mae": train_label_loss,
            "train_retention_mae": train_distillation_loss,
            "retained_draw_rows": retained_total,
            "val_mae": val_loss,
            "lr": optimizer.param_groups[0]["lr"],
            "time_s": elapsed,
        })
        mark = " *" if improved else ""
        print(f"ep{epoch:03d} train={train_loss:.4f} val={val_loss:.4f} "
              f"best={best_val:.4f}@{best_epoch} lr={optimizer.param_groups[0]['lr']:.2e} "
              f"{elapsed:.1f}s{mark}", flush=True)
        if args.checkpoint_out is not None and (epoch + 1) % args.checkpoint_every == 0:
            _atomic_torch_save({
                "kind": args.kind,
                "graph_path": str(graph_path),
                "n_graphs": len(graphs),
                "seed": int(args.seed),
                "split_seed": int(args.split_seed),
                "split_contract": split_contract,
                "next_epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "scaler_state": scaler.state_dict(),
                "best_val": best_val,
                "best_state": best_state,
                "best_epoch": best_epoch,
                "wait": wait,
                "replay_sampling": replay_report,
                "retention_distillation": retention_config,
                "log": log_rows,
            }, args.checkpoint_out)
            _atomic_json_write({
                "complete": False,
                "kind": args.kind,
                "graph_path": str(graph_path),
                "n_graphs": len(graphs),
                "seed": int(args.seed),
                "split_seed": int(args.split_seed),
                "split_contract": split_contract,
                "next_epoch": epoch + 1,
                "best_val_mae": best_val,
                "best_epoch": best_epoch,
                "replay_sampling": replay_report,
                "retention_distillation": retention_config,
                "log": log_rows,
            }, metrics_out)
        if wait >= args.patience:
            print(f"Early stop at epoch {epoch}", flush=True)
            break

    if best_state is None:
        raise RuntimeError("No valid checkpoint state was produced")
    model.load_state_dict(best_state)
    _atomic_torch_save(best_state, model_out)
    print(f"Model -> {model_out}", flush=True)

    test_loader = DataLoader(test_set, batch_size=eval_bs, shuffle=False, num_workers=0)
    pred, true = _evaluate(args.kind, model, test_loader, device)
    result = {
        "kind": args.kind,
        "graph_path": str(graph_path),
        "n_graphs": len(graphs),
        "n_params": int(sum(parameter.numel() for parameter in model.parameters())),
        "training_time_s": float(sum(row["time_s"] for row in log_rows)),
        "best_val_mae": float(best_val),
        "best_epoch": int(best_epoch),
        "seed": int(args.seed),
        "split_seed": int(args.split_seed),
        "split_contract": split_contract,
        "init_from": str(args.init_from) if args.init_from is not None else None,
        "init_compatible": bool(args.init_compatible),
        "init_report": init_report,
        "model_params": model_params,
        "params": p,
        "replay_sampling": replay_report,
        "retention_distillation": retention_config,
        "test_metrics": _metrics(pred, true),
        "log": log_rows,
    }
    _atomic_json_write(result, metrics_out)
    print(f"Metrics -> {metrics_out}", flush=True)

    if not args.no_embeddings:
        _extract_embeddings(args.kind, model, graphs, device, embeddings_out, embed_bs)


if __name__ == "__main__":
    main()
