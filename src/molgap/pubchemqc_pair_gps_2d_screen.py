"""Fair PubChemQC-100K screening for GPS11 and PairGPS2D.

The screen reuses the frozen scaffold-disjoint split from the historical
PubChemQC architecture experiment, but rebuilds both candidates' graphs under
one 18-wide pure-2D input contract.  Validation selects hyperparameters; the
test role remains unread until a separate selected-trial evaluation.
"""

from __future__ import annotations

import copy
import csv
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from .gps import GPSWrapper
from .pair_gps_2d import PairGPS2DWrapper
from .pubchemqc_pair_gps_2d import (
    ATOM_LIST,
    EDGE_DIM,
    NODE_DIM,
    TARGET_NAMES,
    _atomic_json,
    _atomic_torch_save,
    _build_parallel,
    _record_to_graph,
    _size_bucket_batches,
    sha256_file,
)


FORMAT = "molgap-pubchemqc-pair-gps-2d-screen-v1"
FROZEN_SPLIT_SHA256 = "1e6707274dd8465cfe9d96a808064372af705c4a9e4b8d20532ae6fff2cdcf05"
FROZEN_SPLIT_ROWS = {"train": 100_003, "validation": 10_000, "test": 9_997}
ROLE_INDEX = {"train": 0, "validation": 1, "test": 2}


def _read_split(split_csv: Path) -> list[dict[str, object]]:
    if sha256_file(split_csv) != FROZEN_SPLIT_SHA256:
        raise RuntimeError("PubChemQC-100K split hash differs from the frozen screen")
    with split_csv.open(newline="", encoding="utf-8") as handle:
        source = list(csv.DictReader(handle))
    required = {
        "source_idx",
        "split",
        "smiles",
        "canonical_smiles",
        "homo",
        "lumo",
        "gap",
    }
    if not source or not required <= set(source[0]):
        raise RuntimeError("PubChemQC-100K split CSV contract is incomplete")

    rows: list[dict[str, object]] = []
    counts = {role: 0 for role in ROLE_INDEX}
    seen_source_idx: set[int] = set()
    for row in source:
        source_idx = int(row["source_idx"])
        role = row["split"].strip().lower()
        if source_idx < 0 or source_idx in seen_source_idx:
            raise RuntimeError("PubChemQC-100K source_idx is negative or duplicated")
        seen_source_idx.add(source_idx)
        if role not in ROLE_INDEX:
            raise RuntimeError(f"unsupported PubChemQC-100K split role: {role}")
        smiles = row["canonical_smiles"].strip() or row["smiles"].strip()
        if not smiles:
            raise RuntimeError(f"empty SMILES at source_idx {source_idx}")
        target = tuple(float(row[name]) for name in TARGET_NAMES)
        if not np.isfinite(np.asarray(target, dtype=np.float64)).all():
            raise RuntimeError(f"non-finite target at source_idx {source_idx}")
        rows.append(
            {
                "source_idx": source_idx,
                "role": role,
                "smiles": smiles,
                "target": target,
            }
        )
        counts[role] += 1
    if counts != FROZEN_SPLIT_ROWS:
        raise RuntimeError(f"PubChemQC-100K split counts changed: {counts}")
    return rows


def build_screen_cache(
    *,
    split_csv: Path,
    output_dir: Path,
    workers: int = 14,
    shard_rows: int = 5_000,
) -> dict:
    """Build resumable 18-wide topology shards from the frozen split CSV."""
    if shard_rows < 1:
        raise ValueError("shard_rows must be positive")
    rows = _read_split(split_csv)
    output_dir.mkdir(parents=True, exist_ok=True)
    graph_dir = output_dir / "graph_shards"
    report_dir = output_dir / "reports"
    graph_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    contract = {
        "format": FORMAT,
        "split_path": str(split_csv),
        "split_sha256": FROZEN_SPLIT_SHA256,
        "source_rows": len(rows),
        "split_rows": FROZEN_SPLIT_ROWS,
        "node_feature_dim": NODE_DIM,
        "edge_feature_dim": EDGE_DIM,
        "atom_list": list(ATOM_LIST),
        "geometry": "none",
        "label_source": "frozen_pubchemqc100k_split_csv",
        "shard_rows": shard_rows,
    }
    contract_path = output_dir / "input_contract.json"
    if contract_path.exists():
        if json.loads(contract_path.read_text(encoding="utf-8")) != contract:
            raise RuntimeError("PubChemQC-100K cache input contract changed")
    else:
        _atomic_json(contract, contract_path)

    reports = []
    for start in range(0, len(rows), shard_rows):
        stop = min(start + shard_rows, len(rows))
        name = f"graphs_{start:07d}_{stop:07d}.pt"
        graph_path = graph_dir / name
        report_path = report_dir / f"{Path(name).stem}.json"
        if graph_path.exists() or report_path.exists():
            if not (graph_path.exists() and report_path.exists()):
                raise RuntimeError(f"partial cache artifact exists for {name}")
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if (
                report.get("status") != "complete"
                or report.get("format") != FORMAT
                or report.get("sha256") != sha256_file(graph_path)
            ):
                raise RuntimeError(f"existing cache shard failed verification: {name}")
            reports.append(report)
            print(f"reuse completed 100K shard {name}", flush=True)
            continue

        work = [
            (
                int(row["source_idx"]),
                str(row["smiles"]),
                tuple(row["target"]),
            )
            for row in rows[start:stop]
        ]
        graphs = []
        failures = []
        for source_idx, record, error in _build_parallel(
            work,
            workers=max(1, workers),
            progress_label=f"100K 2D {start}:{stop}",
        ):
            if record is None:
                failures.append({"source_idx": int(source_idx), "error": error})
            else:
                graphs.append(_record_to_graph(record))
        graphs.sort(key=lambda graph: int(graph.source_idx.view(-1)[0]))
        _atomic_torch_save(graphs, graph_path)
        report = {
            "status": "complete",
            "format": FORMAT,
            "start": start,
            "stop": stop,
            "requested": len(work),
            "graphs": len(graphs),
            "failed": len(failures),
            "failures": failures,
            "path": name,
            "bytes": graph_path.stat().st_size,
            "sha256": sha256_file(graph_path),
        }
        _atomic_json(report, report_path)
        reports.append(report)
        _atomic_json(
            {
                "status": "running",
                "format": FORMAT,
                "completed_shards": len(reports),
                "graphs": sum(int(item["graphs"]) for item in reports),
                "failed": sum(int(item["failed"]) for item in reports),
            },
            output_dir / "progress.json",
        )
        print(
            f"100K shard {name}: graphs={len(graphs)} failed={len(failures)}",
            flush=True,
        )

    reports.sort(key=lambda item: int(item["start"]))
    completion = {
        "status": "complete",
        "format": FORMAT,
        "source_rows": len(rows),
        "graphs": sum(int(item["graphs"]) for item in reports),
        "failed": sum(int(item["failed"]) for item in reports),
        "shards": len(reports),
        "input_contract_sha256": sha256_file(contract_path),
        "reports": reports,
    }
    _atomic_json(completion, output_dir / "build_completion.json")
    _atomic_json(completion, output_dir / "progress.json")
    return completion


def accept_screen_cache(*, split_csv: Path, cache_dir: Path) -> dict:
    """Independently verify graph identity, labels, topology, and split rows."""
    rows = _read_split(split_csv)
    completion = json.loads(
        (cache_dir / "build_completion.json").read_text(encoding="utf-8")
    )
    if completion.get("status") != "complete" or completion.get("format") != FORMAT:
        raise RuntimeError("PubChemQC-100K 2D cache build is incomplete")
    reports = completion.get("reports", [])
    expected_names = {str(item["path"]) for item in reports}
    paths = sorted((cache_dir / "graph_shards").glob("graphs_*.pt"))
    if {path.name for path in paths} != expected_names:
        raise RuntimeError("PubChemQC-100K cache shard inventory changed")

    row_by_idx = {int(row["source_idx"]): row for row in rows}
    seen: set[int] = set()
    role_counts = {role: 0 for role in ROLE_INDEX}
    train_targets = []
    shard_records = []
    report_by_name = {str(item["path"]): item for item in reports}
    for path in paths:
        report = report_by_name[path.name]
        actual_sha = sha256_file(path)
        if actual_sha != report.get("sha256"):
            raise RuntimeError(f"PubChemQC-100K cache hash mismatch: {path.name}")
        graphs = torch.load(path, map_location="cpu", weights_only=False)
        for graph in graphs:
            keys = set(graph.keys())
            if not {"x", "edge_index", "edge_attr", "y", "source_idx"} <= keys:
                raise RuntimeError(f"missing pure-2D field in {path.name}")
            if {"pos", "z", "charges"} & keys:
                raise RuntimeError(f"3D field found in {path.name}")
            source_idx = int(graph.source_idx.view(-1)[0])
            if source_idx in seen or source_idx not in row_by_idx:
                raise RuntimeError(f"invalid or duplicate source_idx {source_idx}")
            if graph.x.ndim != 2 or int(graph.x.shape[1]) != NODE_DIM:
                raise RuntimeError(f"unexpected node feature shape in {path.name}")
            if graph.edge_index.ndim != 2 or int(graph.edge_index.shape[0]) != 2:
                raise RuntimeError(f"unexpected edge index shape in {path.name}")
            if graph.edge_attr.ndim != 2 or int(graph.edge_attr.shape[1]) != EDGE_DIM:
                raise RuntimeError(f"unexpected edge feature shape in {path.name}")
            target = graph.y.view(-1, 3).view(-1).double().numpy()
            expected_target = np.asarray(row_by_idx[source_idx]["target"], dtype=np.float64)
            if not np.isfinite(target).all() or not np.allclose(
                target, expected_target, rtol=0.0, atol=2e-6
            ):
                raise RuntimeError(f"target mismatch at source_idx {source_idx}")
            role = str(row_by_idx[source_idx]["role"])
            role_counts[role] += 1
            if role == "train":
                train_targets.append(target)
            seen.add(source_idx)
        shard_records.append(
            {"path": path.name, "rows": len(graphs), "sha256": actual_sha}
        )

    if completion.get("failed") != 0 or len(seen) != len(rows):
        raise RuntimeError("the fair 100K screen requires all frozen split rows")
    if role_counts != FROZEN_SPLIT_ROWS:
        raise RuntimeError(f"accepted split counts changed: {role_counts}")
    target_array = np.asarray(train_targets, dtype=np.float64)
    result = {
        "accepted": True,
        "format": FORMAT,
        "geometry": "none",
        "source_rows": len(rows),
        "accepted_rows": len(seen),
        "expected_shards": len(paths),
        "node_feature_dim": NODE_DIM,
        "edge_feature_dim": EDGE_DIM,
        "atom_list": list(ATOM_LIST),
        "target_names": list(TARGET_NAMES),
        "target_mean": target_array.mean(axis=0).tolist(),
        "target_std": target_array.std(axis=0, ddof=1).clip(min=1e-6).tolist(),
        "split_seed": 42,
        "split_sha256": FROZEN_SPLIT_SHA256,
        "split_accepted_rows": role_counts,
        "input_contract_sha256": sha256_file(cache_dir / "input_contract.json"),
        "shards": shard_records,
    }
    _atomic_json(result, cache_dir / "acceptance.json")
    return result


def _load_screen_graphs(cache_dir: Path) -> tuple[dict, list]:
    acceptance_path = cache_dir / "acceptance.json"
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    if acceptance.get("accepted") is not True or acceptance.get("format") != FORMAT:
        raise RuntimeError("PubChemQC-100K screen cache is not accepted")
    records = acceptance.get("shards", [])
    paths = [cache_dir / "graph_shards" / str(record["path"]) for record in records]
    graphs = []
    for path, record in zip(paths, records):
        if sha256_file(path) != record.get("sha256"):
            raise RuntimeError(f"accepted screen shard changed: {path.name}")
        graphs.extend(torch.load(path, map_location="cpu", weights_only=False))
    if len(graphs) != int(acceptance["accepted_rows"]):
        raise RuntimeError("accepted screen graph count changed")
    return acceptance, graphs


def _model(candidate: str, device: torch.device) -> tuple[nn.Module, dict[str, object]]:
    if candidate in {"gps7", "gps9"}:
        config = {
            "in_channels": NODE_DIM,
            "edge_dim": EDGE_DIM,
            "hidden_channels": 192,
            "num_layers": 7 if candidate == "gps7" else 9,
            "num_heads": 4,
            "dropout": 0.05,
            "n_targets": 3,
            "pooling": "mean",
        }
        return GPSWrapper(**config).to(device), config
    if candidate == "gps11_160":
        config = {
            "in_channels": NODE_DIM,
            "edge_dim": EDGE_DIM,
            "hidden_channels": 160,
            "num_layers": 11,
            "num_heads": 4,
            "dropout": 0.05,
            "n_targets": 3,
            "pooling": "mean",
        }
        return GPSWrapper(**config).to(device), config
    if candidate == "pair_gps_2d":
        config = {
            "in_channels": NODE_DIM,
            "edge_dim": EDGE_DIM,
            "hidden_channels": 256,
            "pair_channels": 96,
            "num_layers": 10,
            "num_heads": 8,
            "dropout": 0.05,
            "n_targets": 3,
            "pooling": "mean",
            "path_steps": 5,
            "triplet_rank": 16,
        }
        return PairGPS2DWrapper(**config).to(device), config
    raise ValueError(f"unsupported fair-screen candidate: {candidate}")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    graphs: list,
    device: torch.device,
    batch_size: int,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
) -> dict[str, float]:
    model.eval()
    absolute_error = torch.zeros(3, dtype=torch.float64, device=device)
    count = 0
    batches = _size_bucket_batches(graphs, batch_size, seed=1_000_003)
    loader = DataLoader(
        graphs,
        batch_sampler=batches,
        num_workers=0,
        pin_memory=True,
    )
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        prediction = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        prediction = prediction.float() * target_std + target_mean
        target = batch.y.view(-1, 3).float()
        absolute_error += prediction.sub(target).abs().sum(dim=0).double()
        count += int(batch.num_graphs)
    values = (absolute_error / max(count, 1)).cpu().tolist()
    result = {name: float(values[index]) for index, name in enumerate(TARGET_NAMES)}
    result["average"] = float(sum(values) / len(values))
    return result


def train_screen_trial(
    *,
    split_csv: Path,
    cache_dir: Path,
    output_dir: Path,
    candidate: str,
    epochs: int = 40,
    patience: int = 10,
    batch_size: int = 64,
    learning_rate: float = 4e-4,
    weight_decay: float = 1e-5,
    seed: int = 42,
    resume: bool = True,
) -> dict:
    """Train one validation-only trial under the shared fair-screen contract."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the PubChemQC-100K screen")
    rows = _read_split(split_csv)
    acceptance, graphs = _load_screen_graphs(cache_dir)
    role_by_idx = {int(row["source_idx"]): str(row["role"]) for row in rows}
    split_graphs = {
        role: [
            graph
            for graph in graphs
            if role_by_idx[int(graph.source_idx.view(-1)[0])] == role
        ]
        for role in ROLE_INDEX
    }
    if {role: len(values) for role, values in split_graphs.items()} != FROZEN_SPLIT_ROWS:
        raise RuntimeError("fair-screen graph split counts changed")

    _set_seed(seed)
    device = torch.device("cuda")
    model, model_config = _model(candidate, device)
    target_mean = torch.tensor(acceptance["target_mean"], dtype=torch.float32, device=device)
    target_std = torch.tensor(acceptance["target_std"], dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1), eta_min=1e-6
    )
    criterion = nn.L1Loss()
    config = {
        "format": FORMAT,
        "candidate": candidate,
        "model_config": model_config,
        "cache_acceptance_sha256": sha256_file(cache_dir / "acceptance.json"),
        "split_sha256": FROZEN_SPLIT_SHA256,
        "epochs": epochs,
        "patience": patience,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "seed": seed,
        "precision": "fp32",
        "loss": "normalized_l1",
        "gradient_clip": 1.0,
        "test_role_read": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "checkpoint.pt"
    metrics_path = output_dir / "metrics.json"
    model_path = output_dir / "model.pt"

    if metrics_path.exists():
        existing = json.loads(metrics_path.read_text(encoding="utf-8"))
        if existing.get("status") == "complete":
            if existing.get("training_config") != config:
                raise RuntimeError("completed fair-screen trial contract changed")
            print(f"reuse completed trial {candidate} lr={learning_rate:g}", flush=True)
            return existing

    start_epoch = 0
    best_epoch = -1
    best_mae = float("inf")
    best_state = None
    wait = 0
    log = []
    if resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if checkpoint.get("config") != config:
            raise RuntimeError("fair-screen resume contract changed")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["next_epoch"])
        best_epoch = int(checkpoint["best_epoch"])
        best_mae = float(checkpoint["best_mae"])
        best_state = checkpoint["best_state"]
        wait = int(checkpoint["wait"])
        log = list(checkpoint["log"])
        print(f"resume {candidate} at epoch {start_epoch}", flush=True)

    for epoch in range(start_epoch, epochs):
        started = time.perf_counter()
        model.train()
        total_loss = 0.0
        train_count = 0
        loader = DataLoader(
            split_graphs["train"],
            batch_sampler=_size_bucket_batches(
                split_graphs["train"], batch_size, seed=seed + epoch * 100_003
            ),
            num_workers=0,
            pin_memory=True,
        )
        for batch in loader:
            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            target = (batch.y.view(-1, 3).float() - target_mean) / target_std
            prediction = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = criterion(prediction, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach()) * int(batch.num_graphs)
            train_count += int(batch.num_graphs)

        validation = _evaluate(
            model,
            split_graphs["validation"],
            device,
            batch_size,
            target_mean,
            target_std,
        )
        val_mae = float(validation["average"])
        improved = val_mae < best_mae
        if improved:
            best_mae = val_mae
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
        scheduler.step()
        row = {
            "epoch": epoch,
            "train_normalized_l1": total_loss / max(train_count, 1),
            "validation": validation,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "elapsed_s": time.perf_counter() - started,
            "selected": improved,
        }
        log.append(row)
        checkpoint = {
            "config": config,
            "next_epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_epoch": best_epoch,
            "best_mae": best_mae,
            "best_state": best_state,
            "wait": wait,
            "log": log,
        }
        _atomic_torch_save(checkpoint, checkpoint_path)
        _atomic_json(
            {
                "status": "running",
                "training_config": config,
                "next_epoch": epoch + 1,
                "best_epoch": best_epoch,
                "best_validation_average_mae_eV": best_mae,
                "log": log,
            },
            metrics_path,
        )
        print(
            f"{candidate} ep{epoch:02d} train={row['train_normalized_l1']:.6f} "
            f"val={val_mae:.6f}eV best={best_mae:.6f}@{best_epoch}"
            f"{' *' if improved else ''}",
            flush=True,
        )
        if wait >= patience:
            break

    if best_state is None:
        raise RuntimeError("fair-screen trial produced no best state")
    model.load_state_dict(best_state, strict=True)
    final_validation = _evaluate(
        model,
        split_graphs["validation"],
        device,
        batch_size,
        target_mean,
        target_std,
    )
    _atomic_torch_save(best_state, model_path)
    result = {
        "status": "complete",
        "experiment": "pubchemqc100k_pair_gps_2d_fair_screen",
        "candidate": candidate,
        "training_config": config,
        "n_params": sum(parameter.numel() for parameter in model.parameters()),
        "best_epoch": best_epoch,
        "best_validation_average_mae_eV": best_mae,
        "validation": final_validation,
        "test": None,
        "test_role_read": False,
        "log": log,
        "artifacts": {
            "checkpoint": str(checkpoint_path),
            "model": str(model_path),
            "cache_acceptance": str(cache_dir / "acceptance.json"),
        },
    }
    _atomic_json(result, metrics_path)
    return result


@torch.no_grad()
def evaluate_gps7_gps9_equal_validation(
    *,
    split_csv: Path,
    cache_dir: Path,
    gps7_dir: Path,
    gps9_dir: Path,
    output_dir: Path,
) -> dict:
    """Evaluate the fixed equal GPS7/GPS9 prediction on validation only."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the PubChemQC-100K screen")

    rows = _read_split(split_csv)
    acceptance, graphs = _load_screen_graphs(cache_dir)
    role_by_idx = {int(row["source_idx"]): str(row["role"]) for row in rows}
    validation_graphs = [
        graph
        for graph in graphs
        if role_by_idx[int(graph.source_idx.view(-1)[0])] == "validation"
    ]
    if len(validation_graphs) != FROZEN_SPLIT_ROWS["validation"]:
        raise RuntimeError("equal-screen validation graph count changed")

    component_metrics = {}
    shared_keys = (
        "cache_acceptance_sha256",
        "split_sha256",
        "epochs",
        "patience",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "seed",
        "precision",
        "loss",
        "gradient_clip",
        "test_role_read",
    )
    for candidate, directory in (("gps7", gps7_dir), ("gps9", gps9_dir)):
        metrics_path = directory / "metrics.json"
        model_path = directory / "model.pt"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if (
            metrics.get("status") != "complete"
            or metrics.get("candidate") != candidate
            or metrics.get("test_role_read") is not False
            or not model_path.is_file()
        ):
            raise RuntimeError(f"incomplete {candidate} validation component")
        component_metrics[candidate] = metrics

    gps7_config = component_metrics["gps7"]["training_config"]
    gps9_config = component_metrics["gps9"]["training_config"]
    if any(gps7_config.get(key) != gps9_config.get(key) for key in shared_keys):
        raise RuntimeError("GPS7/GPS9 equal component training contracts differ")

    contract = {
        "format": FORMAT,
        "candidate": "gps7_gps9_equal",
        "fusion": "fixed_prediction_average_0.5_0.5",
        "shared_training": {key: gps7_config.get(key) for key in shared_keys},
        "gps7_model_sha256": sha256_file(gps7_dir / "model.pt"),
        "gps9_model_sha256": sha256_file(gps9_dir / "model.pt"),
        "test_role_read": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    if metrics_path.exists():
        existing = json.loads(metrics_path.read_text(encoding="utf-8"))
        if existing.get("status") == "complete":
            if existing.get("evaluation_contract") != contract:
                raise RuntimeError("completed GPS7/GPS9 equal contract changed")
            return existing

    _set_seed(int(gps7_config["seed"]))
    device = torch.device("cuda")
    gps7, gps7_model_config = _model("gps7", device)
    gps9, gps9_model_config = _model("gps9", device)
    gps7.load_state_dict(
        torch.load(gps7_dir / "model.pt", map_location=device, weights_only=False),
        strict=True,
    )
    gps9.load_state_dict(
        torch.load(gps9_dir / "model.pt", map_location=device, weights_only=False),
        strict=True,
    )
    gps7.eval()
    gps9.eval()
    target_mean = torch.tensor(
        acceptance["target_mean"], dtype=torch.float32, device=device
    )
    target_std = torch.tensor(
        acceptance["target_std"], dtype=torch.float32, device=device
    )
    absolute_error = torch.zeros(3, dtype=torch.float64, device=device)
    count = 0
    batch_size = int(gps7_config["batch_size"])
    loader = DataLoader(
        validation_graphs,
        batch_sampler=_size_bucket_batches(
            validation_graphs, batch_size, seed=1_000_003
        ),
        num_workers=0,
        pin_memory=True,
    )
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        pred7 = gps7(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        pred9 = gps9(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        prediction = ((pred7.float() + pred9.float()) * 0.5) * target_std + target_mean
        target = batch.y.view(-1, 3).float()
        absolute_error += prediction.sub(target).abs().sum(dim=0).double()
        count += int(batch.num_graphs)

    values = (absolute_error / max(count, 1)).cpu().tolist()
    validation = {
        name: float(values[index]) for index, name in enumerate(TARGET_NAMES)
    }
    validation["average"] = float(sum(values) / len(values))
    result = {
        "status": "complete",
        "experiment": "pubchemqc100k_pair_gps_2d_fair_screen",
        "candidate": "gps7_gps9_equal",
        "evaluation_contract": contract,
        "model_configs": {
            "gps7": gps7_model_config,
            "gps9": gps9_model_config,
        },
        "validation": validation,
        "test": None,
        "test_role_read": False,
        "components": {
            "gps7": str(gps7_dir / "metrics.json"),
            "gps9": str(gps9_dir / "metrics.json"),
        },
    }
    _atomic_json(result, metrics_path)
    return result
