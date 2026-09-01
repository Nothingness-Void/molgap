"""Durable official-PCQM geometry cache and conservative warm-start training."""
from __future__ import annotations

import copy
import gc
import hashlib
import json
import math
import multiprocessing as mp
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader

from .pcqm_gap_architecture import make_pcqm_gap_encoder
from .pcqm_geometry import compute_etkdg_geometry, geometry_is_finite
from .pcqm_official_edge_state import (
    OfficialEdgeStateConfig,
    PackedGraphDataset,
    _make_model,
    atomic_json,
    atomic_torch,
    sha256_file,
)
from .pcqm_wedge import WedgeData, with_wedge_cache


CANDIDATE = "ogb_distance_angle_triangle_edge_state_gps9"
NEW_STATE_PREFIXES = (
    "wedge_initial.",
    "wedge_updates.",
    "wedge_to_edge.",
    "wedge_to_node.",
    "distance_basis.",
    "angle_basis.",
    "distance_initial.",
    "distance_updates.",
    "angle_initial.",
    "angle_updates.",
)


@dataclass(frozen=True)
class GeometryWarmstartConfig:
    seed: int = 42
    batch_size: int = 192
    eval_batch_size: int = 384
    shared_learning_rate: float = 2.0e-5
    new_learning_rate: float = 2.0e-4
    minimum_learning_rate: float = 1.0e-6
    weight_decay: float = 1.0e-5
    max_epochs: int = 12
    patience: int = 4
    gradient_clip: float = 1.0
    max_projected_training_s: float = 12.0 * 3600.0
    hard_job_budget_s: float = 13.5 * 3600.0
    minimum_memory_headroom_fraction: float = 0.15


def _atomic_packed(path: Path, graphs: list[WedgeData]) -> None:
    data, slices = InMemoryDataset.collate(graphs)
    atomic_torch(path, (data, slices))


def _geometry_job(payload):
    return compute_etkdg_geometry(*payload)


def _role_base_path(base_graph_dir: Path, role: str, shard_index: int) -> Path:
    return Path(base_graph_dir) / role / f"{role}_shard_{shard_index:04d}.pt"


def _load_smiles(rows_dir: Path, shard_index: int) -> tuple[dict[int, str], dict]:
    manifest_path = Path(rows_dir) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise RuntimeError("Official row manifest is incomplete")
    if not 0 <= shard_index < len(manifest["shards"]):
        raise IndexError(f"shard_index {shard_index} is outside prepared rows")
    record = manifest["shards"][shard_index]
    source = Path(rows_dir) / record["path"]
    if sha256_file(source) != record["sha256"]:
        raise RuntimeError(f"Official row shard changed: {source}")
    frame = pd.read_csv(source, compression="gzip")
    if not {"idx", "smiles", "split_code"}.issubset(frame.columns):
        raise ValueError(f"Unexpected official row columns: {list(frame.columns)}")
    mapping = dict(zip(frame["idx"].astype(int), frame["smiles"].astype(str)))
    if len(mapping) != len(frame):
        raise RuntimeError("Official row shard contains duplicate indices")
    return mapping, record


def _attach_geometry(graph, result) -> WedgeData:
    converted = WedgeData(**graph.to_dict())
    converted.edge_distance = torch.from_numpy(result.edge_distance)
    converted.wedge_angle_cos = torch.from_numpy(result.wedge_angle_cos)
    converted.geometry_valid = torch.tensor([result.geometry_valid], dtype=torch.bool)
    converted.mmff_converged = torch.tensor([result.mmff_converged], dtype=torch.bool)
    return converted


def build_geometry_shard(
    rows_dir: Path,
    base_graph_dir: Path,
    output_dir: Path,
    *,
    shard_index: int,
    workers: int = 8,
) -> dict:
    """Attach wedges and deterministic ETKDGv3/MMFF geometry to one shard."""
    started = time.monotonic()
    rows_dir, base_graph_dir, output_dir = map(Path, (rows_dir, base_graph_dir, output_dir))
    report_path = output_dir / "reports" / f"shard_{shard_index:04d}.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        for record in report.get("outputs", []):
            path = output_dir / record["path"]
            if not path.is_file() or sha256_file(path) != record["sha256"]:
                raise RuntimeError(f"Published geometry shard changed: {path}")
        if report.get("status") == "complete":
            return report

    smiles, source_record = _load_smiles(rows_dir, shard_index)
    outputs, failures = [], []
    counts = {"train": 0, "valid": 0}
    valid_count = converged_count = 0
    for role in ("train", "valid"):
        base_path = _role_base_path(base_graph_dir, role, shard_index)
        if not base_path.is_file():
            continue
        base_sha = sha256_file(base_path)
        dataset = PackedGraphDataset(base_path)
        graphs = [with_wedge_cache(dataset[index]) for index in range(len(dataset))]
        payloads = []
        for graph in graphs:
            source_idx = int(graph.source_idx.view(-1)[0])
            if source_idx not in smiles:
                raise RuntimeError(f"Graph source_idx {source_idx} is absent from row shard")
            payloads.append((
                smiles[source_idx],
                source_idx,
                int(graph.num_nodes),
                graph.edge_index.numpy(),
                graph.wedge_edge_ids.numpy(),
            ))
        if int(workers) == 1:
            results = list(map(_geometry_job, payloads))
        else:
            context = mp.get_context("spawn")
            with context.Pool(processes=int(workers)) as pool:
                results = pool.map(_geometry_job, payloads, chunksize=16)
        converted = []
        for graph, result in zip(graphs, results):
            if not geometry_is_finite(result):
                raise RuntimeError("Geometry worker returned non-finite fallback tensors")
            converted.append(_attach_geometry(graph, result))
            valid_count += int(result.geometry_valid)
            converged_count += int(result.mmff_converged)
            if not result.geometry_valid:
                failures.append({
                    "source_idx": int(graph.source_idx.view(-1)[0]),
                    "role": role,
                    "attempt": result.embed_attempt,
                    "type": result.failure_type,
                    "message": result.failure_message,
                })
        destination = output_dir / role / f"{role}_shard_{shard_index:04d}.pt"
        _atomic_packed(destination, converted)
        record = {
            "role": role,
            "path": str(destination.relative_to(output_dir)).replace("\\", "/"),
            "rows": len(converted),
            "source_idx_min": min(int(g.source_idx.view(-1)[0]) for g in converted),
            "source_idx_max": max(int(g.source_idx.view(-1)[0]) for g in converted),
            "base_path": str(base_path),
            "base_sha256": base_sha,
            "sha256": sha256_file(destination),
            "bytes": destination.stat().st_size,
        }
        outputs.append(record)
        counts[role] = len(converted)
        del dataset, graphs, converted, results
        gc.collect()

    expected = source_record["counts"]
    if counts != expected:
        raise RuntimeError(f"Geometry shard role counts changed: {counts} != {expected}")
    report = {
        "format": "molgap-pcqm4mv2-geometry-shard-v1",
        "status": "complete",
        "shard_index": int(shard_index),
        "source_record": source_record,
        "counts": counts,
        "valid_geometry": valid_count,
        "mmff_converged": converged_count,
        "invalid_geometry": sum(counts.values()) - valid_count,
        "failures": failures,
        "outputs": outputs,
        "geometry": {
            "conformer": "ETKDGv3",
            "seed": "42-plus-source_idx",
            "optimization": "MMFF94s",
            "max_iterations": 200,
        },
        "elapsed_s": time.monotonic() - started,
        "official_test_used": False,
        "external_data_used": False,
    }
    atomic_json(report_path, report)
    return report


def accept_geometry_cache(
    rows_dir: Path,
    base_acceptance_path: Path,
    graph_dir: Path,
    output_path: Path,
) -> dict:
    """Exhaustively validate all published geometry graph shards."""
    rows_dir, graph_dir = Path(rows_dir), Path(graph_dir)
    row_manifest_path = rows_dir / "manifest.json"
    row_manifest = json.loads(row_manifest_path.read_text(encoding="utf-8"))
    base_acceptance_path = Path(base_acceptance_path)
    base_acceptance = json.loads(base_acceptance_path.read_text(encoding="utf-8"))
    if row_manifest.get("status") != "complete" or base_acceptance.get("status") != "accepted":
        raise RuntimeError("An accepted official input is missing")
    if base_acceptance.get("feature_schema") != "ogb":
        raise RuntimeError("Warm-start geometry cache requires OGB categorical graphs")

    all_indices, output_records = [], []
    counts = {"train": 0, "valid": 0}
    valid_count = converged_count = 0
    for shard_index, source_record in enumerate(row_manifest["shards"]):
        report_path = graph_dir / "reports" / f"shard_{shard_index:04d}.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("status") != "complete" or report.get("source_record") != source_record:
            raise RuntimeError(f"Geometry report identity changed: {report_path}")
        for record in report["outputs"]:
            path = graph_dir / record["path"]
            if sha256_file(path) != record["sha256"]:
                raise RuntimeError(f"Geometry graph hash changed: {path}")
            dataset = PackedGraphDataset(path)
            if len(dataset) != record["rows"]:
                raise RuntimeError(f"Geometry graph count changed: {path}")
            for graph in dataset:
                required = (
                    "source_idx", "x", "edge_index", "edge_attr", "random_walk_pe",
                    "wedge_edge_ids", "edge_distance", "wedge_angle_cos",
                    "geometry_valid", "mmff_converged", "y",
                )
                missing = [name for name in required if not hasattr(graph, name)]
                if missing:
                    raise RuntimeError(f"Geometry graph is missing {missing}: {path}")
                if graph.wedge_edge_ids.ndim != 2 or graph.wedge_edge_ids.shape[1] != 2:
                    raise RuntimeError("Wedge cache shape changed")
                if tuple(graph.edge_distance.shape) != (graph.edge_index.shape[1], 1):
                    raise RuntimeError("Distance cache is not aligned to directed edges")
                if tuple(graph.wedge_angle_cos.shape) != (graph.wedge_edge_ids.shape[0], 1):
                    raise RuntimeError("Angle cache is not aligned to wedges")
                if not torch.isfinite(graph.edge_distance).all() or not torch.isfinite(graph.wedge_angle_cos).all():
                    raise RuntimeError("Geometry cache contains non-finite tensors")
                all_indices.append(int(graph.source_idx.view(-1)[0]))
                valid_count += int(graph.geometry_valid.view(-1)[0])
                converged_count += int(graph.mmff_converged.view(-1)[0])
            counts[record["role"]] += len(dataset)
            output_records.append(record)
            del dataset
            gc.collect()

    expected_counts = {key: int(value) for key, value in row_manifest["counts"].items()}
    if counts != expected_counts:
        raise RuntimeError(f"Accepted geometry counts changed: {counts} != {expected_counts}")
    indices = np.asarray(all_indices, dtype=np.int64)
    if len(np.unique(indices)) != len(indices):
        raise RuntimeError("Geometry cache contains duplicate source_idx values")
    aggregate = hashlib.sha256()
    for record in sorted(output_records, key=lambda item: item["path"]):
        aggregate.update(f"{record['path']}\t{record['sha256']}\n".encode("ascii"))
    acceptance = {
        "format": "molgap-pcqm4mv2-geometry-cache-acceptance-v1",
        "status": "accepted",
        "counts": counts,
        "node_feature_dim": int(base_acceptance["node_feature_dim"]),
        "edge_feature_dim": int(base_acceptance["edge_feature_dim"]),
        "rwse_dim": int(base_acceptance["rwse_dim"]),
        "feature_schema": "ogb",
        "target_mean_gap": float(base_acceptance["target_mean_gap"]),
        "target_std_gap": float(base_acceptance["target_std_gap"]),
        "source_idx_count": len(indices),
        "source_idx_min": int(indices.min()),
        "source_idx_max": int(indices.max()),
        "source_idx_sha256": hashlib.sha256(np.sort(indices).tobytes()).hexdigest(),
        "valid_geometry": valid_count,
        "invalid_geometry": len(indices) - valid_count,
        "valid_geometry_fraction": valid_count / len(indices),
        "mmff_converged": converged_count,
        "shards": output_records,
        "aggregate_sha256": aggregate.hexdigest(),
        "row_manifest_sha256": sha256_file(row_manifest_path),
        "base_acceptance_sha256": sha256_file(base_acceptance_path),
        "official_test_used": False,
        "external_data_used": False,
    }
    atomic_json(output_path, acceptance)
    return acceptance


def _embedding_target_key(source_key: str) -> str:
    if source_key.startswith("node_emb.embeddings."):
        return source_key.replace("node_emb.embeddings.", "node_emb.atom_embedding_list.", 1)
    if source_key.startswith("edge_emb.embeddings."):
        return source_key.replace("edge_emb.embeddings.", "edge_emb.bond_embedding_list.", 1)
    return source_key


def load_pretrained_backbone(target: nn.Module, source_state: dict[str, torch.Tensor]) -> dict:
    """Map every source tensor exactly once and leave only declared new tensors."""
    target_state = target.state_dict()
    mapped, source_to_target = {}, {}
    for source_key, value in source_state.items():
        target_key = _embedding_target_key(source_key)
        if target_key not in target_state:
            raise RuntimeError(f"Pretrained source tensor has no target: {source_key}")
        if target_state[target_key].shape != value.shape:
            raise RuntimeError(
                f"Pretrained tensor shape changed for {source_key}: "
                f"{tuple(value.shape)} != {tuple(target_state[target_key].shape)}"
            )
        if target_key in mapped:
            raise RuntimeError(f"Two source tensors map to {target_key}")
        mapped[target_key] = value
        source_to_target[source_key] = target_key
    new_keys = sorted(set(target_state) - set(mapped))
    invalid_new = [key for key in new_keys if not key.startswith(NEW_STATE_PREFIXES)]
    if invalid_new:
        raise RuntimeError(f"Undeclared warm-start tensors: {invalid_new}")
    merged = dict(target_state)
    merged.update(mapped)
    target.load_state_dict(merged, strict=True)
    parameter_keys = dict(target.named_parameters())
    mapped_parameters = sum(parameter_keys[key].numel() for key in mapped if key in parameter_keys)
    source_parameters = sum(value.numel() for key, value in source_state.items() if key in dict(target.named_parameters()) or _embedding_target_key(key) in parameter_keys)
    if mapped_parameters != source_parameters:
        raise RuntimeError("Warm-start parameter coverage is incomplete")
    return {
        "source_tensor_count": len(source_state),
        "mapped_tensor_count": len(mapped),
        "mapped_parameter_count": mapped_parameters,
        "new_tensor_count": len(new_keys),
        "new_keys": new_keys,
        "source_to_target": source_to_target,
    }


def _forward_geometry(model: nn.Module, batch) -> torch.Tensor:
    return model(
        batch.x, batch.edge_index, batch.edge_attr, batch.batch,
        batch.random_walk_pe, batch.wedge_edge_ids, batch.edge_distance,
        batch.wedge_angle_cos, batch.geometry_valid,
    ).view(-1)


@torch.no_grad()
def _evaluate(model, graph_dir: Path, device, batch_size: int, mean: float, std: float, *, predictions=False):
    model.eval()
    absolute = count = 0
    indices, values, targets = [], [], []
    for path in sorted((Path(graph_dir) / "valid").glob("valid_shard_*.pt")):
        dataset = PackedGraphDataset(path)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        for batch in loader:
            batch = batch.to(device)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                prediction = _forward_geometry(model, batch)
            prediction = prediction.float() * std + mean
            target = batch.y.view(-1)
            absolute += float((prediction - target).abs().sum())
            count += target.numel()
            if predictions:
                indices.append(batch.source_idx.view(-1).cpu())
                values.append(prediction.cpu())
                targets.append(target.cpu())
        del loader, dataset
    if not predictions:
        return absolute / max(count, 1)
    order = torch.argsort(torch.cat(indices))
    return absolute / max(count, 1), {
        "source_idx": torch.cat(indices)[order],
        "prediction_eV": torch.cat(values)[order],
        "target_eV": torch.cat(targets)[order],
    }


def _load_source_checkpoint(path: Path, acceptance: dict) -> tuple[dict, nn.Module]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    source_config = OfficialEdgeStateConfig(**checkpoint["config"])
    if source_config.feature_schema != "ogb":
        raise RuntimeError("Warm-start source checkpoint is not OGB-rich")
    if checkpoint.get("best_epoch") != 30:
        raise RuntimeError("Warm-start source is not the accepted epoch-30 checkpoint")
    if not math.isclose(float(checkpoint["target_mean_gap"]), float(acceptance["target_mean_gap"]), abs_tol=1e-12):
        raise RuntimeError("Warm-start source target mean changed")
    if not math.isclose(float(checkpoint["target_std_gap"]), float(acceptance["target_std_gap"]), abs_tol=1e-12):
        raise RuntimeError("Warm-start source target std changed")
    source_model = _make_model(source_config, int(acceptance["node_feature_dim"]))
    source_model.load_state_dict(checkpoint["model"], strict=True)
    return checkpoint, source_model


def cpu_smoke(
    rows_dir: Path,
    base_graph_dir: Path,
    base_acceptance_path: Path,
    source_checkpoint_path: Path,
    output_path: Path,
) -> dict:
    """Validate immutable inputs, one ETKDG graph, and the weight map on CPU."""
    started = time.monotonic()
    base_acceptance_path, source_checkpoint_path = map(
        Path, (base_acceptance_path, source_checkpoint_path)
    )
    acceptance = json.loads(base_acceptance_path.read_text(encoding="utf-8"))
    if acceptance.get("status") != "accepted" or acceptance.get("feature_schema") != "ogb":
        raise RuntimeError("CPU smoke requires accepted OGB-rich base graphs")
    checkpoint, source_model = _load_source_checkpoint(source_checkpoint_path, acceptance)
    target_model = make_pcqm_gap_encoder(CANDIDATE)
    mapping = load_pretrained_backbone(target_model, checkpoint["model"])
    smiles, _ = _load_smiles(rows_dir, 0)
    base_path = _role_base_path(base_graph_dir, "train", 0)
    dataset = PackedGraphDataset(base_path)
    graph = with_wedge_cache(dataset[0])
    source_idx = int(graph.source_idx.view(-1)[0])
    result = compute_etkdg_geometry(
        smiles[source_idx], source_idx, int(graph.num_nodes),
        graph.edge_index.numpy(), graph.wedge_edge_ids.numpy(),
    )
    if not geometry_is_finite(result):
        raise RuntimeError("CPU smoke geometry is non-finite")
    graph = _attach_geometry(graph, result)
    batch = next(iter(DataLoader([graph], batch_size=1)))
    source_model.eval()
    target_model.eval()
    with torch.no_grad():
        source_value = source_model(
            batch.x, batch.edge_index, batch.edge_attr, batch.batch,
            batch.random_walk_pe,
        ).view(-1)
        target_value = _forward_geometry(target_model, batch)
    difference = float((source_value - target_value).abs().max())
    if difference > 2.0e-5:
        raise RuntimeError(f"CPU warm-start identity changed: {difference}")
    payload = {
        "format": "molgap-pcqm4mv2-geometry-warmstart-cpu-smoke-v1",
        "status": "accepted",
        "candidate": CANDIDATE,
        "source_checkpoint_sha256": sha256_file(source_checkpoint_path),
        "base_acceptance_sha256": sha256_file(base_acceptance_path),
        "base_graph_sha256": sha256_file(base_path),
        "source_idx": source_idx,
        "geometry_valid": bool(result.geometry_valid),
        "initial_function_max_abs_difference": difference,
        "mapping": mapping,
        "elapsed_s": time.monotonic() - started,
        "gpu_used": False,
        "official_test_used": False,
    }
    atomic_json(output_path, payload)
    return payload


def gpu_preflight(
    graph_dir: Path,
    acceptance_path: Path,
    source_checkpoint_path: Path,
    output_path: Path,
    *,
    config: GeometryWarmstartConfig = GeometryWarmstartConfig(),
    batches: int = 64,
) -> dict:
    """Check exact warm-start behavior, finite gradients, memory, and throughput."""
    if not torch.cuda.is_available():
        raise RuntimeError("Geometry warm-start preflight requires CUDA")
    acceptance_path, source_checkpoint_path = Path(acceptance_path), Path(source_checkpoint_path)
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    if acceptance.get("status") != "accepted":
        raise RuntimeError("Geometry cache was not accepted")
    checkpoint, source_model = _load_source_checkpoint(source_checkpoint_path, acceptance)
    target_model = make_pcqm_gap_encoder(CANDIDATE)
    mapping = load_pretrained_backbone(target_model, checkpoint["model"])
    device = torch.device("cuda")
    source_model, target_model = source_model.to(device).eval(), target_model.to(device).eval()
    first_path = sorted((Path(graph_dir) / "train").glob("train_shard_*.pt"))[0]
    dataset = PackedGraphDataset(first_path)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)
    iterator = iter(loader)
    batch = next(iterator).to(device)
    with torch.no_grad():
        source_prediction = source_model(
            batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.random_walk_pe
        ).view(-1)
        target_prediction = _forward_geometry(target_model, batch)
    max_difference = float((source_prediction - target_prediction).abs().max())
    if max_difference > 2.0e-5:
        raise RuntimeError(f"Warm-start changed the initial function: {max_difference}")

    target_model.train()
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    rows = 0
    optimizer = torch.optim.AdamW(target_model.parameters(), lr=config.shared_learning_rate)
    mean = torch.tensor(float(acceptance["target_mean_gap"]), device=device)
    std = torch.tensor(float(acceptance["target_std_gap"]), device=device)
    for step, batch in enumerate(loader):
        if step >= batches:
            break
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=True):
            prediction = _forward_geometry(target_model, batch)
            target = (batch.y.view(-1) - mean) / std
            loss = torch.nn.functional.l1_loss(prediction, target)
        loss.backward()
        nn.utils.clip_grad_norm_(target_model.parameters(), config.gradient_clip)
        optimizer.step()
        if not torch.isfinite(loss):
            raise RuntimeError("Warm-start preflight loss is non-finite")
        rows += batch.y.numel()
    torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    peak = int(torch.cuda.max_memory_reserved())
    total = int(torch.cuda.get_device_properties(0).total_memory)
    headroom = 1.0 - peak / total
    projected_epoch_s = elapsed / max(rows, 1) * int(acceptance["counts"]["train"]) * 1.20
    projected_training_s = projected_epoch_s * config.max_epochs
    if headroom < config.minimum_memory_headroom_fraction:
        raise RuntimeError(f"A100 memory headroom is too small: {headroom:.3f}")
    if projected_training_s > config.max_projected_training_s:
        raise RuntimeError(
            f"Projected training exceeds gate: {projected_training_s:.0f}s"
        )
    result = {
        "format": "molgap-pcqm4mv2-geometry-warmstart-preflight-v1",
        "status": "accepted",
        "candidate": CANDIDATE,
        "config": asdict(config),
        "gpu": torch.cuda.get_device_name(0),
        "source_checkpoint_sha256": sha256_file(source_checkpoint_path),
        "acceptance_sha256": sha256_file(acceptance_path),
        "initial_function_max_abs_difference": max_difference,
        "mapping": mapping,
        "sample_rows": rows,
        "sample_elapsed_s": elapsed,
        "peak_memory_bytes": peak,
        "total_memory_bytes": total,
        "memory_headroom_fraction": headroom,
        "projected_epoch_s": projected_epoch_s,
        "projected_training_s": projected_training_s,
        "official_test_used": False,
    }
    atomic_json(output_path, result)
    return result


def train_geometry_warmstart(
    graph_dir: Path,
    acceptance_path: Path,
    source_checkpoint_path: Path,
    preflight_path: Path,
    output_dir: Path,
    *,
    config: GeometryWarmstartConfig = GeometryWarmstartConfig(),
) -> dict:
    """Train the accepted geometry architecture from an audited OGB-rich GPS9."""
    started = time.monotonic()
    if not torch.cuda.is_available():
        raise RuntimeError("Geometry warm-start training requires CUDA")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    acceptance_path, source_checkpoint_path, preflight_path = map(
        Path, (acceptance_path, source_checkpoint_path, preflight_path)
    )
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight.get("status") != "accepted":
        raise RuntimeError("GPU preflight did not pass")
    if preflight.get("acceptance_sha256") != sha256_file(acceptance_path):
        raise RuntimeError("GPU preflight refers to another geometry cache")
    if preflight.get("source_checkpoint_sha256") != sha256_file(source_checkpoint_path):
        raise RuntimeError("GPU preflight refers to another source checkpoint")
    checkpoint, _ = _load_source_checkpoint(source_checkpoint_path, acceptance)
    model = make_pcqm_gap_encoder(CANDIDATE)
    mapping = load_pretrained_backbone(model, checkpoint["model"])
    mapped_keys = set(mapping["source_to_target"].values())
    shared, new = [], []
    for name, parameter in model.named_parameters():
        (shared if name in mapped_keys else new).append(parameter)
    if not shared or not new:
        raise RuntimeError("Warm-start optimizer groups are incomplete")
    device = torch.device("cuda")
    model = model.to(device)
    optimizer = torch.optim.AdamW([
        {"params": shared, "lr": config.shared_learning_rate},
        {"params": new, "lr": config.new_learning_rate},
    ], weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.max_epochs, eta_min=config.minimum_learning_rate
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    last_path, best_path = output_dir / "last.pt", output_dir / "best.pt"
    contract = asdict(config)
    identities = {
        "acceptance_sha256": sha256_file(acceptance_path),
        "source_checkpoint_sha256": sha256_file(source_checkpoint_path),
        "preflight_sha256": sha256_file(preflight_path),
    }
    start_epoch, best_epoch, best_mae, wait, log = 0, -1, float("inf"), 0, []
    if last_path.is_file():
        state = torch.load(last_path, map_location=device, weights_only=False)
        if state.get("config") != contract or state.get("identities") != identities:
            raise RuntimeError("Warm-start resume contract changed")
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_epoch = int(state["epoch"]) + 1
        best_epoch, best_mae = int(state["best_epoch"]), float(state["best_valid_gap_mae_eV"])
        wait, log = int(state["wait"]), list(state["log"])

    mean = torch.tensor(float(acceptance["target_mean_gap"]), device=device)
    std = torch.tensor(float(acceptance["target_std_gap"]), device=device)
    train_files = sorted((Path(graph_dir) / "train").glob("train_shard_*.pt"))
    if not train_files:
        raise FileNotFoundError("No accepted geometry training shards")
    for epoch in range(start_epoch, config.max_epochs):
        epoch_started = time.monotonic()
        model.train()
        generator = random.Random(config.seed + epoch)
        files = train_files.copy()
        generator.shuffle(files)
        loss_sum = row_count = 0
        for path in files:
            dataset = PackedGraphDataset(path)
            loader = DataLoader(
                dataset, batch_size=config.batch_size, shuffle=True,
                num_workers=2, pin_memory=True, persistent_workers=True,
            )
            for batch in loader:
                batch = batch.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=True):
                    prediction = _forward_geometry(model, batch)
                    target = (batch.y.view(-1) - mean) / std
                    loss = torch.nn.functional.l1_loss(prediction, target)
                if not torch.isfinite(loss):
                    raise RuntimeError("Warm-start training loss is non-finite")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
                scaler.step(optimizer)
                scaler.update()
                loss_sum += float(loss.detach()) * batch.y.numel()
                row_count += batch.y.numel()
            del loader, dataset
            gc.collect()
        scheduler.step()
        valid_mae = _evaluate(
            model, graph_dir, device, config.eval_batch_size,
            float(mean), float(std),
        )
        elapsed = time.monotonic() - epoch_started
        projected = elapsed * config.max_epochs
        if epoch == 0 and projected > config.max_projected_training_s:
            raise RuntimeError(f"Measured full training exceeds timing gate: {projected:.0f}s")
        improved = math.isfinite(valid_mae) and valid_mae < best_mae
        if improved:
            best_mae, best_epoch, wait = valid_mae, epoch, 0
            atomic_torch(best_path, {
                "format": "molgap-pcqm4mv2-geometry-warmstart-best-v1",
                "candidate": CANDIDATE, "config": contract,
                "model": copy.deepcopy(model.state_dict()),
                "best_epoch": best_epoch, "best_valid_gap_mae_eV": best_mae,
                "target_mean_gap": float(mean), "target_std_gap": float(std),
                "identities": identities, "mapping": mapping,
                "pretrained_weights_used": True,
            })
        else:
            wait += 1
        row = {
            "epoch": epoch,
            "train_gap_l1_normalized": loss_sum / max(row_count, 1),
            "train_rows": row_count,
            "valid_gap_mae_eV": valid_mae,
            "elapsed_s": elapsed,
            "selected": improved,
            "learning_rates": [group["lr"] for group in optimizer.param_groups],
        }
        log.append(row)
        atomic_torch(last_path, {
            "format": "molgap-pcqm4mv2-geometry-warmstart-checkpoint-v1",
            "candidate": CANDIDATE, "config": contract, "identities": identities,
            "epoch": epoch, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(), "best_epoch": best_epoch,
            "best_valid_gap_mae_eV": best_mae, "wait": wait, "log": log,
        })
        atomic_json(output_dir / "progress.json", {
            "status": "training", "epoch": epoch, "best_epoch": best_epoch,
            "best_valid_gap_mae_eV": best_mae,
            "elapsed_s": time.monotonic() - started,
        })
        print(
            f"geometry-warmstart ep{epoch:02d} train={row['train_gap_l1_normalized']:.6f} "
            f"valid={valid_mae:.6f}eV {elapsed:.1f}s{' *' if improved else ''}",
            flush=True,
        )
        if time.monotonic() - started > config.hard_job_budget_s:
            raise TimeoutError("Warm-start hard job budget reached after durable checkpoint")
        if wait >= config.patience:
            break

    if not best_path.is_file():
        raise RuntimeError("Warm-start training produced no finite best checkpoint")
    best = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best["model"], strict=True)
    valid_mae, predictions = _evaluate(
        model, graph_dir, device, config.eval_batch_size,
        float(mean), float(std), predictions=True,
    )
    atomic_torch(output_dir / "valid_predictions.pt", predictions)
    metrics = {
        "format": "molgap-pcqm4mv2-geometry-warmstart-training-v1",
        "status": "complete", "candidate": CANDIDATE, "config": contract,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "best_epoch": int(best["best_epoch"]), "valid_gap_mae_eV": float(valid_mae),
        "train_log": log, "official_train_rows": acceptance["counts"]["train"],
        "official_valid_rows": acceptance["counts"]["valid"],
        "official_valid_used": True, "official_test_used": False,
        "external_data_used": False, "pretrained_weights_used": True,
        "source_checkpoint_sha256": identities["source_checkpoint_sha256"],
        "production_registry_changed": False,
        "best_sha256": sha256_file(best_path),
        "valid_predictions_sha256": sha256_file(output_dir / "valid_predictions.pt"),
        "runtime_s": time.monotonic() - started,
    }
    atomic_json(output_dir / "metrics.json", metrics)
    atomic_json(output_dir / "completion_manifest.json", {
        "status": "complete",
        "best": {"path": "best.pt", "sha256": metrics["best_sha256"]},
        "metrics": {"path": "metrics.json", "sha256": sha256_file(output_dir / "metrics.json")},
        "valid_predictions": {"path": "valid_predictions.pt", "sha256": metrics["valid_predictions_sha256"]},
    })
    return metrics
