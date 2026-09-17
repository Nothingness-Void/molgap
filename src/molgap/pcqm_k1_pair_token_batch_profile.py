"""Non-scientific Kunshan batch profiling for the frozen PairToken model.

This module deliberately permits physical batches other than 128.  It produces
runtime evidence only: no validation score, checkpoint, model bundle, or
scientific decision is emitted.
"""
from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import time
from pathlib import Path

from .pcqm_gptrans_v4 import _forward
from .pcqm_k1_pair_token_500k import PARAMETERS, SEED, make_model, optimizer_for
from .pcqm_k1_scale import FIXED_500K_MANIFEST_SHA256, SCALE_TRAIN_ROWS
from .pcqm_k1_scale_runner import find_cache
from .runtime_profiling import RuntimeProfile
from .training_reproducibility import (
    atomic_json,
    build_runtime_manifest,
    configure_fp32_determinism,
    sha256_file,
)


FORMAT = "molgap-pairtoken-batch-profile-v1"
BATCH_SIZES = (64, 128, 256, 512, 1024, 2048)
REPRESENTATIVE_ROWS = 16_384
TAIL_ROWS = 4_096
WARMUP_ROWS = 4_096
MIN_MEMORY_RESERVE_FRACTION = 0.15


def _hash_indices(indices) -> str:
    payload = ",".join(str(int(value)) for value in indices).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _load_train_only(root: Path, manifest: dict):
    """Load and verify train shards without deserializing development payloads."""
    import torch
    from torch.utils.data import ConcatDataset
    from torch_geometric.data import InMemoryDataset

    class PackedGraphDataset(InMemoryDataset):
        def __init__(self, path: Path):
            super().__init__(root=None)
            self.data, self.slices = torch.load(
                path, map_location="cpu", weights_only=False
            )

    shards = []
    records = []
    for record in manifest["geometry_shards"]:
        if record["role"] != "train":
            continue
        path = root / record["file"]
        if sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"Train shard changed: {path.name}")
        dataset = PackedGraphDataset(path)
        if len(dataset) != record["rows"]:
            raise RuntimeError(f"Train shard row count changed: {path.name}")
        shards.append(dataset)
        records.append(
            {
                "file": record["file"],
                "rows": int(record["rows"]),
                "sha256": record["sha256"],
            }
        )
    graphs = ConcatDataset(shards)
    if len(graphs) != SCALE_TRAIN_ROWS:
        raise RuntimeError(f"Expected {SCALE_TRAIN_ROWS} train graphs, got {len(graphs)}")
    return graphs, shards, records


def _graph_sizes(shards):
    """Read node/edge counts from packed slice tables without materializing graphs."""
    import torch

    nodes = []
    edges = []
    for shard in shards:
        nodes.append((shard.slices["x"][1:] - shard.slices["x"][:-1]).long())
        edges.append(
            (
                shard.slices["edge_index"][1:]
                - shard.slices["edge_index"][:-1]
            ).long()
        )
    return torch.cat(nodes), torch.cat(edges)


def _cohorts(node_counts, edge_counts) -> dict[str, list[int]]:
    import torch

    generator = torch.Generator().manual_seed(SEED + 50_000)
    representative = torch.randperm(len(node_counts), generator=generator)[
        : REPRESENTATIVE_ROWS + WARMUP_ROWS
    ]
    # Lexicographic-like integer score: nodes dominate, edges break ties.
    score = node_counts * (int(edge_counts.max()) + 1) + edge_counts
    tail = torch.argsort(score, descending=True)[: TAIL_ROWS + WARMUP_ROWS]
    return {
        "representative": representative.tolist(),
        "graph_size_tail": tail.tolist(),
    }


def _normalizer(shards) -> tuple[float, float]:
    import torch

    values = torch.cat([shard._data.y.view(-1) for shard in shards]).float()
    mean = float(values.mean())
    std = float(values.std(unbiased=True).clamp_min(1e-6))
    if not math.isfinite(mean) or not math.isfinite(std):
        raise RuntimeError("Non-finite train-only target normalizer")
    return mean, std


def _loader(graphs, indices, batch_size: int):
    import torch
    from torch.utils.data import Subset
    from torch_geometric.loader import DataLoader

    workers = int(os.environ.get("MOLGAP_PROFILE_LOADER_WORKERS", "2"))
    return DataLoader(
        Subset(graphs, indices),
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        prefetch_factor=2 if workers > 0 else None,
        generator=torch.Generator().manual_seed(SEED + batch_size),
    )


def _one_step(model, optimizer, batch, mean, std, profile=None) -> None:
    import torch

    optimizer.zero_grad(set_to_none=True)
    if profile is None:
        batch = batch.to("cuda", non_blocking=True)
        prediction = _forward(model, batch)
        target = (batch.y.view(-1) - mean) / std
        loss = torch.nn.functional.l1_loss(prediction, target)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("Non-finite profiling loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        torch.cuda.synchronize()
        return

    started = time.perf_counter()
    batch = batch.to("cuda", non_blocking=True)
    torch.cuda.synchronize()
    profile.add("h2d", time.perf_counter() - started)

    started = time.perf_counter()
    prediction = _forward(model, batch)
    target = (batch.y.view(-1) - mean) / std
    loss = torch.nn.functional.l1_loss(prediction, target)
    torch.cuda.synchronize()
    profile.add("forward_loss", time.perf_counter() - started)
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("Non-finite profiling loss")

    started = time.perf_counter()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    torch.cuda.synchronize()
    profile.add("backward_optimizer", time.perf_counter() - started)


def _profile_one(graphs, indices, batch_size: int, mean, std) -> dict:
    import torch

    configure_fp32_determinism(SEED)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = make_model().to("cuda")
    optimizer = optimizer_for(model)
    model.train()
    total_memory = int(torch.cuda.get_device_properties(0).total_memory)
    baseline_allocated = int(torch.cuda.memory_allocated())
    warmup_batches = max(1, math.ceil(WARMUP_ROWS / batch_size))
    loader = _loader(graphs, indices, batch_size)
    iterator = iter(loader)
    status = "complete"
    error = None
    profile = RuntimeProfile()
    measured_graphs = 0
    measured_steps = 0

    try:
        for _ in range(warmup_batches):
            _one_step(model, optimizer, next(iterator), mean, std)
        while True:
            started = time.perf_counter()
            try:
                batch = next(iterator)
            except StopIteration:
                break
            profile.add("loader_collate", time.perf_counter() - started)
            _one_step(model, optimizer, batch, mean, std, profile)
            measured_graphs += int(batch.num_graphs)
            measured_steps += 1
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        status = "oom"
        error = str(exc).splitlines()[0][:500]
        torch.cuda.synchronize()

    peak_allocated = int(torch.cuda.max_memory_allocated())
    peak_reserved = int(torch.cuda.max_memory_reserved())
    stage = profile.to_dict()["stages"]
    compute_seconds = sum(
        stage[name]["total_seconds"]
        for name in ("h2d", "forward_loss", "backward_optimizer")
    )
    end_to_end_seconds = compute_seconds + stage["loader_collate"]["total_seconds"]
    result = {
        "format": FORMAT,
        "status": status,
        "error": error,
        "batch_size": batch_size,
        "measured_steps": measured_steps,
        "measured_graphs": measured_graphs,
        "graphs_per_second_compute": (
            measured_graphs / compute_seconds if compute_seconds > 0 else None
        ),
        "graphs_per_second_end_to_end": (
            measured_graphs / end_to_end_seconds if end_to_end_seconds > 0 else None
        ),
        "memory": {
            "total_bytes": total_memory,
            "baseline_allocated_bytes": baseline_allocated,
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            "reserve_fraction_from_peak_reserved": max(
                0.0, (total_memory - peak_reserved) / total_memory
            ),
        },
        "stages": stage,
    }
    del iterator, loader, optimizer, model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _summarize(results: list[dict]) -> dict:
    recommendations = {}
    for cohort in ("representative", "graph_size_tail"):
        safe = [
            row
            for row in results
            if row["cohort"] == cohort
            and row["status"] == "complete"
            and row["memory"]["reserve_fraction_from_peak_reserved"]
            >= MIN_MEMORY_RESERVE_FRACTION
        ]
        winner = max(
            safe,
            key=lambda row: row["graphs_per_second_end_to_end"],
            default=None,
        )
        recommendations[cohort] = None if winner is None else {
            "batch_size": winner["batch_size"],
            "graphs_per_second_end_to_end": winner[
                "graphs_per_second_end_to_end"
            ],
            "reserve_fraction": winner["memory"][
                "reserve_fraction_from_peak_reserved"
            ],
        }
    return recommendations


def run(
    output_root: Path,
    *,
    source_commit: str,
    source_archive_sha256: str,
    allocation_seconds: float,
    slurm_job_id: str,
) -> dict:
    import torch

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    determinism = configure_fp32_determinism(SEED)
    runtime = build_runtime_manifest(determinism)
    cache_root, manifest = find_cache(FIXED_500K_MANIFEST_SHA256)
    graphs, shards, shard_records = _load_train_only(cache_root, manifest)
    node_counts, edge_counts = _graph_sizes(shards)
    cohorts = _cohorts(node_counts, edge_counts)
    mean, std = _normalizer(shards)
    target_mean = torch.tensor(mean, dtype=torch.float32, device="cuda")
    target_std = torch.tensor(std, dtype=torch.float32, device="cuda")

    results = []
    for batch_size in BATCH_SIZES:
        for cohort, indices in cohorts.items():
            row = _profile_one(
                graphs, indices, batch_size, target_mean, target_std
            )
            row["cohort"] = cohort
            row["index_sha256"] = _hash_indices(indices)
            results.append(row)
            atomic_json(
                output_root / f"batch{batch_size}_{cohort}.json", row
            )
            print(
                f"profile batch={batch_size} cohort={cohort} "
                f"status={row['status']} graphs/s={row['graphs_per_second_end_to_end']} "
                f"peak_reserved={row['memory']['peak_reserved_bytes']}",
                flush=True,
            )

    summary = {
        "format": FORMAT,
        "status": "complete",
        "profiling_only": True,
        "scientific_result_produced": False,
        "training_checkpoint_produced": False,
        "model_bundle_produced": False,
        "scientific_contract_changed": False,
        "validation_executed": False,
        "development_role_read": False,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
        "test_challenge_role_read": False,
        "train_role_read": True,
        "source_commit": source_commit,
        "source_archive_sha256": source_archive_sha256,
        "slurm_job_id": slurm_job_id,
        "allocation_seconds": float(allocation_seconds),
        "cache_manifest_sha256": FIXED_500K_MANIFEST_SHA256,
        "train_rows": len(graphs),
        "parameters": PARAMETERS,
        "batch_sizes": list(BATCH_SIZES),
        "warmup_rows_per_cohort": WARMUP_ROWS,
        "representative_rows": REPRESENTATIVE_ROWS,
        "tail_rows": TAIL_ROWS,
        "node_count_quantiles": {
            str(q): float(torch.quantile(node_counts.float(), q))
            for q in (0.5, 0.9, 0.95, 0.99, 1.0)
        },
        "edge_count_quantiles": {
            str(q): float(torch.quantile(edge_counts.float(), q))
            for q in (0.5, 0.9, 0.95, 0.99, 1.0)
        },
        "cohort_index_sha256": {
            name: _hash_indices(indices) for name, indices in cohorts.items()
        },
        "train_shards": shard_records,
        "runtime_manifest": runtime,
        "minimum_memory_reserve_fraction": MIN_MEMORY_RESERVE_FRACTION,
        "recommendations_for_future_contract_study_only": _summarize(results),
        "results": results,
        "stage_notes": {
            "validation": "not run; profiling reads train role only",
            "checkpoint_hash_archive": "measured while writing and hashing profiling payload only",
            "allocation": "SLURM SubmitTime-to-StartTime supplied by launcher",
        },
    }
    payload_path = output_root / "profiling_payload.json"
    io_started = time.perf_counter()
    atomic_json(payload_path, summary)
    payload_sha256 = sha256_file(payload_path)
    io_seconds = time.perf_counter() - io_started
    summary["profiling_payload_sha256"] = payload_sha256
    summary["checkpoint_hash_archive_seconds"] = io_seconds
    atomic_json(output_root / "profile_summary.json", summary)
    artifacts = {
        path.name: sha256_file(path)
        for path in sorted(output_root.glob("*.json"))
        if path.name != "completion_manifest.json"
    }
    completion = {**summary, "artifact_sha256": artifacts}
    atomic_json(output_root / "completion_manifest.json", completion)
    return completion
