"""V5 profiling-only audit of GPTrans dynamic versus cached shortest paths."""
from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import random
import time
import types
from pathlib import Path

from .pcqm_gptrans_prenorm_500k import (
    BATCH_SIZE,
    PARAMETERS,
    SEED,
    make_model,
    optimizer_for,
    optimizer_step,
)
from .pcqm_gptrans_v4 import ExponentialMovingAverage, _state_sha256
from .pcqm_k1_pair_token_batch_profile import (
    _cohorts,
    _graph_sizes,
    _load_train_only,
    _loader,
    _normalizer,
)
from .pcqm_k1_scale import FIXED_500K_MANIFEST_SHA256
from .pcqm_k1_scale_runner import find_cache
from .training_reproducibility import (
    atomic_json,
    build_runtime_manifest,
    configure_fp32_determinism,
    sha256_file,
)


FORMAT = "molgap-gptrans-shortest-path-profile-v1"
COHORTS = ("representative", "graph_size_tail")
WARMUP_STEPS = 2
MEASURED_STEPS = 8
REPEATS = 3
PATH_STAGE_REPEATS = 20
MAX_EQUIVALENCE_DELTA = 1.0e-7
MIN_SPEEDUP_FOR_FOLLOWUP = 1.05


def _hash_indices(indices: list[int]) -> str:
    return hashlib.sha256(
        ",".join(str(int(value)) for value in indices).encode("ascii")
    ).hexdigest()


def _state_digest(state: dict[str, object]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _max_state_delta(left: dict, right: dict) -> tuple[float, str]:
    maximum, maximum_name = 0.0, ""
    if left.keys() != right.keys():
        raise RuntimeError("State keys changed between exact implementations")
    for name in left:
        a, b = left[name], right[name]
        if a.shape != b.shape or a.dtype != b.dtype:
            raise RuntimeError(f"State tensor identity changed: {name}")
        if not a.is_floating_point():
            if not bool(a.equal(b)):
                raise RuntimeError(f"Integer state changed: {name}")
            continue
        delta = float((a - b).abs().max())
        if delta > maximum:
            maximum, maximum_name = delta, name
    return maximum, maximum_name


def _adjacency_context(model, batch):
    import torch
    from torch_geometric.utils import to_dense_batch

    _, node_mask = to_dense_batch(batch.x.long(), batch.batch)
    batch_size, max_nodes = node_mask.shape
    edge_batch, edge_src, edge_dst = model._local_edges(
        batch.edge_index, batch.batch, int(batch.x.shape[0])
    )
    adjacency = torch.zeros(
        (batch_size, max_nodes, max_nodes),
        dtype=torch.bool,
        device=batch.x.device,
    )
    adjacency[edge_batch, edge_src, edge_dst] = True
    pair_mask = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
    return adjacency, pair_mask


def _cache_distances(model, batches) -> tuple[list, dict]:
    import torch

    distances = []
    started = time.perf_counter()
    for cpu_batch in batches:
        batch = cpu_batch.clone().to("cuda", non_blocking=True)
        adjacency, pair_mask = _adjacency_context(model, batch)
        value = model._shortest_path(adjacency, pair_mask)
        torch.cuda.synchronize()
        cpu_value = value.cpu()
        try:
            cpu_value = cpu_value.pin_memory()
        except RuntimeError:
            pass
        distances.append(cpu_value)
        del batch, adjacency, pair_mask, value
    elapsed = time.perf_counter() - started
    return distances, {
        "offline_build_seconds": elapsed,
        "cache_bytes": sum(value.numel() * value.element_size() for value in distances),
        "dtype": str(distances[0].dtype),
        "batch_shapes": [list(value.shape) for value in distances],
    }


def _install_cached_path(model) -> None:
    def cached_shortest_path(self, adjacency, pair_mask):
        del pair_mask
        cached = self._profile_cached_distance.to(
            adjacency.device, non_blocking=True
        )
        if cached.shape != adjacency.shape:
            raise RuntimeError(
                f"Cached shortest-path shape changed: {cached.shape} != {adjacency.shape}"
            )
        return cached

    model._shortest_path = types.MethodType(cached_shortest_path, model)


def _profile_path_stage(model, cpu_batch, cached_distance) -> dict:
    import torch

    batch = cpu_batch.clone().to("cuda", non_blocking=True)
    adjacency, pair_mask = _adjacency_context(model, batch)
    torch.cuda.synchronize()
    started = time.perf_counter()
    dynamic = None
    for _ in range(PATH_STAGE_REPEATS):
        dynamic = model._shortest_path(adjacency, pair_mask)
    torch.cuda.synchronize()
    dynamic_seconds = time.perf_counter() - started

    started = time.perf_counter()
    transferred = None
    for _ in range(PATH_STAGE_REPEATS):
        transferred = cached_distance.to(adjacency.device, non_blocking=True)
    torch.cuda.synchronize()
    transfer_seconds = time.perf_counter() - started
    exact = bool(torch.equal(dynamic, transferred))
    del batch, adjacency, pair_mask, dynamic, transferred
    return {
        "repeats": PATH_STAGE_REPEATS,
        "dynamic_seconds": dynamic_seconds,
        "cached_h2d_seconds": transfer_seconds,
        "speedup": dynamic_seconds / max(transfer_seconds, 1.0e-12),
        "distance_exact": exact,
    }


def _run_variant(batches, cached_distances, *, variant: str, mean, std) -> tuple:
    import torch

    configure_fp32_determinism(SEED)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = make_model("reference").to("cuda").train()
    if variant == "cached":
        _install_cached_path(model)
    optimizer = optimizer_for(model)
    ema = ExponentialMovingAverage(model)
    initial_sha256 = _state_sha256(model)
    losses, h2d_seconds, step_seconds = [], 0.0, 0.0
    measured_graphs = 0

    for step, cpu_batch in enumerate(batches):
        work = cpu_batch.clone()
        started = time.perf_counter()
        work = work.to("cuda", non_blocking=True)
        if variant == "cached":
            model._profile_cached_distance = cached_distances[step]
        torch.cuda.synchronize()
        transfer = time.perf_counter() - started

        started = time.perf_counter()
        loss = optimizer_step(model, optimizer, ema, work, mean, std)
        torch.cuda.synchronize()
        compute = time.perf_counter() - started
        losses.append(float(loss))
        if step >= WARMUP_STEPS:
            h2d_seconds += transfer
            step_seconds += compute
            measured_graphs += int(work.num_graphs)
        del work, loss

    model_state = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }
    ema_state = {
        name: value.detach().cpu().clone()
        for name, value in ema.state_dict().items()
    }
    total_seconds = h2d_seconds + step_seconds
    result = {
        "variant": variant,
        "initial_state_sha256": initial_sha256,
        "final_state_sha256": _state_digest(model_state),
        "final_ema_sha256": _state_digest(ema_state),
        "losses": losses,
        "measured_graphs": measured_graphs,
        "h2d_seconds": h2d_seconds,
        "forward_backward_optimizer_ema_seconds": step_seconds,
        "end_to_end_seconds": total_seconds,
        "graphs_per_second": measured_graphs / total_seconds,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
    }
    del optimizer, ema, model
    gc.collect()
    torch.cuda.empty_cache()
    return result, model_state, ema_state


def _profile_cohort(graphs, indices, mean, std, cohort: str) -> dict:
    import torch

    selected = indices[: (WARMUP_STEPS + MEASURED_STEPS) * BATCH_SIZE]
    loader_started = time.perf_counter()
    batches = list(_loader(graphs, selected, BATCH_SIZE))
    loader_seconds = time.perf_counter() - loader_started
    if len(batches) != WARMUP_STEPS + MEASURED_STEPS:
        raise RuntimeError(f"Unexpected profiling batch count: {len(batches)}")

    configure_fp32_determinism(SEED)
    cache_model = make_model("reference").to("cuda").eval()
    cached_distances, cache = _cache_distances(cache_model, batches)
    path_stage = _profile_path_stage(cache_model, batches[0], cached_distances[0])
    del cache_model
    gc.collect()
    torch.cuda.empty_cache()

    rng = random.Random(SEED + (0 if cohort == "representative" else 1))
    repeat_rows = []
    speedups = []
    all_equivalent = path_stage["distance_exact"]
    for repeat in range(REPEATS):
        order = ["dynamic", "cached"]
        rng.shuffle(order)
        outputs = {}
        states = {}
        emas = {}
        for variant in order:
            outputs[variant], states[variant], emas[variant] = _run_variant(
                batches,
                cached_distances,
                variant=variant,
                mean=mean,
                std=std,
            )
        model_delta, model_name = _max_state_delta(
            states["dynamic"], states["cached"]
        )
        ema_delta, ema_name = _max_state_delta(emas["dynamic"], emas["cached"])
        loss_delta = max(
            abs(left - right)
            for left, right in zip(
                outputs["dynamic"]["losses"],
                outputs["cached"]["losses"],
                strict=True,
            )
        )
        equivalent = max(model_delta, ema_delta, loss_delta) <= MAX_EQUIVALENCE_DELTA
        all_equivalent = all_equivalent and equivalent
        speedup = (
            outputs["dynamic"]["end_to_end_seconds"]
            / outputs["cached"]["end_to_end_seconds"]
        )
        speedups.append(speedup)
        repeat_rows.append(
            {
                "repeat": repeat,
                "execution_order": order,
                "dynamic": outputs["dynamic"],
                "cached": outputs["cached"],
                "equivalence": {
                    "passed": equivalent,
                    "maximum_loss_delta": loss_delta,
                    "maximum_model_delta": model_delta,
                    "maximum_model_delta_name": model_name,
                    "maximum_ema_delta": ema_delta,
                    "maximum_ema_delta_name": ema_name,
                    "maximum_allowed_delta": MAX_EQUIVALENCE_DELTA,
                },
                "end_to_end_speedup": speedup,
            }
        )
        del states, emas

    speedups_sorted = sorted(speedups)
    median_speedup = speedups_sorted[len(speedups_sorted) // 2]
    return {
        "cohort": cohort,
        "status": "complete",
        "batch_size": BATCH_SIZE,
        "indices_sha256": _hash_indices(selected),
        "profiled_rows": len(selected),
        "loader_collate_seconds": loader_seconds,
        "cache": cache,
        "path_stage": path_stage,
        "repeats": repeat_rows,
        "all_equivalence_checks_passed": all_equivalent,
        "median_end_to_end_speedup": median_speedup,
        "followup_speed_gate_passed": median_speedup >= MIN_SPEEDUP_FOR_FOLLOWUP,
    }


def run(
    output_root: Path,
    *,
    source_commit: str,
    source_archive_sha256: str,
    allocation_seconds: float,
    slurm_job_id: str,
) -> dict:
    import torch

    output_root.mkdir(parents=True, exist_ok=True)
    settings = configure_fp32_determinism(SEED)
    runtime = build_runtime_manifest(settings)
    root, manifest = find_cache(FIXED_500K_MANIFEST_SHA256)
    graphs, shards, shard_records = _load_train_only(root, manifest)
    node_counts, edge_counts = _graph_sizes(shards)
    cohorts = _cohorts(node_counts, edge_counts)
    mean_value, std_value = _normalizer(shards)
    mean = torch.tensor(mean_value, device="cuda")
    std = torch.tensor(std_value, device="cuda")

    results = []
    for cohort in COHORTS:
        try:
            row = _profile_cohort(
                graphs, cohorts[cohort], mean, std, cohort
            )
        except RuntimeError as error:
            if "out of memory" not in str(error).lower():
                raise
            torch.cuda.empty_cache()
            row = {
                "cohort": cohort,
                "status": "oom",
                "error": str(error).splitlines()[0][:500],
                "batch_size": BATCH_SIZE,
            }
        results.append(row)
        atomic_json(output_root / f"{cohort}.json", row)
        print(
            f"PROFILE cohort={cohort} status={row['status']} "
            f"speedup={row.get('median_end_to_end_speedup')}",
            flush=True,
        )

    complete = [row for row in results if row["status"] == "complete"]
    representative = next(
        (row for row in complete if row["cohort"] == "representative"), None
    )
    eligible = bool(
        representative
        and all(row["all_equivalence_checks_passed"] for row in complete)
        and representative["followup_speed_gate_passed"]
    )
    summary = {
        "format": FORMAT,
        "status": "complete",
        "profiling_only": True,
        "scientific_result_produced": False,
        "training_checkpoint_produced": False,
        "model_bundle_produced": False,
        "scientific_contract_changed": False,
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
        "parameter_count": PARAMETERS,
        "precision": "fp32",
        "physical_batch_per_device": BATCH_SIZE,
        "runtime_manifest": runtime,
        "train_shards": shard_records,
        "node_count_quantiles": {
            str(q): float(torch.quantile(node_counts.float(), q))
            for q in (0.5, 0.9, 0.95, 0.99, 1.0)
        },
        "results": results,
        "exact_cached_shortest_path_followup_eligible": eligible,
        "followup_note": (
            "Eligibility permits a separately versioned exact implementation and "
            "runtime-certificate study only; it does not alter a scientific result."
        ),
    }
    atomic_json(output_root / "profile_summary.json", summary)
    artifacts = {
        path.name: sha256_file(path)
        for path in sorted(output_root.glob("*.json"))
        if path.name != "completion_manifest.json"
    }
    completion = {**summary, "artifact_sha256": artifacts}
    atomic_json(output_root / "completion_manifest.json", completion)
    return completion
