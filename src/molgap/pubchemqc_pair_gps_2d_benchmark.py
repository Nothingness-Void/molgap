"""Short A100 throughput benchmark for the PubChemQC-100K PairGPS2D path.

The benchmark consumes only the frozen training role.  It performs one training
epoch per runtime configuration, writes each result atomically, and never
persists weights or evaluates validation/test targets.  Its purpose is to
select a hardware-efficient execution contract before another long run.
"""

from __future__ import annotations

import json
import math
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from .pubchemqc_pair_gps_2d import _atomic_json, _size_bucket_batches, sha256_file
from .pubchemqc_pair_gps_2d_screen import (
    FROZEN_SPLIT_ROWS,
    FROZEN_SPLIT_SHA256,
    FORMAT as SCREEN_FORMAT,
    _load_screen_graphs,
    _model,
    _read_split,
    _set_seed,
)


FORMAT = "molgap-pubchemqc-pair-gps-2d-a100-benchmark-v1"
SUPPORTED_PRECISIONS = {"fp32", "tf32", "bf16"}


def parse_benchmark_specs(values: Iterable[str]) -> list[dict[str, object]]:
    """Parse ``precision:batch_size:num_workers`` benchmark specifications."""
    specs = []
    seen = set()
    for value in values:
        parts = value.split(":")
        if len(parts) != 3:
            raise ValueError(f"invalid benchmark spec {value!r}")
        precision = parts[0].lower()
        batch_size = int(parts[1])
        num_workers = int(parts[2])
        if precision not in SUPPORTED_PRECISIONS:
            raise ValueError(f"unsupported benchmark precision: {precision}")
        if batch_size < 1 or num_workers < 0:
            raise ValueError("batch_size must be positive and num_workers nonnegative")
        key = (precision, batch_size, num_workers)
        if key in seen:
            raise ValueError(f"duplicate benchmark spec: {value}")
        seen.add(key)
        specs.append(
            {
                "precision": precision,
                "batch_size": batch_size,
                "num_workers": num_workers,
            }
        )
    if not specs:
        raise ValueError("at least one benchmark spec is required")
    return specs


def _spec_key(spec: dict[str, object]) -> str:
    return (
        f"{spec['precision']}_bs{int(spec['batch_size'])}"
        f"_workers{int(spec['num_workers'])}"
    )


def _set_precision(precision: str) -> None:
    allow_tf32 = precision == "tf32"
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    torch.set_float32_matmul_precision("high" if allow_tf32 else "highest")


def _autocast(precision: str):
    if precision == "bf16":
        return torch.amp.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def _new_optimizer(model: nn.Module, learning_rate: float) -> tuple[torch.optim.Optimizer, bool]:
    kwargs = {
        "lr": learning_rate,
        "weight_decay": 1e-5,
    }
    try:
        return torch.optim.AdamW(model.parameters(), fused=True, **kwargs), True
    except (RuntimeError, TypeError):
        return torch.optim.AdamW(model.parameters(), **kwargs), False


def _run_configuration(
    *,
    train_graphs: list,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    spec: dict[str, object],
    learning_rate: float,
    seed: int,
) -> dict[str, object]:
    precision = str(spec["precision"])
    batch_size = int(spec["batch_size"])
    num_workers = int(spec["num_workers"])
    _set_precision(precision)
    _set_seed(seed)
    device = torch.device("cuda")
    model, model_config = _model("pair_gps_2d", device)
    optimizer, fused_optimizer = _new_optimizer(model, learning_rate)
    criterion = nn.L1Loss()
    batches = _size_bucket_batches(train_graphs, batch_size, seed=seed)
    loader_kwargs: dict[str, object] = {
        "batch_sampler": batches,
        "num_workers": num_workers,
        "pin_memory": True,
    }
    if num_workers:
        loader_kwargs.update(
            {
                "persistent_workers": True,
                "prefetch_factor": 2,
            }
        )
    loader = DataLoader(train_graphs, **loader_kwargs)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started_unix = time.time()
    started = time.perf_counter()
    total_loss = 0.0
    graph_count = 0
    step_count = 0
    largest_batch_nodes = 0
    largest_graph_nodes = 0
    first_loss = None
    last_loss = None
    model.train()
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        target = (batch.y.view(-1, 3).float() - target_mean) / target_std
        with _autocast(precision):
            prediction = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = criterion(prediction, target)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite {precision} loss at step {step_count}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(grad_norm):
            raise RuntimeError(f"non-finite {precision} gradient at step {step_count}")
        optimizer.step()
        value = float(loss.detach())
        first_loss = value if first_loss is None else first_loss
        last_loss = value
        total_loss += value * int(batch.num_graphs)
        graph_count += int(batch.num_graphs)
        step_count += 1
        node_counts = torch.bincount(batch.batch)
        largest_batch_nodes = max(largest_batch_nodes, int(batch.num_nodes))
        largest_graph_nodes = max(largest_graph_nodes, int(node_counts.max().item()))

    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    finished_unix = time.time()
    return {
        "status": "complete",
        "key": _spec_key(spec),
        "precision": precision,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "model_config": model_config,
        "fused_adamw": fused_optimizer,
        "started_unix": started_unix,
        "finished_unix": finished_unix,
        "elapsed_s": elapsed,
        "steps": step_count,
        "graphs": graph_count,
        "graphs_per_s": graph_count / elapsed,
        "steps_per_s": step_count / elapsed,
        "train_normalized_l1": total_loss / max(graph_count, 1),
        "first_batch_loss": first_loss,
        "last_batch_loss": last_loss,
        "largest_batch_nodes": largest_batch_nodes,
        "largest_graph_nodes": largest_graph_nodes,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "finite": True,
        "test_role_read": False,
    }


def benchmark_pair_gps_2d_a100(
    *,
    split_csv: Path,
    cache_dir: Path,
    output_path: Path,
    specs: list[dict[str, object]],
    learning_rate: float = 2e-4,
    seed: int = 42,
) -> dict[str, object]:
    """Benchmark complete training-role epochs under several A100 contracts."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the PairGPS2D A100 benchmark")
    device_name = torch.cuda.get_device_name(0)
    if "A100" not in device_name.upper():
        raise RuntimeError(f"A100 required, got {device_name}")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("the assigned CUDA device does not support BF16")

    normalized_specs = parse_benchmark_specs(
        f"{spec['precision']}:{int(spec['batch_size'])}:{int(spec['num_workers'])}"
        for spec in specs
    )
    contract = {
        "format": FORMAT,
        "screen_format": SCREEN_FORMAT,
        "split_sha256": FROZEN_SPLIT_SHA256,
        "cache_acceptance_sha256": sha256_file(cache_dir / "acceptance.json"),
        "candidate": "pair_gps_2d",
        "role": "train_only",
        "train_rows": FROZEN_SPLIT_ROWS["train"],
        "learning_rate": learning_rate,
        "weight_decay": 1e-5,
        "gradient_clip": 1.0,
        "seed": seed,
        "specs": normalized_specs,
        "test_role_read": False,
    }
    previous_results: list[dict[str, object]] = []
    if output_path.exists():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        if existing.get("contract") != contract:
            raise RuntimeError("A100 benchmark contract changed")
        if existing.get("status") == "complete":
            return existing
        previous_results = list(existing.get("results", []))

    rows = _read_split(split_csv)
    acceptance, graphs = _load_screen_graphs(cache_dir)
    role_by_idx = {int(row["source_idx"]): str(row["role"]) for row in rows}
    train_graphs = [
        graph
        for graph in graphs
        if role_by_idx[int(graph.source_idx.view(-1)[0])] == "train"
    ]
    if len(train_graphs) != FROZEN_SPLIT_ROWS["train"]:
        raise RuntimeError("A100 benchmark training graph count changed")
    node_counts = np.asarray([int(graph.num_nodes) for graph in train_graphs])
    target_mean = torch.tensor(
        acceptance["target_mean"], dtype=torch.float32, device="cuda"
    )
    target_std = torch.tensor(
        acceptance["target_std"], dtype=torch.float32, device="cuda"
    )
    gpu = torch.cuda.get_device_properties(0)
    base = {
        "contract": contract,
        "device": {
            "name": device_name,
            "total_memory_bytes": int(gpu.total_memory),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "bf16_supported": torch.cuda.is_bf16_supported(),
        },
        "train_node_count": {
            "min": int(node_counts.min()),
            "p50": float(np.percentile(node_counts, 50)),
            "p90": float(np.percentile(node_counts, 90)),
            "p99": float(np.percentile(node_counts, 99)),
            "max": int(node_counts.max()),
        },
        "test_role_read": False,
    }
    results_by_key = {str(result["key"]): result for result in previous_results}
    ordered_results = list(previous_results)
    for spec in normalized_specs:
        key = _spec_key(spec)
        if key in results_by_key:
            print(f"reuse completed benchmark {key}", flush=True)
            continue
        print(f"start benchmark {key}", flush=True)
        try:
            result = _run_configuration(
                train_graphs=train_graphs,
                target_mean=target_mean,
                target_std=target_std,
                spec=spec,
                learning_rate=learning_rate,
                seed=seed,
            )
        except (torch.OutOfMemoryError, RuntimeError) as error:
            if "out of memory" not in str(error).lower():
                raise
            result = {
                "status": "oom",
                "key": key,
                **spec,
                "error": str(error),
                "finite": False,
                "test_role_read": False,
            }
        ordered_results.append(result)
        results_by_key[key] = result
        torch.cuda.empty_cache()
        _atomic_json({"status": "running", **base, "results": ordered_results}, output_path)
        if result["status"] == "complete":
            print(
                f"finish benchmark {key}: {result['graphs_per_s']:.1f} graphs/s "
                f"peak={result['peak_reserved_bytes'] / (1 << 30):.2f} GiB",
                flush=True,
            )
        else:
            print(f"finish benchmark {key}: OOM", flush=True)

    successful = [result for result in ordered_results if result["status"] == "complete"]
    if not successful:
        raise RuntimeError("all A100 benchmark configurations failed")
    total_memory = int(gpu.total_memory)
    headroom = [
        result
        for result in successful
        if int(result["peak_reserved_bytes"]) <= math.floor(total_memory * 0.85)
    ]
    recommendation_pool = headroom or successful
    recommended = max(recommendation_pool, key=lambda result: float(result["graphs_per_s"]))
    final = {
        "status": "complete",
        **base,
        "results": ordered_results,
        "selection_rule": "highest graphs_per_s with at least 15% reserved-memory headroom",
        "recommended_key": recommended["key"],
    }
    _atomic_json(final, output_path)
    return final
