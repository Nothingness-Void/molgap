"""Bounded batch-128 capacity probe for the Xi'an local-geometry screen."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from molgap.pcqm_local_geometry_pretraining import (
    EXPECTED_MODEL_PARAMETERS,
    LEARNING_RATE,
    LocalGeometryHeads,
    PARENT_GEOMETRY_CACHE_SHA256,
    WEIGHT_DECAY,
    _atomic_torch_save,
    _find_cache,
    _forward,
    _geometry_loss,
    _make_encoder,
    _set_seed,
    _target_stats,
    _torch_load,
    _verify_parent_manifest,
    atomic_json,
    sha256_file,
)


BATCH_SIZE = 128
MEASURED_BATCHES = 32
WARMUP_BATCHES = 4
MIN_MEMORY_RESERVE_FRACTION = 0.15


def _load_probe_graphs(graph_root: Path, required_graphs: int):
    graph_root, manifest = _find_cache(
        "molgap-pcqm-gap100k-etkdg-geometry-cache-v1", graph_root
    )
    _verify_parent_manifest(manifest)
    graphs = []
    shards = []
    for item in manifest["shards"]:
        if item["role"] != "train":
            continue
        path = graph_root / item["file"]
        if sha256_file(path) != item["sha256"]:
            raise RuntimeError(f"Geometry shard hash changed: {item['file']}")
        payload = _torch_load(path, map_location="cpu", weights_only=False)
        if len(payload) != int(item["graph_count"]):
            raise RuntimeError(f"Geometry shard count changed: {item['file']}")
        graphs.extend(payload)
        shards.append({"file": item["file"], "sha256": item["sha256"]})
        if len(graphs) >= required_graphs:
            break
    if len(graphs) < required_graphs:
        raise RuntimeError("Not enough accepted train graphs for capacity probe")
    return graphs[:required_graphs], manifest, shards


def _run_mode(graphs, *, mode: str, target_mean: float, target_std: float, device):
    import torch
    import torch.nn.functional as functional
    from torch_geometric.loader import DataLoader

    _set_seed(42)
    model = _make_encoder().to(device)
    if sum(parameter.numel() for parameter in model.parameters()) != EXPECTED_MODEL_PARAMETERS:
        raise RuntimeError("EdgeState parameter contract changed")
    heads = LocalGeometryHeads(model) if mode == "local_geometry" else None
    if heads is not None:
        heads.module = heads.module.to(device)
    parameters = list(model.parameters())
    if heads is not None:
        parameters += list(heads.module.parameters())
    optimizer = torch.optim.AdamW(
        parameters, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    loader = DataLoader(
        graphs,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    mean = torch.tensor(target_mean, device=device)
    std = torch.tensor(target_std, device=device)
    finite = True
    measured_graphs = 0
    measured_steps = 0
    started = None
    torch.cuda.reset_peak_memory_stats(device)
    for step, batch in enumerate(loader):
        if step >= WARMUP_BATCHES + MEASURED_BATCHES:
            break
        batch = batch.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if mode == "gap":
            target = (batch.y.view(-1) - mean) / std
            loss = functional.l1_loss(_forward(model, batch), target)
        else:
            _forward(model, batch)
            loss, _, _ = _geometry_loss(heads, batch)
        if not bool(torch.isfinite(loss)):
            finite = False
            break
        loss.backward()
        finite = finite and all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in parameters
        )
        if not finite:
            break
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        if step + 1 == WARMUP_BATCHES:
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
        elif step + 1 > WARMUP_BATCHES:
            measured_graphs += int(batch.num_graphs)
            measured_steps += 1
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started if started is not None else float("nan")
    total_memory = int(torch.cuda.get_device_properties(device).total_memory)
    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    if heads is not None:
        heads.close()
    return {
        "mode": mode,
        "finite_forward_backward": bool(finite),
        "measured_steps": measured_steps,
        "measured_graphs": measured_graphs,
        "elapsed_s": elapsed,
        "graphs_per_s": measured_graphs / elapsed,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "total_device_bytes": total_memory,
        "reserved_memory_headroom_fraction": 1.0 - peak_reserved / total_memory,
    }, model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()

    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expected exactly one SCNet DCU")
    required_graphs = BATCH_SIZE * (WARMUP_BATCHES + MEASURED_BATCHES)
    graphs, manifest, shards = _load_probe_graphs(args.graph_root, required_graphs)
    target_mean, target_std = _target_stats(graphs)
    device = torch.device("cuda:0")
    rows = []
    checkpoint_model = None
    for mode in ("gap", "local_geometry"):
        row, model = _run_mode(
            graphs,
            mode=mode,
            target_mean=target_mean,
            target_std=target_std,
            device=device,
        )
        rows.append(row)
        if mode == "local_geometry":
            checkpoint_model = model
        else:
            del model
        torch.cuda.empty_cache()
    if checkpoint_model is None:
        raise RuntimeError("Local-geometry capacity mode did not run")
    args.output_root.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_root / "batch128_probe_model.pt"
    _atomic_torch_save(checkpoint, checkpoint_model.state_dict())
    accepted = all(
        row["finite_forward_backward"]
        and row["measured_steps"] == MEASURED_BATCHES
        and math.isfinite(row["graphs_per_s"])
        and row["reserved_memory_headroom_fraction"] >= MIN_MEMORY_RESERVE_FRACTION
        for row in rows
    )
    payload = {
        "format": "molgap-pcqm-local-geometry-xian-batch128-capacity-v1",
        "accepted": accepted,
        "source_commit": args.source_commit,
        "architecture": "ogb_edge_state_structural_gps9",
        "inference_parameters": EXPECTED_MODEL_PARAMETERS,
        "batch_size": BATCH_SIZE,
        "warmup_batches": WARMUP_BATCHES,
        "measured_batches": MEASURED_BATCHES,
        "minimum_memory_reserve_fraction": MIN_MEMORY_RESERVE_FRACTION,
        "geometry_cache_aggregate_sha256": manifest["aggregate_sha256"],
        "train_shards_read": shards,
        "train_graphs_loaded": len(graphs),
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "torch_hip": torch.version.hip,
        "rows": rows,
        "checkpoint_sha256": sha256_file(checkpoint),
        "official_validation_role_read": False,
        "test_dev_role_read": False,
    }
    atomic_json(args.output_root / "capacity.json", payload)
    print(json.dumps(payload, indent=2), flush=True)
    if not accepted:
        raise RuntimeError("Batch-128 capacity gate failed")


if __name__ == "__main__":
    main()
