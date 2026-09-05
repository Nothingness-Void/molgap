"""Measure Xi'an Card2 throughput on accepted PCQM train-role graphs only."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path


CANDIDATE = "ogb_distance_angle_triangle_edge_state_graph_state9"
EXPECTED_PARAMS = 3_665_809
EXPECTED_CACHE_AGGREGATE = (
    "3e4206fd239942ab79f9c4978cd4334f6025ba14e8f1cd1df78c20060a0d1d22"
)
EXPECTED_SHARDS = {
    "train-0000.pt": "2ed2408945a231e87e3527f9594a75a50c545e1ed3fccc967e7239aadefa4c0d",
    "train-0001.pt": "513655169060e82d747f0bba6e08c9ca30fb33e4490657f90fa08175481527fd",
}
T4_REFERENCE_GRAPHS_PER_S = 566.5467159302983


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_torch(path: Path):
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def set_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def synchronize(device) -> None:
    import torch

    torch.cuda.synchronize(device)


def forward(model, batch):
    return model(
        batch.x,
        batch.edge_index,
        batch.edge_attr,
        batch.batch,
        batch.random_walk_pe,
        batch.wedge_edge_ids,
        batch.edge_distance,
        batch.wedge_angle_cos,
        batch.geometry_valid,
    ).view(-1)


def load_train_subset(cache_root: Path) -> list:
    manifest = json.loads((cache_root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("aggregate_sha256") != EXPECTED_CACHE_AGGREGATE:
        raise RuntimeError("cache aggregate identity changed")
    if manifest.get("official_validation_role_read") is not False:
        raise RuntimeError("cache role contract changed")
    shard_rows = {row["file"]: row for row in manifest["shards"]}
    graphs = []
    for name, expected_sha in EXPECTED_SHARDS.items():
        row = shard_rows.get(name)
        if row is None or row.get("role") != "train" or row.get("sha256") != expected_sha:
            raise RuntimeError(f"train shard manifest changed: {name}")
        path = cache_root / name
        if sha256_file(path) != expected_sha:
            raise RuntimeError(f"train shard bytes changed: {name}")
        payload = load_torch(path)
        if len(payload) != int(row["graph_count"]):
            raise RuntimeError(f"train shard graph count changed: {name}")
        graphs.extend(payload)
    if len(graphs) != 10_000:
        raise RuntimeError("throughput subset must contain exactly 10,000 train graphs")
    return graphs


def benchmark_batch(graphs: list, batch_size: int, device) -> dict:
    import torch
    from torch_geometric.loader import DataLoader
    from molgap.pcqm_gap_architecture import make_pcqm_gap_encoder

    set_seed(42)
    model = make_pcqm_gap_encoder(CANDIDATE).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != EXPECTED_PARAMS:
        raise RuntimeError(f"parameter count changed: {parameter_count}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.6e-4, weight_decay=1.0e-6)
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=True, num_workers=0)

    model.train()
    for index, batch in enumerate(loader):
        if index >= 10:
            break
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.l1_loss(forward(model, batch), batch.y.view(-1).float())
        loss.backward()
        optimizer.step()
    synchronize(device)

    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    count = 0
    loss_sum = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.l1_loss(forward(model, batch), batch.y.view(-1).float())
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss for batch size {batch_size}")
        loss.backward()
        optimizer.step()
        count += int(batch.num_graphs)
        loss_sum += float(loss.item()) * int(batch.num_graphs)
    synchronize(device)
    elapsed = time.perf_counter() - started
    throughput = count / elapsed
    return {
        "batch_size": batch_size,
        "measured_train_graphs": count,
        "elapsed_s": elapsed,
        "train_graphs_per_s": throughput,
        "mean_batch_weighted_loss_eV": loss_sum / count,
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "t4_reference_graphs_per_s": T4_REFERENCE_GRAPHS_PER_S,
        "throughput_ratio_vs_t4_batch48": throughput / T4_REFERENCE_GRAPHS_PER_S,
        "matches_or_exceeds_t4": throughput >= T4_REFERENCE_GRAPHS_PER_S,
        "within_10_percent_of_t4": throughput >= 0.9 * T4_REFERENCE_GRAPHS_PER_S,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[48, 96, 192])
    args = parser.parse_args()

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("no DCU-compatible device visible")
    if args.batch_sizes != [48, 96, 192]:
        raise ValueError("Xi'an throughput gate batch-size contract changed")
    device = torch.device("cuda:0")
    graphs = load_train_subset(args.cache_root)
    rows = [benchmark_batch(graphs, size, device) for size in args.batch_sizes]
    if not all(math.isfinite(row["train_graphs_per_s"]) for row in rows):
        raise RuntimeError("non-finite throughput")
    payload = {
        "format": "molgap-xian-card2-graphstate-throughput-v1",
        "accepted_runtime": True,
        "candidate": CANDIDATE,
        "parameter_count": EXPECTED_PARAMS,
        "precision": "fp32",
        "optimizer": "AdamW",
        "learning_rate": 1.6e-4,
        "weight_decay": 1.0e-6,
        "cache_aggregate_sha256": EXPECTED_CACHE_AGGREGATE,
        "train_shards_read": list(EXPECTED_SHARDS),
        "train_graphs_loaded": len(graphs),
        "device_name": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "torch_hip": torch.version.hip,
        "rows": rows,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
