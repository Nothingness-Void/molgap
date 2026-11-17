"""Validate aligned GPS/RWSE caches and one exact-shape accelerator step."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import torch
from torch_geometric.data import Batch

from molgap.gps import (
    EdgeStateStructuralGPSWrapper,
    GatedStructuralGPSWrapper,
    GPSWrapper,
    NormalizedStructuralGPSWrapper,
    StructuralGPSWrapper,
)
from molgap.structural_encoding import sha256


def _atomic_json(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _source_idx(graph) -> int:
    return int(graph.source_idx.view(-1)[0])


def _split_indices(path: Path) -> tuple[set[int], dict[str, int]]:
    roles = {"train": 0, "validation": 0, "test": 0}
    indices = set()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not {"source_idx", "split"}.issubset(reader.fieldnames or ()):
            raise ValueError("Split CSV requires source_idx and split")
        for row in reader:
            index = int(row["source_idx"])
            role = row["split"].strip().lower()
            if role not in roles or index in indices:
                raise ValueError("Split CSV contains an invalid role or duplicate index")
            roles[role] += 1
            indices.add(index)
    return indices, roles


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-graph", type=Path, required=True)
    parser.add_argument("--rwse-graph", type=Path, required=True)
    parser.add_argument("--split-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Structural GPS preflight requires an accelerator")
    base = torch.load(args.base_graph, map_location="cpu", weights_only=False)
    structural = torch.load(args.rwse_graph, map_location="cpu", weights_only=False)
    if len(base) != len(structural) or not base:
        raise ValueError("Base and RWSE graph caches have different row counts")
    graph_indices = set()
    for position, (left, right) in enumerate(zip(base, structural)):
        left_index, right_index = _source_idx(left), _source_idx(right)
        if left_index != right_index or left_index in graph_indices:
            raise ValueError(f"Graph identity mismatch at position {position}")
        if not torch.equal(left.y, right.y):
            raise ValueError(f"Graph target mismatch at position {position}")
        positional = getattr(right, "random_walk_pe", None)
        if positional is None or positional.shape != (right.num_nodes, 16):
            raise ValueError(f"RWSE shape mismatch at position {position}")
        if not torch.isfinite(positional).all():
            raise ValueError(f"RWSE non-finite at position {position}")
        graph_indices.add(left_index)
    split_indices, split_counts = _split_indices(args.split_csv)
    if graph_indices != split_indices:
        raise ValueError("Graph cache and split CSV source_idx sets differ")

    configuration = {
        "hidden_channels": 192,
        "num_layers": 9,
        "num_heads": 4,
        "dropout": 0.05,
    }
    torch.manual_seed(42)
    gps = GPSWrapper(**configuration)
    torch.manual_seed(42)
    rwse = StructuralGPSWrapper(**configuration, rwse_dim=16)
    rwse_state = rwse.state_dict()
    for name, value in gps.state_dict().items():
        if not torch.equal(value, rwse_state[name]):
            raise RuntimeError(f"Shared initialization differs: {name}")
    gap_configuration = {**configuration, "n_targets": 1}
    torch.manual_seed(42)
    gap_rwse = StructuralGPSWrapper(**gap_configuration, rwse_dim=16)
    torch.manual_seed(42)
    normalized_gap_rwse = NormalizedStructuralGPSWrapper(
        **gap_configuration,
        rwse_dim=16,
        rwse_alpha_init=0.25,
    )
    gated_rwse = GatedStructuralGPSWrapper(**configuration, rwse_dim=16)
    edge_state_rwse = EdgeStateStructuralGPSWrapper(
        **configuration,
        rwse_dim=16,
        edge_state_channels=64,
    )
    normalized_state = normalized_gap_rwse.state_dict()
    for name, value in gap_rwse.state_dict().items():
        if not torch.equal(value, normalized_state[name]):
            raise RuntimeError(f"Normalized shared initialization differs: {name}")

    device = torch.device("cuda")
    gps = gps.to(device)
    rwse = rwse.to(device)
    normalized_gap_rwse = normalized_gap_rwse.to(device)
    gated_rwse = gated_rwse.to(device)
    edge_state_rwse = edge_state_rwse.to(device)
    batch_base = Batch.from_data_list(base[:8]).to(device)
    batch_rwse = Batch.from_data_list(structural[:8]).to(device)
    with torch.amp.autocast("cuda"):
        base_output = gps(
            batch_base.x,
            batch_base.edge_index,
            batch_base.edge_attr,
            batch_base.batch,
        )
        rwse_output = rwse(
            batch_rwse.x,
            batch_rwse.edge_index,
            batch_rwse.edge_attr,
            batch_rwse.batch,
            batch_rwse.random_walk_pe,
        )
        loss = torch.nn.functional.l1_loss(rwse_output, batch_rwse.y)
        normalized_gap_output = normalized_gap_rwse(
            batch_rwse.x,
            batch_rwse.edge_index,
            batch_rwse.edge_attr,
            batch_rwse.batch,
            batch_rwse.random_walk_pe,
        )
        normalized_gap_loss = torch.nn.functional.l1_loss(
            normalized_gap_output,
            batch_rwse.y[:, 2:3],
        )
        gated_output = gated_rwse(
            batch_rwse.x,
            batch_rwse.edge_index,
            batch_rwse.edge_attr,
            batch_rwse.batch,
            batch_rwse.random_walk_pe,
        )
        gated_loss = torch.nn.functional.l1_loss(gated_output, batch_rwse.y)
        edge_state_output = edge_state_rwse(
            batch_rwse.x,
            batch_rwse.edge_index,
            batch_rwse.edge_attr,
            batch_rwse.batch,
            batch_rwse.random_walk_pe,
        )
        edge_state_loss = torch.nn.functional.l1_loss(
            edge_state_output,
            batch_rwse.y,
        )
    (loss + normalized_gap_loss + gated_loss + edge_state_loss).backward()
    gradients = [parameter.grad for parameter in rwse.parameters() if parameter.grad is not None]
    if not all(
        torch.isfinite(output).all()
        for output in (
            base_output,
            rwse_output,
            normalized_gap_output,
            gated_output,
            edge_state_output,
        )
    ):
        raise RuntimeError("Preflight produced non-finite predictions")
    if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
        raise RuntimeError("Preflight produced missing or non-finite gradients")

    report = {
        "status": "accepted",
        "rows": len(base),
        "split_counts": split_counts,
        "base_graph_sha256": sha256(args.base_graph),
        "rwse_graph_sha256": sha256(args.rwse_graph),
        "split_sha256": sha256(args.split_csv),
        "rwse_dim": 16,
        "model_config": configuration,
        "shared_initialization_exact": True,
        "normalized_shared_initialization_exact": True,
        "normalized_gap_output_shape": list(normalized_gap_output.shape),
        "gated_output_shape": list(gated_output.shape),
        "gated_parameters": int(sum(p.numel() for p in gated_rwse.parameters())),
        "edge_state_output_shape": list(edge_state_output.shape),
        "edge_state_parameters": int(
            sum(p.numel() for p in edge_state_rwse.parameters())
        ),
        "rwse_alpha_init": float(normalized_gap_rwse.rwse_alpha.detach().cpu()),
        "forward_backward_finite": True,
        "torch_version": torch.__version__,
        "device": torch.cuda.get_device_name(0),
    }
    _atomic_json(report, args.output)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
