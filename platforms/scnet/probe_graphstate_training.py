"""Run a bounded synthetic GraphState training step on one SCNet accelerator."""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import traceback
from pathlib import Path


CANDIDATE = "ogb_distance_angle_triangle_edge_state_graph_state9"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def categorical_features(torch, rows: int, dimensions: list[int], seed: int):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.stack(
        [torch.randint(0, width, (rows,), generator=generator) for width in dimensions],
        dim=1,
    )


def make_graph(torch, Data, atom_dims, bond_dims, nodes: int, seed: int):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    undirected = [(index, index + 1) for index in range(nodes - 1)]
    undirected.append((nodes - 1, 0))
    directed = [pair for edge in undirected for pair in (edge, edge[::-1])]
    edge_index = torch.tensor(directed, dtype=torch.long).t().contiguous()
    edge_attr = categorical_features(torch, len(directed), bond_dims, seed + 1)
    pos = torch.randn(nodes, 3, generator=generator)
    source, target = edge_index
    edge_distance = (pos[source] - pos[target]).norm(dim=-1, keepdim=True)
    return Data(
        x=categorical_features(torch, nodes, atom_dims, seed + 2),
        edge_index=edge_index,
        edge_attr=edge_attr,
        random_walk_pe=torch.randn(nodes, 16, generator=generator),
        wedge_edge_ids=torch.empty((0, 2), dtype=torch.long),
        edge_distance=edge_distance,
        wedge_angle_cos=torch.empty((0, 1), dtype=torch.float32),
        geometry_valid=torch.ones(1, dtype=torch.float32),
        y=torch.randn(1, generator=generator),
    )


def run() -> dict[str, object]:
    import torch
    import torch_geometric
    from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims
    from torch_geometric.data import Batch, Data

    from molgap.pcqm_gap_architecture import make_pcqm_gap_encoder

    if not torch.cuda.is_available():
        raise RuntimeError("Platform PyTorch does not expose the allocated DCU")
    device = torch.device("cuda:0")
    graphs = [
        make_graph(
            torch,
            Data,
            get_atom_feature_dims(),
            get_bond_feature_dims(),
            nodes,
            9000 + nodes,
        )
        for nodes in (6, 8, 10, 12)
    ]
    batch = Batch.from_data_list(graphs).to(device)
    model = make_pcqm_gap_encoder(CANDIDATE).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4)
    losses = []
    started = time.perf_counter()
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.batch,
            batch.random_walk_pe,
            batch.wedge_edge_ids,
            batch.edge_distance,
            batch.wedge_angle_cos,
            batch.geometry_valid,
        ).reshape(-1)
        loss = torch.nn.functional.l1_loss(prediction, batch.y.reshape(-1))
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    torch.cuda.synchronize()
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_geometric": torch_geometric.__version__,
        "cuda_build": torch.version.cuda,
        "hip_build": getattr(torch.version, "hip", None),
        "accelerator_ready": True,
        "device_name": torch.cuda.get_device_name(0),
        "device_count": torch.cuda.device_count(),
        "candidate": CANDIDATE,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "steps": len(losses),
        "losses": losses,
        "finite_losses": all(value == value and abs(value) != float("inf") for value in losses),
        "elapsed_s": time.perf_counter() - started,
        "peak_memory_bytes": torch.cuda.max_memory_allocated(device),
    }


def main() -> None:
    args = parse_args()
    report: dict[str, object] = {
        "format": "molgap-scnet-graphstate-training-probe-v1",
        "model_execution": "synthetic_only",
        "official_data_read": False,
    }
    exit_code = 0
    try:
        report.update(run())
        report["passed"] = bool(report["finite_losses"])
    except Exception as exc:
        report.update(
            passed=False,
            error_type=type(exc).__name__,
            error=str(exc),
            traceback=traceback.format_exc(),
        )
        exit_code = 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
