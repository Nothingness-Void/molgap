"""Remote-only DCU preflight for the K2 projected-moment readout."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


BASELINE = "ogb_distance_angle_triangle_edge_state_graph_state9"
CANDIDATE = (
    "ogb_distance_angle_triangle_edge_state_graph_state9_moment_readout32"
)
COUNTS = {BASELINE: 3_665_809, CANDIDATE: 3_684_753}
ATOM_DIMS = (119, 5, 12, 12, 10, 6, 6, 2, 2)
BOND_DIMS = (5, 6, 2)


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def run() -> dict:
    import torch
    from torch_geometric.data import Batch, Data

    from molgap.pcqm_gap_architecture import make_pcqm_gap_encoder
    from molgap.pcqm_moment_readout import make_moment_readout_encoder

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one visible CUDA/DCU device is required")
    device = torch.device("cuda:0")

    class SyntheticData(Data):
        def __inc__(self, key, value, *args, **kwargs):
            if key == "wedge_edge_ids":
                return int(self.edge_index.shape[1])
            return super().__inc__(key, value, *args, **kwargs)

    def graph(nodes: int, offset: float = 0.0):
        x = torch.stack(
            [torch.arange(nodes).remainder(dim) for dim in ATOM_DIMS], dim=1
        )
        pairs = [(i, i + 1) for i in range(nodes - 1)]
        directed = [edge for pair in pairs for edge in (pair, pair[::-1])]
        edge_index = torch.tensor(directed, dtype=torch.long).t().contiguous()
        edge_attr = torch.zeros((edge_index.shape[1], 3), dtype=torch.long)
        for column, dim in enumerate(BOND_DIMS):
            edge_attr[:, column] = torch.arange(edge_attr.shape[0]) % dim
        wedges = []
        edges = edge_index.t().tolist()
        for first, (source, center) in enumerate(edges):
            for second, (other_source, target) in enumerate(edges):
                if center == other_source and target != source:
                    wedges.append((first, second))
        wedge_ids = torch.tensor(wedges, dtype=torch.long).reshape(-1, 2)
        return SyntheticData(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            random_walk_pe=torch.arange(nodes * 16, dtype=torch.float32).view(
                nodes, 16
            ) / 100.0,
            wedge_edge_ids=wedge_ids,
            edge_distance=torch.ones((edge_index.shape[1], 1)),
            wedge_angle_cos=torch.zeros((wedge_ids.shape[0], 1)),
            geometry_valid=torch.tensor([1.0]),
            y=torch.tensor([0.5 + offset]),
            num_nodes=nodes,
        )

    def args(batch):
        return (
            batch.x, batch.edge_index, batch.edge_attr, batch.batch,
            batch.random_walk_pe, batch.wedge_edge_ids,
            batch.edge_distance, batch.wedge_angle_cos, batch.geometry_valid,
        )

    batch = Batch.from_data_list([graph(4)]).to(device)
    torch.manual_seed(42)
    baseline = make_pcqm_gap_encoder(BASELINE).to(device)
    torch.manual_seed(42)
    candidate = make_moment_readout_encoder().to(device)
    actual = {
        BASELINE: sum(p.numel() for p in baseline.parameters()),
        CANDIDATE: sum(p.numel() for p in candidate.parameters()),
    }
    if actual != COUNTS:
        raise AssertionError(f"parameter counts changed: {actual}")
    baseline_state = baseline.state_dict()
    candidate_state = candidate.state_dict()
    mismatches = [
        name for name, value in baseline_state.items()
        if name not in candidate_state or not torch.equal(value, candidate_state[name])
    ]
    if mismatches:
        raise AssertionError(f"shared initialization changed: {mismatches}")
    baseline.eval()
    candidate.eval()
    with torch.no_grad():
        base_prediction = baseline(*args(batch))
        candidate_prediction = candidate(*args(batch))
    if not torch.equal(base_prediction, candidate_prediction):
        raise AssertionError("zero-return readout changed initial prediction")
    if int(torch.count_nonzero(candidate.moment_return.weight)) != 0:
        raise AssertionError("moment return is not zero initialized")

    candidate.train()
    candidate.zero_grad(set_to_none=True)
    candidate(*args(batch)).square().mean().backward()
    gradient = candidate.moment_return.weight.grad
    if gradient is None or not torch.isfinite(gradient).all() or not torch.count_nonzero(gradient):
        raise AssertionError("moment return lacks a finite nonzero gradient")

    candidate.eval()
    with torch.no_grad():
        candidate.moment_return.weight.normal_(0.0, 0.02)
        single_a = candidate(*args(Batch.from_data_list([graph(4)]).to(device)))
        single_b = candidate(*args(Batch.from_data_list([graph(3, 0.2)]).to(device)))
        together = candidate(
            *args(Batch.from_data_list([graph(4), graph(3, 0.2)]).to(device))
        )
    torch.testing.assert_close(together[0], single_a[0], rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(together[1], single_b[0], rtol=2e-5, atol=2e-5)
    return {
        "accepted": True,
        "device": "cuda:0",
        "device_name": torch.cuda.get_device_name(0),
        "device_count": 1,
        "parameter_counts": actual,
        "checks": {
            "shared_parameter_equality_seed42": True,
            "initial_prediction_identity": True,
            "moment_return_gradient": True,
            "batch_separation": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run()
    except Exception as error:
        result = {
            "accepted": False,
            "error": f"{type(error).__name__}: {error}",
            "parameter_counts": COUNTS,
        }
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2), flush=True)
    if result.get("accepted") is not True:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
