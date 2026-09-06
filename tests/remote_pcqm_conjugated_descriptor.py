"""Remote-only DCU preflight for the K3a conjugated descriptor control."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


BASELINE = "ogb_distance_angle_triangle_edge_state_graph_state9"
CANDIDATE = (
    "ogb_distance_angle_triangle_edge_state_graph_state9_conjugated_descriptor"
)
COUNTS = {BASELINE: 3_665_809, CANDIDATE: 3_672_257}
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

    from molgap.pcqm_conjugated_cache import with_conjugated_components
    from molgap.pcqm_conjugated_state import make_conjugated_encoder
    from molgap.pcqm_gap_architecture import make_pcqm_gap_encoder

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
        if nodes > 2:
            x[1, 7] = 1
        pairs = [(i, i + 1) for i in range(nodes - 1)]
        directed = [edge for pair in pairs for edge in (pair, pair[::-1])]
        edge_index = torch.tensor(directed, dtype=torch.long).t().contiguous()
        edge_attr = torch.zeros((edge_index.shape[1], 3), dtype=torch.long)
        for column, dim in enumerate(BOND_DIMS):
            edge_attr[:, column] = torch.arange(edge_attr.shape[0]) % dim
        if edge_attr.shape[0] >= 4:
            edge_attr[:4, 2] = 1
        wedges = []
        edges = edge_index.t().tolist()
        for first, (source, center) in enumerate(edges):
            for second, (other_source, target) in enumerate(edges):
                if center == other_source and target != source:
                    wedges.append((first, second))
        wedge_ids = torch.tensor(wedges, dtype=torch.long).reshape(-1, 2)
        data = SyntheticData(
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
        return with_conjugated_components(data)

    def base_args(batch):
        return (
            batch.x, batch.edge_index, batch.edge_attr, batch.batch,
            batch.random_walk_pe, batch.wedge_edge_ids,
            batch.edge_distance, batch.wedge_angle_cos, batch.geometry_valid,
        )

    def candidate_args(batch):
        return base_args(batch) + (
            batch.conjugated_component_id,
            batch.conjugated_component_count,
            batch.conjugated_features,
        )

    batch = Batch.from_data_list([graph(4)]).to(device)
    torch.manual_seed(42)
    baseline = make_pcqm_gap_encoder(BASELINE).to(device)
    torch.manual_seed(42)
    candidate = make_conjugated_encoder(CANDIDATE).to(device)
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
        baseline_prediction = baseline(*base_args(batch))
        candidate_prediction = candidate(*candidate_args(batch))
    if not torch.equal(baseline_prediction, candidate_prediction):
        raise AssertionError("zero descriptor return changed initial prediction")
    if torch.count_nonzero(candidate.descriptor_to_atom.weight):
        raise AssertionError("descriptor return is not zero initialized")

    candidate.train()
    candidate.zero_grad(set_to_none=True)
    candidate(*candidate_args(batch)).square().mean().backward()
    gradient = candidate.descriptor_to_atom.weight.grad
    if gradient is None or not torch.isfinite(gradient).all() or not torch.count_nonzero(gradient):
        raise AssertionError("descriptor return lacks a finite nonzero gradient")

    candidate.eval()
    with torch.no_grad():
        candidate.descriptor_to_atom.weight.normal_(0.0, 0.02)
        first = graph(4)
        second = graph(3, 0.2)
        first_prediction = candidate(*candidate_args(Batch.from_data_list([first]).to(device)))
        second_prediction = candidate(*candidate_args(Batch.from_data_list([second]).to(device)))
        combined = candidate(*candidate_args(Batch.from_data_list([first, second]).to(device)))
    torch.testing.assert_close(combined[0], first_prediction[0], rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(combined[1], second_prediction[0], rtol=2e-5, atol=2e-5)
    return {
        "accepted": True,
        "device": "cuda:0",
        "device_name": torch.cuda.get_device_name(0),
        "device_count": 1,
        "parameter_counts": actual,
        "checks": {
            "shared_parameter_equality_seed42": True,
            "initial_prediction_identity": True,
            "descriptor_return_gradient": True,
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
