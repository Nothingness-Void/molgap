"""Remote-only CUDA/DCU preflight for the bounded PCQM vector-state candidate."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


BASELINE_ID = "ogb_distance_angle_triangle_edge_state_graph_state9"
CANDIDATE_ID = "ogb_distance_angle_vector_state_triangle_edge_state_graph_state9"
BASELINE_PARAMETER_COUNT = 3_665_809
CANDIDATE_PARAMETER_COUNT = 3_696_209
DEVICE_NAME = "cuda:0"

ATOM_DIMS = (119, 5, 12, 12, 10, 6, 6, 2, 2)
BOND_DIMS = (5, 6, 2)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def run(output: Path) -> dict:
    # Imports and all model execution intentionally live inside the remote run.
    import torch
    from torch_geometric.data import Batch, Data

    from molgap.pcqm_gap_architecture import make_pcqm_gap_encoder
    from molgap.pcqm_vector_state import make_vector_state_encoder

    expected_counts = {
        BASELINE_ID: BASELINE_PARAMETER_COUNT,
        CANDIDATE_ID: CANDIDATE_PARAMETER_COUNT,
    }
    if not torch.cuda.is_available():
        raise RuntimeError("a CUDA/DCU device is required")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"exactly one visible accelerator is required, got {torch.cuda.device_count()}"
        )
    torch.cuda.set_device(0)
    device = torch.device(DEVICE_NAME)

    class SyntheticData(Data):
        def __inc__(self, key, value, *args, **kwargs):
            if key == "wedge_edge_ids":
                return int(self.edge_index.shape[1])
            return super().__inc__(key, value, *args, **kwargs)

    def wedge_ids(edge_index: torch.Tensor) -> torch.Tensor:
        edges = edge_index.t().tolist()
        pairs = []
        for first, (source, center) in enumerate(edges):
            for second, (second_source, target) in enumerate(edges):
                if center == second_source and target != source:
                    pairs.append((first, second))
        return torch.tensor(pairs, dtype=torch.long).reshape(-1, 2)

    def make_graph(
        node_count: int = 4,
        phase: float = 0.0,
        valid: bool = True,
        edgeless: bool = False,
    ) -> SyntheticData:
        x = torch.stack(
            [torch.arange(node_count, dtype=torch.long).remainder(dim) for dim in ATOM_DIMS],
            dim=1,
        )
        positions = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.15, 0.15, 0.05],
                [2.25, -0.10, 0.35],
                [3.05, 0.25, 0.55],
            ],
            dtype=torch.float32,
        )[:node_count]
        positions[:, 1] += float(phase)

        if edgeless:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            undirected = [(i, i + 1) for i in range(node_count - 1)]
            directed = [edge for pair in undirected for edge in (pair, pair[::-1])]
            edge_index = torch.tensor(directed, dtype=torch.long).t().contiguous()
        edge_attr = torch.zeros((edge_index.shape[1], 3), dtype=torch.long)
        for column, dim in enumerate(BOND_DIMS):
            if edge_attr.shape[0]:
                edge_attr[:, column] = torch.arange(edge_attr.shape[0]) % dim
        if edge_index.shape[1]:
            source, destination = edge_index
            edge_distance = (
                (positions[destination] - positions[source]).square().sum(dim=1).sqrt()
            ).view(-1, 1)
        else:
            edge_distance = torch.empty((0, 1), dtype=torch.float32)
        ids = wedge_ids(edge_index)
        return SyntheticData(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            batch=None,
            pos=positions,
            random_walk_pe=torch.arange(node_count * 16, dtype=torch.float32).view(
                node_count, 16
            )
            / 100.0,
            wedge_edge_ids=ids,
            edge_distance=edge_distance,
            wedge_angle_cos=torch.zeros((ids.shape[0], 1), dtype=torch.float32),
            geometry_valid=torch.tensor(
                [1.0 if valid else 0.0], dtype=torch.float32
            ),
            y=torch.tensor([0.5 + phase], dtype=torch.float32),
            num_nodes=node_count,
        )

    def make_batch(graphs):
        return Batch.from_data_list(graphs).to(device)

    def baseline_args(batch):
        return (
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.batch,
            batch.random_walk_pe,
            batch.wedge_edge_ids,
            batch.edge_distance,
            batch.wedge_angle_cos,
            batch.geometry_valid,
        )

    def candidate_args(batch):
        return baseline_args(batch) + (batch.pos,)

    def parameter_count(model):
        return int(sum(parameter.numel() for parameter in model.parameters()))

    def permute_graph(graph, permutation):
        inverse = torch.empty_like(permutation)
        inverse[permutation] = torch.arange(permutation.numel())
        edge_index = inverse[graph.edge_index]
        return SyntheticData(
            x=graph.x[permutation],
            edge_index=edge_index,
            edge_attr=graph.edge_attr.clone(),
            pos=graph.pos[permutation],
            random_walk_pe=graph.random_walk_pe[permutation],
            wedge_edge_ids=graph.wedge_edge_ids.clone(),
            edge_distance=graph.edge_distance.clone(),
            wedge_angle_cos=graph.wedge_angle_cos.clone(),
            geometry_valid=graph.geometry_valid.clone(),
            y=graph.y.clone(),
            num_nodes=graph.num_nodes,
        )

    def transform_graph(graph, matrix, translation):
        position = graph.pos @ matrix.t() + translation
        if graph.edge_index.shape[1]:
            source, destination = graph.edge_index
            distance = (
                (position[destination] - position[source]).square().sum(dim=1).sqrt()
            ).view(-1, 1)
        else:
            distance = graph.edge_distance.clone()
        return SyntheticData(
            x=graph.x.clone(),
            edge_index=graph.edge_index.clone(),
            edge_attr=graph.edge_attr.clone(),
            pos=position,
            random_walk_pe=graph.random_walk_pe.clone(),
            wedge_edge_ids=graph.wedge_edge_ids.clone(),
            edge_distance=distance,
            wedge_angle_cos=graph.wedge_angle_cos.clone(),
            geometry_valid=graph.geometry_valid.clone(),
            y=graph.y.clone(),
            num_nodes=graph.num_nodes,
        )

    checks = {}

    def check(name, function):
        try:
            checks[name] = {"accepted": True, "details": function()}
        except Exception as error:  # keep every mechanical result in JSON
            checks[name] = {
                "accepted": False,
                "error": f"{type(error).__name__}: {error}",
            }

    graph = make_graph()
    batch = make_batch([graph])

    torch.manual_seed(42)
    baseline = make_pcqm_gap_encoder(BASELINE_ID).to(device)
    torch.manual_seed(42)
    candidate = make_vector_state_encoder().to(device)
    baseline.eval()
    candidate.eval()

    actual_counts = {
        BASELINE_ID: parameter_count(baseline),
        CANDIDATE_ID: parameter_count(candidate),
    }

    def count_check():
        if actual_counts != expected_counts:
            raise AssertionError(f"expected {expected_counts}, got {actual_counts}")
        if actual_counts[CANDIDATE_ID] > 4_000_000:
            raise AssertionError("candidate exceeds 4M parameter budget")
        return actual_counts

    check("exact_parameter_count", count_check)

    def shared_check():
        baseline_state = baseline.state_dict()
        candidate_state = candidate.state_dict()
        mismatches = [
            name
            for name in baseline_state
            if name not in candidate_state
            or not torch.equal(baseline_state[name], candidate_state[name])
        ]
        if mismatches:
            raise AssertionError(f"shared initialization mismatches: {mismatches}")
        return {"mismatches": [], "shared_keys": len(baseline_state)}

    check("shared_parameter_equality_seed42", shared_check)

    with torch.no_grad():
        baseline_prediction = baseline(*baseline_args(batch))
        candidate_prediction = candidate(*candidate_args(batch))

    def identity_check():
        if not torch.equal(baseline_prediction, candidate_prediction):
            raise AssertionError("zero-return candidate changed initial eval output")
        if int(torch.count_nonzero(candidate.scalar_return.weight)) != 0:
            raise AssertionError("scalar_return is not zero initialized")
        return {"exact_tensor_equal": True, "scalar_return_nonzero": 0}

    check("initial_eval_prediction_identity", identity_check)

    def gradient_check():
        torch.manual_seed(42)
        gradient_model = make_vector_state_encoder().to(device)
        gradient_model.train()
        gradient_model.zero_grad(set_to_none=True)
        prediction = gradient_model(*candidate_args(batch))
        loss = prediction.square().mean()
        loss.backward()
        gradients = [
            parameter.grad
            for parameter in gradient_model.parameters()
            if parameter.grad is not None
        ]
        scalar_gradient = gradient_model.scalar_return.weight.grad
        finite = bool(gradients) and all(torch.isfinite(value).all() for value in gradients)
        nonzero = scalar_gradient is not None and int(torch.count_nonzero(scalar_gradient)) > 0
        if not finite or not nonzero:
            raise AssertionError(
                f"finite_gradients={finite}, scalar_return_gradient_nonzero={nonzero}"
            )
        del gradient_model
        return {"finite_gradients": True, "scalar_return_gradient_nonzero": True}

    check("finite_gradients_and_scalar_return_gradient", gradient_check)

    with torch.no_grad():
        torch.manual_seed(1234)
        candidate.scalar_return.weight.normal_(mean=0.0, std=0.05)
        randomized_prediction = candidate(*candidate_args(batch))

    def randomized_path_check():
        if int(torch.count_nonzero(candidate.scalar_return.weight)) == 0:
            raise AssertionError("scalar-return weights were not randomized")
        if torch.equal(randomized_prediction, candidate_prediction):
            raise AssertionError("invariance checks would be vacuous")
        return {"scalar_return_nonzero": int(torch.count_nonzero(candidate.scalar_return.weight))}

    check("randomized_scalar_return_is_active", randomized_path_check)

    candidate.eval()

    def symmetry_check():
        rotation = torch.tensor(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        )
        reflection = torch.diag(torch.tensor([-1.0, 1.0, 1.0]))
        translation = torch.tensor([2.0, -3.0, 1.5])
        outputs = {}
        for name, matrix, shift in (
            ("rotation", rotation, torch.zeros(3)),
            ("reflection", reflection, torch.zeros(3)),
            ("translation", torch.eye(3), translation),
        ):
            transformed = make_batch([transform_graph(graph, matrix, shift)])
            outputs[name] = candidate(*candidate_args(transformed))
            torch.testing.assert_close(
                randomized_prediction,
                outputs[name],
                rtol=2.0e-5,
                atol=2.0e-5,
            )
        return {name: True for name in outputs}

    check("rotation_reflection_translation_invariance", symmetry_check)

    def permutation_check():
        permutation = torch.tensor([2, 0, 3, 1], dtype=torch.long)
        permuted = make_batch([permute_graph(graph, permutation)])
        result = candidate(*candidate_args(permuted))
        torch.testing.assert_close(randomized_prediction, result, rtol=2.0e-5, atol=2.0e-5)
        return {"node_permutation": True}

    check("node_permutation_invariance", permutation_check)

    def invalid_and_edgeless_check():
        invalid = make_graph(valid=False)
        invalid.pos[0, 0] = float("nan")
        invalid.edge_distance[:] = float("nan")
        invalid.wedge_angle_cos[:] = float("nan")
        invalid_result = candidate(*candidate_args(make_batch([invalid])))
        edgeless = candidate(*candidate_args(make_batch([make_graph(3, edgeless=True)])))
        if not bool(torch.isfinite(invalid_result).all()):
            raise AssertionError("invalid geometry produced non-finite output")
        if not bool(torch.isfinite(edgeless).all()):
            raise AssertionError("edgeless graph produced non-finite output")
        return {"invalid_geometry_finite": True, "edgeless_finite": True}

    check("invalid_geometry_and_edgeless_graph", invalid_and_edgeless_check)

    def batch_separation_check():
        second = make_graph(3, phase=0.7)
        batched = candidate(*candidate_args(make_batch([graph, second])))
        first_single = candidate(*candidate_args(make_batch([graph])))
        second_single = candidate(*candidate_args(make_batch([second])))
        torch.testing.assert_close(batched[0], first_single[0], rtol=2.0e-5, atol=2.0e-5)
        torch.testing.assert_close(batched[1], second_single[0], rtol=2.0e-5, atol=2.0e-5)
        return {"independent_graph_outputs": True}

    check("batch_separation", batch_separation_check)

    return {
        "accepted": bool(checks) and all(item["accepted"] for item in checks.values()),
        "device": DEVICE_NAME,
        "device_name": torch.cuda.get_device_name(0),
        "device_count": torch.cuda.device_count(),
        "parameter_counts": expected_counts,
        "actual_parameter_counts": actual_counts,
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run(args.output)
    except Exception as error:
        result = {
            "accepted": False,
            "device": DEVICE_NAME,
            "parameter_counts": {
                BASELINE_ID: BASELINE_PARAMETER_COUNT,
                CANDIDATE_ID: CANDIDATE_PARAMETER_COUNT,
            },
            "checks": {
                "runtime": {
                    "accepted": False,
                    "error": f"{type(error).__name__}: {error}",
                }
            },
        }
    _write_json(args.output, result)
    print(json.dumps(result, indent=2), flush=True)
    if result.get("accepted") is not True:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
