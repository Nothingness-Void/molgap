from __future__ import annotations

import torch

from molgap.pcqm_gap_architecture import make_pcqm_gap_encoder


COMPARATOR = "ogb_distance_angle_triangle_edge_state_gps9"
CANDIDATE = "ogb_distance_angle_dual_stream_triangle_edge_state_gps9"
EXPECTED_PARAMETERS = 5_083_889


def synthetic_inputs():
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long
    )
    wedges = []
    for first in range(edge_index.shape[1]):
        for second in range(edge_index.shape[1]):
            if (
                edge_index[1, first] == edge_index[0, second]
                and edge_index[0, first] != edge_index[1, second]
            ):
                wedges.append((first, second))
    wedge_edge_ids = torch.tensor(wedges, dtype=torch.long)
    return {
        "x": torch.zeros((4, 9), dtype=torch.long),
        "edge_index": edge_index,
        "edge_attr": torch.zeros((edge_index.shape[1], 3), dtype=torch.long),
        "batch": torch.zeros(4, dtype=torch.long),
        "random_walk_pe": torch.zeros((4, 16)),
        "wedge_edge_ids": wedge_edge_ids,
        "edge_distance": torch.ones((edge_index.shape[1], 1)),
        "wedge_angle_cos": torch.zeros((wedge_edge_ids.shape[0], 1)),
        "geometry_valid": torch.ones((1, 1)),
    }


def test_dual_stream_parameter_budget_and_zero_residuals():
    model = make_pcqm_gap_encoder(CANDIDATE)
    assert sum(parameter.numel() for parameter in model.parameters()) == EXPECTED_PARAMETERS
    assert sum(parameter.numel() for parameter in model.parameters()) <= 5_200_000
    assert tuple(model.bond_stream_blocks.keys()) == ("1", "3", "5", "7")
    zero_names = [
        name
        for name in model.state_dict()
        if name.endswith("attention_output.weight")
        or name.endswith("attention_output.bias")
        or name.endswith("ffn_output.weight")
        or name.endswith("ffn_output.bias")
        or name.endswith("atom_to_bond.value.weight")
        or name.endswith("atom_to_bond.value.bias")
        or name.endswith("bond_to_atom.value.weight")
        or name.endswith("bond_to_atom.value.bias")
    ]
    assert len(zero_names) == 20
    assert all(torch.count_nonzero(model.state_dict()[name]).item() == 0 for name in zero_names)


def test_dual_stream_initial_function_equals_comparator_and_backpropagates():
    torch.manual_seed(42)
    comparator = make_pcqm_gap_encoder(COMPARATOR).eval()
    torch.manual_seed(42)
    candidate = make_pcqm_gap_encoder(CANDIDATE).eval()
    comparator_state = comparator.state_dict()
    candidate_state = candidate.state_dict()
    assert all(
        torch.equal(value, candidate_state[name])
        for name, value in comparator_state.items()
    )

    inputs = synthetic_inputs()
    with torch.no_grad():
        comparator_output = comparator(**inputs)
        candidate_output = candidate(**inputs)
    assert torch.equal(comparator_output, candidate_output)

    candidate.train()
    prediction = candidate(**inputs)
    loss = prediction.abs().mean()
    loss.backward()
    assert torch.isfinite(prediction).all()
    assert torch.isfinite(loss)
    gradients = [
        parameter.grad
        for parameter in candidate.parameters()
        if parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
