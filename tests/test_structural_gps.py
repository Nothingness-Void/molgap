from __future__ import annotations

import torch
from torch_geometric.data import Batch, Data

from molgap.gps import (
    EdgeStateStructuralGPSWrapper,
    GatedStructuralGPSWrapper,
    GPSWrapper,
    NormalizedStructuralGPSWrapper,
    OrbitalCenterHead,
    StructuralGPSWrapper,
    reconstruct_frontier_orbitals,
)
from molgap.structural_encoding import add_random_walk_pe


def _chain() -> Data:
    return Data(
        x=torch.randn(3, 9),
        edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
        edge_attr=torch.randn(4, 4),
        source_idx=torch.tensor([7]),
        y=torch.randn(1, 3),
    )


def test_random_walk_pe_has_expected_return_probabilities() -> None:
    graph = add_random_walk_pe(_chain(), walk_length=4)
    assert graph.random_walk_pe.shape == (3, 4)
    assert torch.isfinite(graph.random_walk_pe).all()
    torch.testing.assert_close(graph.random_walk_pe[:, 0], torch.zeros(3))
    torch.testing.assert_close(
        graph.random_walk_pe[:, 1],
        torch.tensor([0.5, 1.0, 0.5]),
    )


def test_structural_gps_preserves_common_initialization_and_runs() -> None:
    configuration = {
        "hidden_channels": 16,
        "num_layers": 2,
        "num_heads": 4,
        "dropout": 0.0,
    }
    torch.manual_seed(42)
    base = GPSWrapper(**configuration)
    torch.manual_seed(42)
    structural = StructuralGPSWrapper(**configuration, rwse_dim=4)
    structural_state = structural.state_dict()
    for name, value in base.state_dict().items():
        torch.testing.assert_close(value, structural_state[name])

    batch = Batch.from_data_list([add_random_walk_pe(_chain(), walk_length=4)])
    output = structural(
        batch.x,
        batch.edge_index,
        batch.edge_attr,
        batch.batch,
        batch.random_walk_pe,
    )
    assert output.shape == (1, 3)
    assert torch.isfinite(output).all()


def test_normalized_structural_gps_uses_bounded_rwse_gate_and_gap_output() -> None:
    configuration = {
        "hidden_channels": 16,
        "num_layers": 2,
        "num_heads": 4,
        "dropout": 0.0,
        "n_targets": 1,
    }
    torch.manual_seed(42)
    legacy = StructuralGPSWrapper(**configuration, rwse_dim=4)
    torch.manual_seed(42)
    normalized = NormalizedStructuralGPSWrapper(
        **configuration,
        rwse_dim=4,
        rwse_alpha_init=0.25,
    )
    normalized_state = normalized.state_dict()
    for name, value in legacy.state_dict().items():
        torch.testing.assert_close(value, normalized_state[name])

    batch = Batch.from_data_list(
        [
            add_random_walk_pe(_chain(), walk_length=4),
            add_random_walk_pe(_chain(), walk_length=4),
        ]
    )
    output = normalized(
        batch.x,
        batch.edge_index,
        batch.edge_attr,
        batch.batch,
        batch.random_walk_pe,
    )
    assert output.shape == (2, 1)
    assert torch.isfinite(output).all()
    torch.testing.assert_close(normalized.rwse_alpha, torch.tensor(0.25))
    output.abs().mean().backward()
    assert normalized.rwse_alpha_logit.grad is not None
    assert torch.isfinite(normalized.rwse_alpha_logit.grad)


def test_gated_structural_gps_runs_edge_aware_forward_and_backward() -> None:
    model = GatedStructuralGPSWrapper(
        hidden_channels=16,
        num_layers=2,
        num_heads=4,
        dropout=0.0,
        rwse_dim=4,
    )
    batch = Batch.from_data_list(
        [
            add_random_walk_pe(_chain(), walk_length=4),
            add_random_walk_pe(_chain(), walk_length=4),
        ]
    )
    output = model(
        batch.x,
        batch.edge_index,
        batch.edge_attr,
        batch.batch,
        batch.random_walk_pe,
    )
    assert output.shape == (2, 3)
    assert torch.isfinite(output).all()
    output.abs().mean().backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_edge_state_structural_gps_updates_compact_edge_state() -> None:
    model = EdgeStateStructuralGPSWrapper(
        hidden_channels=16,
        num_layers=2,
        num_heads=4,
        dropout=0.0,
        rwse_dim=4,
        edge_state_channels=8,
    )
    batch = Batch.from_data_list(
        [
            add_random_walk_pe(_chain(), walk_length=4),
            add_random_walk_pe(_chain(), walk_length=4),
        ]
    )
    output = model(
        batch.x,
        batch.edge_index,
        batch.edge_attr,
        batch.batch,
        batch.random_walk_pe,
    )
    assert output.shape == (2, 3)
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    assert model.edge_updates[0].source.weight.grad is not None
    assert torch.isfinite(model.edge_updates[0].source.weight.grad).all()


def test_orbital_center_reconstruction_is_exact() -> None:
    embedding = torch.randn(3, 8)
    head = OrbitalCenterHead(8, hidden_channels=4, dropout=0.0)
    center = head(embedding)
    gap = torch.tensor([[1.0], [2.0], [3.0]])
    output = reconstruct_frontier_orbitals(gap, center)
    assert output.shape == (3, 3)
    torch.testing.assert_close(output[:, 1] - output[:, 0], output[:, 2])
