"""CPU-only nested-function and chemistry-sidecar tests."""
import torch
from torch_geometric.data import Batch, Data

from molgap.k1_conjugated_hyperedge import MODES, check_mechanism
from molgap.pcqm_conjugated_cache import with_conjugated_components
from molgap.pcqm_k1_variants import make_encoder


def _batch():
    edge_attr = torch.zeros((4, 3), dtype=torch.long)
    edge_attr[:, 2] = 1
    graph = Data(
        x=torch.zeros((3, 9), dtype=torch.long),
        edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
        edge_attr=edge_attr,
        random_walk_pe=torch.zeros((3, 16)),
    )
    graph = with_conjugated_components(graph)
    return Batch.from_data_list([graph, graph])


def test_both_arms_are_bitwise_k1_at_initialization():
    batch = _batch()
    torch.manual_seed(42)
    baseline = make_encoder("neural_atom_k1_v4").eval()
    with torch.no_grad():
        expected = baseline(
            batch.x, batch.edge_index, batch.edge_attr,
            batch.batch, batch.random_walk_pe,
        )
    for mode in MODES:
        torch.manual_seed(42)
        model = make_encoder(mode).eval()
        with torch.no_grad():
            actual = model(
                batch.x, batch.edge_index, batch.edge_attr,
                batch.batch, batch.random_walk_pe,
                batch.conjugated_component_id,
                batch.conjugated_component_count,
            )
        assert torch.equal(expected, actual)
        assert check_mechanism(model, batch)["zero_return_exact"]


def test_component_return_is_trainable_without_extra_labels():
    batch = _batch()
    for mode in MODES:
        torch.manual_seed(42)
        model = make_encoder(mode).train()
        model(
            batch.x, batch.edge_index, batch.edge_attr,
            batch.batch, batch.random_walk_pe,
            batch.conjugated_component_id,
            batch.conjugated_component_count,
        ).square().mean().backward()
        gradient = model.return_projection.weight.grad
        assert gradient is not None and torch.isfinite(gradient).all()
        assert gradient.abs().sum() > 0
