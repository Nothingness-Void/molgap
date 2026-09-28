"""Synthetic CPU contract checks; no fixed-role labels or GPU training."""
from __future__ import annotations

import pytest


def _graph(links, node_count):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    from torch_geometric.data import Data
    from molgap.pcqm_metagin_sidecar import derive_hops

    directed = [edge for source, target in links for edge in ((source, target), (target, source))]
    edge_index = torch.tensor(directed, dtype=torch.long).t().contiguous()
    if not directed:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    return Data(
        x=torch.zeros((node_count, 9), dtype=torch.long),
        edge_index=edge_index,
        edge_attr=torch.zeros((len(directed), 3), dtype=torch.long),
        random_walk_pe=torch.zeros((node_count, 16), dtype=torch.float),
        y=torch.zeros(1),
        source_idx=torch.tensor([0]),
        **derive_hops(edge_index, node_count),
    )


def test_simple_paths_and_cycles_are_counted_without_geometry():
    path = _graph([(0, 1), (1, 2), (2, 3)], 4)
    second = dict(zip(map(tuple, path.hop2_edge_index.t().tolist()), path.hop2_count.tolist()))
    third = dict(zip(map(tuple, path.hop3_edge_index.t().tolist()), path.hop3_count.tolist()))
    assert second == {(0, 2): 1, (1, 3): 1, (2, 0): 1, (3, 1): 1}
    assert third == {(0, 3): 1, (3, 0): 1}
    cycle = _graph([(0, 1), (1, 2), (2, 3), (3, 0)], 4)
    counts = dict(zip(map(tuple, cycle.hop2_edge_index.t().tolist()), cycle.hop2_count.tolist()))
    assert counts[(0, 2)] == counts[(2, 0)] == 2
    assert "pos" not in cycle.keys()


def test_isolated_nodes_and_invalid_bonds_fail_closed():
    torch = pytest.importorskip("torch")
    from molgap.pcqm_metagin_sidecar import derive_hops

    isolated = _graph([], 1)
    assert isolated.hop2_edge_index.shape == (2, 0)
    assert isolated.hop3_edge_index.shape == (2, 0)
    with pytest.raises(ValueError, match="bidirectional"):
        derive_hops(torch.tensor([[0], [1]]), 2)


def test_metagin_batching_backward_and_node_permutation():
    torch = pytest.importorskip("torch")
    from torch_geometric.data import Batch
    from molgap.pcqm_metagin import MetaGIN2D, model_parameters
    from molgap.pcqm_metagin_sidecar import derive_hops

    first = _graph([(0, 1), (1, 2), (2, 3)], 4)
    second = _graph([(0, 1), (1, 2)], 3)
    batch = Batch.from_data_list([first, second])
    assert int(batch.hop2_edge_index.max()) >= first.num_nodes
    torch.manual_seed(42)
    model = MetaGIN2D(width=64).eval()
    assert model_parameters() == 5_268_481
    predicted = model(batch)
    assert predicted.shape == (2,) and bool(torch.isfinite(predicted).all())
    predicted.sum().backward()
    assert model.blocks[0].hops[2].sender.weight.grad is not None

    order = torch.tensor([2, 0, 3, 1])
    inverse = torch.argsort(order)
    permuted_edges = inverse[first.edge_index]
    permuted = first.clone()
    permuted.x = first.x[order]
    permuted.random_walk_pe = first.random_walk_pe[order]
    permuted.edge_index = permuted_edges
    for key, value in derive_hops(permuted_edges, 4).items():
        setattr(permuted, key, value)
    with torch.no_grad():
        original = model(Batch.from_data_list([first]))
        changed = model(Batch.from_data_list([permuted]))
    torch.testing.assert_close(original, changed, rtol=1e-5, atol=1e-5)
