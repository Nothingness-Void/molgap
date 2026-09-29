"""Small topology-only and batching probes; no training dataset is opened."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
from torch_geometric.data import Batch, Data

from molgap.k1_motif_hierarchy import MODE, make_encoder
from molgap.pcqm_motif_partition import derive_motif_partition


def _graph(atoms):
    x = torch.zeros((len(atoms), 9), dtype=torch.long)
    x[:, 0] = torch.tensor(atoms)
    pairs = [(i, i + 1) for i in range(len(atoms) - 1)]
    edges = [edge for i, j in pairs for edge in ((i, j), (j, i))]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_attr = torch.zeros((len(edges), 3), dtype=torch.long)
    graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                 random_walk_pe=torch.zeros((len(atoms), 16)))
    row = derive_motif_partition(graph)
    graph.motif_membership = row["motif_index"]
    graph.motif_count = torch.tensor([row["motif_count"]])
    graph.motif_source = row["motif_edge_index"][0]
    graph.motif_target = row["motif_edge_index"][1]
    graph.motif_edge_count = torch.tensor([row["motif_edge_index"].shape[1]])
    graph.motif_bond_type = row["motif_bond_type"]
    graph.motif_bridge_rule_mask = row["motif_bridge_rule_mask"]
    return graph


def test_two_graph_motif_batch_preserves_local_ids_and_zero_start():
    batch = Batch.from_data_list([_graph([5, 7, 5]), _graph([5, 7])])
    assert batch.motif_count.tolist() == [3, 2]
    assert batch.motif_membership.tolist() == [0, 1, 2, 0, 1]
    assert batch.motif_edge_count.tolist() == [4, 2]
    model = make_encoder(MODE)
    assert sum(p.numel() for p in model.parameters()) == 3_693_505
    hidden = torch.randn((5, 192))
    returned = model.exchange(
        hidden, batch.batch, batch.motif_membership, batch.motif_count,
        batch.motif_source, batch.motif_target, batch.motif_edge_count,
        batch.motif_bond_type, batch.motif_bridge_rule_mask,
    )
    assert torch.equal(returned, hidden)
    returned.sum().backward()
    assert model.return_projection.weight.grad is not None
    assert torch.isfinite(model.return_projection.weight.grad).all()


def test_motif_edge_cannot_cross_graph_boundary():
    batch = Batch.from_data_list([_graph([5, 7, 5]), _graph([5, 7])])
    batch.motif_target[-1] = 2  # Second graph has exactly two motifs.
    model = make_encoder(MODE)
    with pytest.raises(ValueError, match="leaves its molecule"):
        model.exchange(
            torch.randn((5, 192)), batch.batch, batch.motif_membership,
            batch.motif_count, batch.motif_source, batch.motif_target,
            batch.motif_edge_count, batch.motif_bond_type,
            batch.motif_bridge_rule_mask,
        )


def test_full_candidate_initially_matches_k1_prediction():
    from molgap.qm9_neural_atom import make_encoder as make_k1
    from molgap.pcqm_k1_variants_runner import _forward

    batch = Batch.from_data_list([_graph([5, 7, 5]), _graph([5, 7])])
    torch.manual_seed(42)
    reference = make_k1("neural_atom_k1").eval()
    torch.manual_seed(42)
    candidate = make_encoder(MODE).eval()
    with torch.no_grad():
        assert torch.equal(_forward(reference, batch), _forward(candidate, batch))
