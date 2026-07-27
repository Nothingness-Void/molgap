import torch

from molgap.pcqm_route_b import ATOM_LIST, build_route_b_row


def test_route_b_row_builds_aligned_expanded_element_graphs():
    source_idx, result = build_route_b_row((17, "P(=O)(O)O", 4.2, 0))

    assert source_idx == 17
    assert result is not None
    gps, primary, secondary = result
    assert gps.x.shape[1] == len(ATOM_LIST) + 3
    assert gps.x[:, ATOM_LIST.index(15)].sum().item() == 1
    assert primary.pos.shape == secondary.pos.shape
    assert not torch.equal(primary.pos, secondary.pos)
    assert all(graph.source_idx.item() == 17 for graph in result)
    assert all(graph.y.item() == torch.tensor(4.2).item() for graph in result)
