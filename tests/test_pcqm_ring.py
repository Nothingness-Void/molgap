from __future__ import annotations

import torch
from ogb.utils import smiles2graph
from torch_geometric.data import Batch, Data

from molgap.pcqm_ring import (
    RING_EDGE_FEATURE_CHANNELS,
    RING_FEATURE_CHANNELS,
    RingHierarchyData,
    with_ring_hierarchy,
)


def graph_from_smiles(smiles: str) -> Data:
    payload = smiles2graph(smiles)
    return Data(
        x=torch.from_numpy(payload["node_feat"]).long(),
        edge_index=torch.from_numpy(payload["edge_index"]).long(),
        edge_attr=torch.from_numpy(payload["edge_feat"]).long(),
        wedge_edge_ids=torch.empty((0, 2), dtype=torch.long),
    )


def test_ring_hierarchy_handles_acyclic_and_single_ring_molecules():
    acyclic = with_ring_hierarchy(graph_from_smiles("CCO"), "CCO")
    assert acyclic.ring_features.shape == (0, RING_FEATURE_CHANNELS)
    assert acyclic.atom_ring_index.shape == (2, 0)
    assert acyclic.ring_edge_attr.shape == (0, RING_EDGE_FEATURE_CHANNELS)

    benzene = with_ring_hierarchy(
        graph_from_smiles("c1ccccc1"), "c1ccccc1"
    )
    assert benzene.ring_features.shape == (1, RING_FEATURE_CHANNELS)
    assert benzene.atom_ring_index.shape == (2, 6)
    assert benzene.ring_edge_index.shape == (2, 0)
    assert benzene.ring_features[0, 6].item() == 1.0
    assert benzene.ring_features[0, 7].item() == 1.0


def test_ring_relations_distinguish_fused_spiro_and_direct_pairs():
    fused = with_ring_hierarchy(
        graph_from_smiles("c1ccc2ccccc2c1"), "c1ccc2ccccc2c1"
    )
    assert fused.ring_features.shape[0] == 2
    assert fused.ring_edge_index.shape[1] == 2
    assert fused.ring_edge_attr[:, 1].tolist() == [1.0, 1.0]

    spiro_smiles = "C1CCC2(CC1)CCCC2"
    spiro = with_ring_hierarchy(
        graph_from_smiles(spiro_smiles), spiro_smiles
    )
    assert spiro.ring_features.shape[0] == 2
    assert spiro.ring_edge_attr[:, 0].tolist() == [1.0, 1.0]

    biphenyl = with_ring_hierarchy(
        graph_from_smiles("c1ccccc1-c2ccccc2"),
        "c1ccccc1-c2ccccc2",
    )
    assert biphenyl.ring_features.shape[0] == 2
    assert biphenyl.ring_edge_attr[:, 2].tolist() == [1.0, 1.0]


def test_ring_indices_increment_by_nodes_and_rings_when_batched():
    first = with_ring_hierarchy(
        graph_from_smiles("c1ccccc1"), "c1ccccc1"
    )
    second = with_ring_hierarchy(
        graph_from_smiles("c1ccncc1"), "c1ccncc1"
    )
    assert isinstance(first, RingHierarchyData)
    batch = Batch.from_data_list([first, second])
    assert batch.ring_features.shape[0] == 2
    assert batch.atom_ring_index[:, :6].tolist() == [
        [0, 1, 2, 3, 4, 5],
        [0, 0, 0, 0, 0, 0],
    ]
    assert batch.atom_ring_index[:, 6:].tolist() == [
        [6, 7, 8, 9, 10, 11],
        [1, 1, 1, 1, 1, 1],
    ]


def test_smiles_parent_mismatch_is_rejected():
    graph = graph_from_smiles("CCO")
    try:
        with_ring_hierarchy(graph, "CCC")
    except ValueError as error:
        assert "features do not match" in str(error)
    else:
        raise AssertionError("mismatched parent graph was accepted")
