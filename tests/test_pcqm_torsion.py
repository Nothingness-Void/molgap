from __future__ import annotations

import numpy as np
import torch

from molgap.pcqm_torsion import (
    TorsionData,
    directed_nonbacktracking_torsions,
    torsion_fourier_features,
)


def bidirectional_chain() -> torch.Tensor:
    return torch.tensor(
        [
            [0, 1, 1, 2, 2, 3],
            [1, 0, 2, 1, 3, 2],
        ],
        dtype=torch.long,
    )


def test_torsions_are_nonbacktracking_and_link_adjacent_wedges():
    edge_index = bidirectional_chain()
    wedge_pairs = []
    for first in range(edge_index.shape[1]):
        for second in range(edge_index.shape[1]):
            if edge_index[1, first] == edge_index[0, second] and edge_index[0, first] != edge_index[1, second]:
                wedge_pairs.append((first, second))
    wedges = torch.tensor(wedge_pairs, dtype=torch.long)
    torsion_edges, torsion_wedges = directed_nonbacktracking_torsions(
        edge_index, wedges
    )
    assert torsion_edges.shape[1] == 3
    assert torsion_wedges.shape[1] == 2
    assert torsion_edges.shape[0] == 2
    for first, second, third in torsion_edges.tolist():
        assert edge_index[1, first] == edge_index[0, second]
        assert edge_index[1, second] == edge_index[0, third]
        assert edge_index[0, first] != edge_index[1, second]
        assert edge_index[0, second] != edge_index[1, third]
    for first_wedge, second_wedge in torsion_wedges.tolist():
        assert wedges[first_wedge, 1] == wedges[second_wedge, 0]


def test_torsion_features_are_fixed_periodic_and_mask_invalid_rows():
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long
    )
    torsion_edges = torch.tensor([[0, 2, 4]], dtype=torch.long)
    positions = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [2.0, 1.0, 1.0]],
        dtype=np.float32,
    )
    features, valid = torsion_fourier_features(
        positions, edge_index, torsion_edges, True
    )
    assert features.shape == (1, 4)
    assert valid.tolist() == [[1.0]]
    assert np.allclose(features[0, 0] ** 2 + features[0, 1] ** 2, 1.0, atol=1e-6)
    assert np.allclose(features[0, 2] ** 2 + features[0, 3] ** 2, 1.0, atol=1e-6)
    invalid_features, invalid = torsion_fourier_features(
        positions, edge_index, torsion_edges, False
    )
    assert not invalid.any()
    assert not invalid_features.any()


def test_torsion_data_increments_edge_and_wedge_ids_separately():
    first = TorsionData(
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        num_nodes=2,
        wedge_edge_ids=torch.tensor([[0, 1]], dtype=torch.long),
        torsion_edge_ids=torch.tensor([[0, 1, 0]], dtype=torch.long),
        torsion_wedge_ids=torch.tensor([[0, 0]], dtype=torch.long),
        torsion_fourier=torch.zeros((1, 4)),
        torsion_valid=torch.ones((1, 1)),
    )
    second = TorsionData(
        edge_index=torch.tensor([[0, 1, 1, 0], [1, 0, 0, 1]], dtype=torch.long),
        num_nodes=2,
        wedge_edge_ids=torch.tensor([[0, 1]], dtype=torch.long),
        torsion_edge_ids=torch.tensor([[0, 1, 2]], dtype=torch.long),
        torsion_wedge_ids=torch.tensor([[0, 0]], dtype=torch.long),
        torsion_fourier=torch.zeros((1, 4)),
        torsion_valid=torch.ones((1, 1)),
    )
    from torch_geometric.data import Batch

    batch = Batch.from_data_list([first, second])
    assert batch.wedge_edge_ids.tolist() == [[0, 1], [2, 3]]
    assert batch.torsion_edge_ids.tolist() == [[0, 1, 0], [2, 3, 4]]
    assert batch.torsion_wedge_ids.tolist() == [[0, 0], [1, 1]]
