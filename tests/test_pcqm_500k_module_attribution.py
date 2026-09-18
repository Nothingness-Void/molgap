from __future__ import annotations

import numpy as np

from molgap.pcqm_500k_module_attribution import _graph_distances, _quantile_bins


def test_graph_distances_for_chain() -> None:
    edges = np.array([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
    diameter, mean_distance, components = _graph_distances(4, edges)
    assert diameter == 3
    assert mean_distance == 10 / 6
    assert components == 1


def test_graph_distances_for_disconnected_graph() -> None:
    edges = np.array([[0, 1, 2, 3], [1, 0, 3, 2]])
    diameter, mean_distance, components = _graph_distances(4, edges)
    assert diameter == 1
    assert mean_distance == 1
    assert components == 2


def test_quantile_bins_cover_rows_once() -> None:
    values = np.arange(20, dtype=np.float64)
    bins = _quantile_bins(values)
    assert len(bins) == 4
    assert np.array_equal(np.sum(np.stack(bins), axis=0), np.ones(20))

