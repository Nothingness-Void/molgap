"""Sparse topology-wedge cache primitives for the PCQM Gap screen.

The cache stores directed non-backtracking pairs of bond edges.  A pair is
represented by the indices of ``i -> j`` and ``j -> k`` in a graph's directed
``edge_index``.  Keeping edge ids instead of materializing a dense atom-pair
matrix makes the representation scale with local molecular connectivity.
"""
from __future__ import annotations

from typing import Iterable

import torch
from torch_geometric.data import Data


class WedgeData(Data):
    """PyG data with a batched edge-id pair attribute.

    PyG increments attributes containing ``index`` by node count during
    batching.  Wedge pairs index directed edges, so their increment must be
    the graph's directed edge count instead.
    """

    def __inc__(self, key, value, *args, **kwargs):
        if key == "wedge_edge_ids":
            return int(self.edge_index.shape[1])
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key == "wedge_edge_ids":
            return 0
        return super().__cat_dim__(key, value, *args, **kwargs)


def directed_nonbacktracking_wedges(edge_index: torch.Tensor) -> torch.Tensor:
    """Return ``[num_wedges, 2]`` directed edge-id pairs for one graph.

    For every directed path ``i -> j -> k`` the immediate reversal ``k == i``
    is excluded.  Both orientations of an undirected chemical wedge are kept,
    because the EdgeState backbone is directed by construction.
    """
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E]")
    if edge_index.dtype != torch.long:
        edge_index = edge_index.long()
    num_edges = int(edge_index.shape[1])
    num_nodes = int(edge_index.max().item()) + 1 if num_edges else 0
    source = edge_index[0].tolist()
    target = edge_index[1].tolist()
    incoming: list[list[int]] = [[] for _ in range(num_nodes)]
    outgoing: list[list[int]] = [[] for _ in range(num_nodes)]
    for edge_id, (src, dst) in enumerate(zip(source, target)):
        if not 0 <= src < num_nodes or not 0 <= dst < num_nodes:
            raise ValueError("edge_index contains an out-of-range node")
        incoming[dst].append(edge_id)
        outgoing[src].append(edge_id)

    pairs: list[tuple[int, int]] = []
    for center in range(num_nodes):
        for first in incoming[center]:
            for second in outgoing[center]:
                if source[first] != target[second]:
                    pairs.append((first, second))
    if not pairs:
        return torch.empty((0, 2), dtype=torch.long)
    return torch.tensor(pairs, dtype=torch.long)


def with_wedge_cache(graph: Data) -> WedgeData:
    """Copy one accepted graph and attach its immutable wedge edge ids."""
    if not hasattr(graph, "edge_index"):
        raise ValueError("graph is missing edge_index")
    payload = graph.to_dict()
    result = WedgeData(**payload)
    result.wedge_edge_ids = directed_nonbacktracking_wedges(graph.edge_index)
    return result


def wedge_count(graphs: Iterable[Data]) -> int:
    """Return the total number of cached wedges without running a model."""
    return sum(int(graph.wedge_edge_ids.shape[0]) for graph in graphs)
