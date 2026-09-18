"""Sparse topology-wedge cache primitives used by accepted PCQM graph caches."""
from __future__ import annotations

from typing import Iterable

import torch
from torch_geometric.data import Data


class WedgeData(Data):
    """PyG data whose wedge pairs index directed edges rather than nodes."""

    def __inc__(self, key, value, *args, **kwargs):
        if key == "wedge_edge_ids":
            return int(self.edge_index.shape[1])
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key == "wedge_edge_ids":
            return 0
        return super().__cat_dim__(key, value, *args, **kwargs)


def directed_nonbacktracking_wedges(edge_index: torch.Tensor) -> torch.Tensor:
    """Return directed edge-id pairs for every non-backtracking path i-j-k."""
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
    """Copy an accepted graph and attach immutable wedge edge ids."""
    if not hasattr(graph, "edge_index"):
        raise ValueError("graph is missing edge_index")
    result = WedgeData(**graph.to_dict())
    result.wedge_edge_ids = directed_nonbacktracking_wedges(graph.edge_index)
    return result


def wedge_count(graphs: Iterable[Data]) -> int:
    """Return the total number of cached wedges."""
    return sum(int(graph.wedge_edge_ids.shape[0]) for graph in graphs)
