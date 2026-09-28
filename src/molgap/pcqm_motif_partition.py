"""Deterministic HieGT-inspired motif partition from accepted OGB 2D graphs.

This is a topology-only preprocessing primitive, not the published HieGT
architecture.  A motif is a connected component after cutting selected
non-ring bonds; no coordinates, labels, or model outputs enter the partition.
"""
from __future__ import annotations

from collections import deque


RULES = (
    "ring_to_chain_bridge",
    "nonring_carbon_hetero_bridge",
    "nonring_nonsingle_bridge",
)


def derive_motif_partition(graph) -> dict:
    """Return node-to-motif assignment and directed inter-motif real bonds.

    The OGB categorical contract stores carbon at atomic-number index 5,
    `is_in_ring` at atom column 8, and SINGLE at bond-type index 0.
    Graph-theoretic bridges are exactly the bonds outside all cycles, so
    cutting them cannot fragment an aromatic/cyclic motif.
    """
    import torch

    x = graph.x.long()
    edge_index = graph.edge_index.long()
    edge_attr = graph.edge_attr.long()
    n = int(graph.num_nodes)
    if x.ndim != 2 or x.shape[0] != n or x.shape[1] != 9 or n < 1:
        raise ValueError("Expected the fixed OGB nine-category atom tensor")
    if (
        edge_index.ndim != 2 or edge_index.shape[0] != 2
        or edge_attr.ndim != 2 or edge_attr.shape != (edge_index.shape[1], 3)
    ):
        raise ValueError("Expected paired real bonds with three OGB categories")
    if not bool(((x[:, 8] == 0) | (x[:, 8] == 1)).all()):
        raise ValueError("Invalid OGB ring flags")

    directed: dict[tuple[int, int], tuple[int, int, int]] = {}
    for index in range(edge_index.shape[1]):
        u, v = (int(value) for value in edge_index[:, index])
        if u == v or not (0 <= u < n and 0 <= v < n) or (u, v) in directed:
            raise ValueError("Real-bond graph has a loop, duplicate or invalid atom")
        directed[u, v] = tuple(int(value) for value in edge_attr[index])
    if any(directed.get((v, u)) != attr for (u, v), attr in directed.items()):
        raise ValueError("Real bonds must be bidirectional with matching categories")

    adjacency: list[list[int]] = [[] for _ in range(n)]
    bonds: list[tuple[int, int, tuple[int, int, int]]] = []
    for (u, v), attr in sorted(directed.items()):
        if u < v:
            adjacency[u].append(v)
            adjacency[v].append(u)
            bonds.append((u, v, attr))
    for neighbors in adjacency:
        neighbors.sort()

    # Tarjan low-link labels only graph-theoretic bridges.  The iterative
    # version avoids recursion-limit dependence on unusual input graphs.
    discovered = [-1] * n
    low = [0] * n
    parent = [-1] * n
    bridges: set[tuple[int, int]] = set()
    clock = 0
    for root in range(n):
        if discovered[root] != -1:
            continue
        discovered[root] = low[root] = clock
        clock += 1
        stack = [(root, 0)]
        while stack:
            vertex, offset = stack[-1]
            if offset == len(adjacency[vertex]):
                stack.pop()
                ancestor = parent[vertex]
                if ancestor != -1:
                    low[ancestor] = min(low[ancestor], low[vertex])
                    if low[vertex] > discovered[ancestor]:
                        bridges.add((min(ancestor, vertex), max(ancestor, vertex)))
                continue
            neighbor = adjacency[vertex][offset]
            stack[-1] = (vertex, offset + 1)
            if neighbor == parent[vertex]:
                continue
            if discovered[neighbor] == -1:
                parent[neighbor] = vertex
                discovered[neighbor] = low[neighbor] = clock
                clock += 1
                stack.append((neighbor, 0))
            else:
                low[vertex] = min(low[vertex], discovered[neighbor])

    cut: dict[tuple[int, int], int] = {}
    for u, v, attr in bonds:
        if (u, v) not in bridges:
            continue
        ring_u, ring_v = int(x[u, 8]), int(x[v, 8])
        carbon_u, carbon_v = int(x[u, 0]) == 5, int(x[v, 0]) == 5
        reason = (
            int(ring_u != ring_v)
            | (int(carbon_u != carbon_v) << 1)
            | (int(attr[0] != 0) << 2)
        )
        if reason:
            cut[u, v] = reason

    retained: list[list[int]] = [[] for _ in range(n)]
    for u, v, _ in bonds:
        if (u, v) not in cut:
            retained[u].append(v)
            retained[v].append(u)
    motif = [-1] * n
    count = 0
    for root in range(n):
        if motif[root] != -1:
            continue
        motif[root] = count
        pending = deque([root])
        while pending:
            vertex = pending.popleft()
            for neighbor in retained[vertex]:
                if motif[neighbor] == -1:
                    motif[neighbor] = count
                    pending.append(neighbor)
        count += 1

    inter_edges: list[tuple[int, int]] = []
    inter_bond_type: list[int] = []
    inter_rule_mask: list[int] = []
    for u, v, attr in bonds:
        if motif[u] == motif[v]:
            if (u, v) in cut:
                raise RuntimeError("Cut bond failed to separate motifs")
            continue
        if (u, v) not in cut:
            raise RuntimeError("Uncut bond crosses motif boundary")
        for source, target in ((motif[u], motif[v]), (motif[v], motif[u])):
            inter_edges.append((source, target))
            inter_bond_type.append(attr[0])
            inter_rule_mask.append(cut[u, v])
    motif_edges = (
        torch.tensor(inter_edges, dtype=torch.long).t().contiguous()
        if inter_edges else torch.empty((2, 0), dtype=torch.long)
    )
    return {
        "motif_index": torch.tensor(motif, dtype=torch.long),
        "motif_count": count,
        "motif_edge_index": motif_edges,
        "motif_bond_type": torch.tensor(inter_bond_type, dtype=torch.long),
        "motif_bridge_rule_mask": torch.tensor(inter_rule_mask, dtype=torch.uint8),
    }
