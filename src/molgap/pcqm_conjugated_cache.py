"""Deterministic conjugated-system membership and descriptor construction."""
from __future__ import annotations

import math

import torch


FEATURE_NAMES = (
    "log_atom_count",
    "log_conjugated_bond_count",
    "cycle_rank_fraction",
    "aromatic_atom_fraction",
    "hetero_atom_fraction",
    "branch_atom_fraction",
    "endpoint_atom_fraction",
    "aromatic_bond_fraction",
)


def with_conjugated_components(graph):
    """Attach local component ids and repeated invariant descriptors.

    OGB bond feature column 2 is the conjugation flag. Components contain only
    atoms incident to at least one conjugated bond; other atoms receive id -1
    and an all-zero descriptor. Local IDs are ordered by smallest atom index.
    """
    node_count = int(graph.num_nodes)
    if graph.x.ndim != 2 or graph.x.shape[0] != node_count:
        raise ValueError("atom features are not aligned")
    if graph.edge_attr.ndim != 2 or graph.edge_attr.shape[1] != 3:
        raise ValueError("expected three OGB bond feature columns")
    if graph.edge_index.shape != (2, graph.edge_attr.shape[0]):
        raise ValueError("bond features are not aligned")

    parent = list(range(node_count))

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(first: int, second: int) -> None:
        first_root, second_root = find(first), find(second)
        if first_root == second_root:
            return
        if first_root < second_root:
            parent[second_root] = first_root
        else:
            parent[first_root] = second_root

    conjugated_edges = set()
    for edge_id in range(graph.edge_index.shape[1]):
        if int(graph.edge_attr[edge_id, 2]) != 1:
            continue
        source = int(graph.edge_index[0, edge_id])
        target = int(graph.edge_index[1, edge_id])
        if source == target:
            continue
        edge = (source, target) if source < target else (target, source)
        conjugated_edges.add(edge)
        union(*edge)

    members_by_root: dict[int, set[int]] = {}
    for source, target in conjugated_edges:
        root = find(source)
        members_by_root.setdefault(root, set()).update((source, target))
    components = sorted(
        (sorted(members) for members in members_by_root.values()),
        key=lambda members: members[0],
    )

    component_id = torch.full((node_count,), -1, dtype=torch.long)
    features = torch.zeros((node_count, len(FEATURE_NAMES)), dtype=torch.float32)
    for local_id, members in enumerate(components):
        member_set = set(members)
        edges = [
            edge for edge in conjugated_edges
            if edge[0] in member_set and edge[1] in member_set
        ]
        degrees = {atom: 0 for atom in members}
        for source, target in edges:
            degrees[source] += 1
            degrees[target] += 1
        atom_count = len(members)
        edge_count = len(edges)
        cycle_rank = max(0, edge_count - atom_count + 1)
        aromatic_atoms = sum(int(graph.x[atom, 7]) == 1 for atom in members)
        hetero_atoms = sum(int(graph.x[atom, 0]) != 5 for atom in members)
        branch_atoms = sum(degree >= 3 for degree in degrees.values())
        endpoint_atoms = sum(degree == 1 for degree in degrees.values())
        aromatic_edges = 0
        for source, target in edges:
            matches = (
                ((graph.edge_index[0] == source) & (graph.edge_index[1] == target))
                | ((graph.edge_index[0] == target) & (graph.edge_index[1] == source))
            )
            edge_ids = torch.nonzero(matches, as_tuple=False).reshape(-1)
            if edge_ids.numel() and int(graph.edge_attr[edge_ids[0], 0]) == 3:
                aromatic_edges += 1
        descriptor = torch.tensor(
            [
                math.log1p(atom_count) / 5.0,
                math.log1p(edge_count) / 5.0,
                cycle_rank / max(1, atom_count),
                aromatic_atoms / atom_count,
                hetero_atoms / atom_count,
                branch_atoms / atom_count,
                endpoint_atoms / atom_count,
                aromatic_edges / max(1, edge_count),
            ],
            dtype=torch.float32,
        )
        ids = torch.tensor(members, dtype=torch.long)
        component_id[ids] = local_id
        features[ids] = descriptor

    graph.conjugated_component_id = component_id
    graph.conjugated_component_count = torch.tensor(
        [len(components)], dtype=torch.long
    )
    graph.conjugated_features = features
    return graph


def conjugated_counts(graphs) -> dict[str, int]:
    counts = [int(graph.conjugated_component_count.view(-1)[0]) for graph in graphs]
    return {
        "graphs": len(graphs),
        "components": sum(counts),
        "component_atoms": sum(
            int((graph.conjugated_component_id >= 0).sum()) for graph in graphs
        ),
        "graphs_with_components": sum(value > 0 for value in counts),
        "max_components_per_graph": max(counts, default=0),
    }


__all__ = ["FEATURE_NAMES", "with_conjugated_components", "conjugated_counts"]
