"""Deterministic non-covalent contact relations for accepted PCQM geometry."""
from __future__ import annotations

from collections import Counter, deque
from copy import copy

import torch


CONTACT_CUTOFF_ANGSTROM = 5.0
EXCLUDED_COVALENT_HOPS = 3


def _adjacency(edge_index: torch.Tensor, node_count: int) -> list[set[int]]:
    adjacency = [set() for _ in range(node_count)]
    for source, target in edge_index.t().tolist():
        source = int(source)
        target = int(target)
        if source == target:
            continue
        adjacency[source].add(target)
        adjacency[target].add(source)
    return adjacency


def _within_hops(adjacency: list[set[int]], source: int, maximum: int) -> set[int]:
    reached = {source}
    queue = deque([(source, 0)])
    while queue:
        node, depth = queue.popleft()
        if depth == maximum:
            continue
        for neighbor in sorted(adjacency[node]):
            if neighbor in reached:
                continue
            reached.add(neighbor)
            queue.append((neighbor, depth + 1))
    return reached


def _components(adjacency: list[set[int]]) -> list[int]:
    labels = [-1] * len(adjacency)
    component = 0
    for root in range(len(adjacency)):
        if labels[root] >= 0:
            continue
        labels[root] = component
        queue = deque([root])
        while queue:
            node = queue.popleft()
            for neighbor in sorted(adjacency[node]):
                if labels[neighbor] >= 0:
                    continue
                labels[neighbor] = component
                queue.append(neighbor)
        component += 1
    return labels


def with_non_covalent_contacts(graph):
    """Return a shallow graph copy with deterministic directed contact edges."""
    result = copy(graph)
    node_count = int(graph.num_nodes)
    empty_index = torch.empty((2, 0), dtype=torch.long)
    empty_distance = torch.empty((0, 1), dtype=torch.float32)
    empty_cross_component = torch.empty((0,), dtype=torch.bool)
    geometry_valid = bool(float(graph.geometry_valid.view(-1)[0]))
    if not geometry_valid or node_count < 2:
        result.contact_edge_index = empty_index
        result.contact_distance = empty_distance
        result.contact_cross_component = empty_cross_component
        return result

    positions = graph.pos.to(dtype=torch.float64, device="cpu")
    if positions.shape != (node_count, 3) or not torch.isfinite(positions).all():
        raise ValueError("Contact construction requires finite [num_nodes, 3] positions")
    adjacency = _adjacency(graph.edge_index.to(device="cpu"), node_count)
    excluded = [
        _within_hops(adjacency, source, EXCLUDED_COVALENT_HOPS)
        for source in range(node_count)
    ]
    components = _components(adjacency)

    undirected: list[tuple[int, int, float, bool]] = []
    for source in range(node_count):
        for target in range(source + 1, node_count):
            if target in excluded[source]:
                continue
            distance = float(torch.linalg.vector_norm(positions[source] - positions[target]))
            if 0.0 < distance <= CONTACT_CUTOFF_ANGSTROM:
                undirected.append(
                    (source, target, distance, components[source] != components[target])
                )

    if not undirected:
        result.contact_edge_index = empty_index
        result.contact_distance = empty_distance
        result.contact_cross_component = empty_cross_component
        return result

    directed = []
    distances = []
    cross_component = []
    for source, target, distance, is_cross_component in undirected:
        directed.extend(((source, target), (target, source)))
        distances.extend((distance, distance))
        cross_component.extend((is_cross_component, is_cross_component))
    result.contact_edge_index = torch.tensor(directed, dtype=torch.long).t().contiguous()
    result.contact_distance = torch.tensor(distances, dtype=torch.float32).view(-1, 1)
    result.contact_cross_component = torch.tensor(cross_component, dtype=torch.bool)
    return result


def contact_statistics(graphs) -> dict:
    """Aggregate role-level cache statistics without reading target values."""
    atom_type_pairs: Counter[str] = Counter()
    graphs_with_contacts = 0
    atoms_with_contacts = 0
    directed_edges = 0
    cross_component_directed_edges = 0
    valid_geometry_graphs = 0
    invalid_geometry_graphs = 0
    maximum_directed_edges = 0
    for graph in graphs:
        valid = bool(float(graph.geometry_valid.view(-1)[0]))
        valid_geometry_graphs += int(valid)
        invalid_geometry_graphs += int(not valid)
        edge_count = int(graph.contact_edge_index.shape[1])
        directed_edges += edge_count
        maximum_directed_edges = max(maximum_directed_edges, edge_count)
        cross_component_directed_edges += int(graph.contact_cross_component.sum())
        if edge_count == 0:
            continue
        graphs_with_contacts += 1
        atoms_with_contacts += int(torch.unique(graph.contact_edge_index).numel())
        atomic_type = graph.x[:, 0].to(dtype=torch.long, device="cpu")
        for source, target in graph.contact_edge_index[:, ::2].t().tolist():
            left = int(atomic_type[source])
            right = int(atomic_type[target])
            key = f"{min(left, right)}:{max(left, right)}"
            atom_type_pairs[key] += 1
    return {
        "graphs": len(graphs),
        "valid_geometry_graphs": valid_geometry_graphs,
        "invalid_geometry_graphs": invalid_geometry_graphs,
        "graphs_with_contacts": graphs_with_contacts,
        "atoms_with_contacts": atoms_with_contacts,
        "undirected_pairs": directed_edges // 2,
        "directed_edges": directed_edges,
        "cross_component_directed_edges": cross_component_directed_edges,
        "maximum_directed_edges_per_graph": maximum_directed_edges,
        "atom_type_pairs": dict(sorted(atom_type_pairs.items())),
    }


def contact_contract_violations(graph) -> list[str]:
    """Return mechanical relation violations for a converted graph."""
    errors: list[str] = []
    edge_index = graph.contact_edge_index.to(device="cpu")
    distance = graph.contact_distance.to(device="cpu").view(-1)
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        return ["contact_edge_index_shape"]
    if edge_index.shape[1] != distance.numel():
        errors.append("contact_distance_count")
        return errors
    if distance.numel() and (
        not torch.isfinite(distance).all()
        or bool((distance <= 0).any())
        or bool((distance > CONTACT_CUTOFF_ANGSTROM).any())
    ):
        errors.append("contact_distance_range")
    directed = [tuple(map(int, pair)) for pair in edge_index.t().tolist()]
    if len(directed) != len(set(directed)):
        errors.append("duplicate_directed_contact")
    directed_set = set(directed)
    if any((target, source) not in directed_set for source, target in directed):
        errors.append("missing_reverse_contact")
    adjacency = _adjacency(graph.edge_index.to(device="cpu"), int(graph.num_nodes))
    excluded = [
        _within_hops(adjacency, source, EXCLUDED_COVALENT_HOPS)
        for source in range(int(graph.num_nodes))
    ]
    if any(target in excluded[source] for source, target in directed):
        errors.append("excluded_covalent_hop_overlap")
    return errors
