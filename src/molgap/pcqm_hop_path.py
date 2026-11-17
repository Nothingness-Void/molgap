"""Static two/three-hop path relations for bounded PCQM GraphState screens."""
from __future__ import annotations

from collections import defaultdict, deque
import math

import torch
from torch import nn

from .pcqm_gap_architecture import (
    OGBLocalGlobalGeometrySparseTriangleEdgeStateGPSWrapper,
)


BASELINE = "ogb_distance_angle_triangle_edge_state_graph_state9"
HOP_PATH_CANDIDATE = (
    "ogb_distance_angle_hop_path_triangle_edge_state_graph_state9"
)
PATH_FEATURE_DIM = 8
PATH_BLOCKS = (3, 6, 9)


def _undirected_graph(graph) -> tuple[list[list[int]], dict[tuple[int, int], tuple[int, bool]]]:
    """Return sorted neighbors and deterministic OGB bond metadata."""
    node_count = int(graph.num_nodes)
    neighbors = [set() for _ in range(node_count)]
    bonds: dict[tuple[int, int], tuple[int, bool]] = {}
    edge_index = graph.edge_index.long().cpu()
    edge_attr = graph.edge_attr.long().cpu()
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E]")
    if tuple(edge_attr.shape) != (edge_index.shape[1], 3):
        raise ValueError("edge_attr must have shape [E, 3]")
    for ordinal in range(edge_index.shape[1]):
        source = int(edge_index[0, ordinal])
        target = int(edge_index[1, ordinal])
        if source == target:
            continue
        neighbors[source].add(target)
        neighbors[target].add(source)
        key = (min(source, target), max(source, target))
        metadata = (int(edge_attr[ordinal, 0]), bool(edge_attr[ordinal, 2]))
        previous = bonds.setdefault(key, metadata)
        if previous != metadata:
            raise ValueError(f"inconsistent directed bond metadata for {key}")
    return [sorted(items) for items in neighbors], bonds


def _shortest_hops(neighbors: list[list[int]], source: int) -> list[int]:
    distances = [-1] * len(neighbors)
    distances[source] = 0
    queue = deque([source])
    while queue:
        current = queue.popleft()
        if distances[current] == 3:
            continue
        for target in neighbors[current]:
            if distances[target] == -1:
                distances[target] = distances[current] + 1
                queue.append(target)
    return distances


def attach_hop_path_features(graph):
    """Attach ordered exact-shortest 2/3-hop relations and path statistics.

    Features are hop-2, hop-3, log path multiplicity, the mean fraction of
    single/double/triple/aromatic bonds across all shortest simple paths, and
    the fraction of paths whose complete bond sequence is conjugated.
    """
    neighbors, bonds = _undirected_graph(graph)
    grouped: dict[tuple[int, int, int], list[tuple[int, ...]]] = defaultdict(list)
    for source in range(len(neighbors)):
        distances = _shortest_hops(neighbors, source)

        def visit(path: tuple[int, ...], depth: int) -> None:
            if depth in (2, 3):
                target = path[-1]
                if distances[target] == depth:
                    grouped[(source, target, depth)].append(path)
            if depth == 3:
                return
            for target in neighbors[path[-1]]:
                if target not in path:
                    visit((*path, target), depth + 1)

        visit((source,), 0)

    relations: list[tuple[int, int]] = []
    features: list[list[float]] = []
    hop_counts = {2: 0, 3: 0}
    multipath_relations = 0
    maximum_path_count = 0
    for (source, target, hop), paths in sorted(grouped.items()):
        path_count = len(paths)
        if path_count == 0:
            continue
        bond_histogram = [0.0, 0.0, 0.0, 0.0]
        fully_conjugated = 0
        for path in paths:
            path_conjugated = True
            for left, right in zip(path, path[1:]):
                bond_type, conjugated = bonds[(min(left, right), max(left, right))]
                if 0 <= bond_type < 4:
                    bond_histogram[bond_type] += 1.0
                path_conjugated = path_conjugated and conjugated
            fully_conjugated += int(path_conjugated)
        denominator = float(path_count * hop)
        relations.append((source, target))
        features.append(
            [
                float(hop == 2),
                float(hop == 3),
                math.log1p(path_count),
                *(value / denominator for value in bond_histogram),
                fully_conjugated / float(path_count),
            ]
        )
        hop_counts[hop] += 1
        multipath_relations += int(path_count > 1)
        maximum_path_count = max(maximum_path_count, path_count)

    if relations:
        graph.hop_path_edge_index = torch.tensor(relations, dtype=torch.long).t().contiguous()
        graph.hop_path_features = torch.tensor(features, dtype=torch.float32)
    else:
        graph.hop_path_edge_index = torch.empty((2, 0), dtype=torch.long)
        graph.hop_path_features = torch.empty((0, PATH_FEATURE_DIM), dtype=torch.float32)
    if tuple(graph.hop_path_features.shape) != (
        graph.hop_path_edge_index.shape[1],
        PATH_FEATURE_DIM,
    ):
        raise RuntimeError("hop-path feature alignment failed")
    if not torch.isfinite(graph.hop_path_features).all():
        raise RuntimeError("hop-path cache contains non-finite values")
    return graph, {
        "hop2_relations": hop_counts[2],
        "hop3_relations": hop_counts[3],
        "multipath_relations": multipath_relations,
        "maximum_path_count": maximum_path_count,
    }


class _SharedHopPathMixer(nn.Module):
    """Shared sparse 2/3-hop message path with an identity-preserving return."""

    def __init__(
        self,
        atom_channels: int,
        feature_channels: int = PATH_FEATURE_DIM,
        path_channels: int = 32,
    ) -> None:
        super().__init__()
        self.feature_norm = nn.LayerNorm(feature_channels)
        self.feature_projection = nn.Linear(feature_channels, path_channels)
        self.source_projection = nn.Linear(atom_channels, path_channels, bias=False)
        self.target_projection = nn.Linear(atom_channels, path_channels, bias=False)
        context_channels = 3 * path_channels
        self.context_norm = nn.LayerNorm(context_channels)
        self.message_value = nn.Linear(context_channels, path_channels)
        self.message_gate = nn.Linear(context_channels, path_channels)
        self.output_norm = nn.LayerNorm(path_channels)
        self.output_value = nn.Linear(path_channels, atom_channels)
        self.output_gate = nn.Linear(path_channels, atom_channels)
        nn.init.zeros_(self.output_value.weight)
        nn.init.zeros_(self.output_value.bias)

    def forward(
        self,
        h: torch.Tensor,
        hop_path_edge_index: torch.Tensor,
        hop_path_features: torch.Tensor,
    ) -> torch.Tensor:
        if hop_path_edge_index.ndim != 2 or hop_path_edge_index.shape[0] != 2:
            raise ValueError("hop_path_edge_index must have shape [2, P]")
        if tuple(hop_path_features.shape) != (
            hop_path_edge_index.shape[1],
            PATH_FEATURE_DIM,
        ):
            raise ValueError("hop_path_features must have shape [P, 8]")
        if hop_path_edge_index.shape[1] == 0:
            return h
        source, target = hop_path_edge_index
        path = torch.nn.functional.silu(
            self.feature_projection(self.feature_norm(hop_path_features.float()))
        )
        context = self.context_norm(
            torch.cat(
                [
                    self.source_projection(h[source]),
                    self.target_projection(h[target]),
                    path,
                ],
                dim=-1,
            )
        )
        message = self.message_value(context) * torch.sigmoid(
            self.message_gate(context)
        )
        aggregate = message.new_zeros((h.shape[0], message.shape[1]))
        counts = message.new_zeros((h.shape[0], 1))
        aggregate.index_add_(0, target, message)
        counts.index_add_(0, target, message.new_ones((target.shape[0], 1)))
        aggregate = aggregate / counts.clamp_min_(1.0)
        latent = self.output_norm(aggregate)
        return h + self.output_value(latent) * torch.sigmoid(
            self.output_gate(latent)
        )


class OGBHopPathGraphStateWrapper(
    OGBLocalGlobalGeometrySparseTriangleEdgeStateGPSWrapper
):
    """GraphState9 plus a shared sparse exact-shortest 2/3-hop mixer."""

    def __init__(self, *args, path_channels: int = 32, **kwargs) -> None:
        super().__init__(*args, global_mode="graph_state", **kwargs)
        self.hop_path_mixer = _SharedHopPathMixer(
            self.head[0].in_features,
            path_channels=path_channels,
        )

    def forward(
        self,
        x,
        edge_index,
        edge_attr,
        batch,
        random_walk_pe,
        wedge_edge_ids,
        edge_distance,
        wedge_angle_cos,
        geometry_valid,
        hop_path_edge_index,
        hop_path_features,
    ):
        embedding = self._encode_geometry(
            x,
            edge_index,
            edge_attr,
            batch,
            random_walk_pe,
            wedge_edge_ids,
            edge_distance,
            wedge_angle_cos,
            geometry_valid,
            auxiliary_payload=(hop_path_edge_index, hop_path_features),
        )
        return self.head(embedding)

    def _initialize_geometry_auxiliary(self, h, batch, auxiliary_payload):
        if not isinstance(auxiliary_payload, tuple) or len(auxiliary_payload) != 2:
            raise ValueError("Hop-path GraphState requires cached path relations")
        hop_path_edge_index, hop_path_features = auxiliary_payload
        return {
            "graph_state": self.graph_context.initialize(h, batch),
            "hop_path_edge_index": hop_path_edge_index,
            "hop_path_features": hop_path_features,
        }

    def _update_geometry_auxiliary(
        self,
        layer,
        h,
        edge_index,
        edge_state,
        batch,
        auxiliary_state,
    ):
        block = layer + 1
        if block in self.GLOBAL_BLOCKS:
            h, auxiliary_state["graph_state"] = self.graph_context(
                h, batch, auxiliary_state["graph_state"]
            )
        if block in PATH_BLOCKS:
            h = self.hop_path_mixer(
                h,
                auxiliary_state["hop_path_edge_index"],
                auxiliary_state["hop_path_features"],
            )
        return h, edge_state, auxiliary_state


def make_hop_path_encoder(candidate: str):
    if candidate != HOP_PATH_CANDIDATE:
        raise ValueError(f"Unknown hop-path candidate: {candidate}")
    return OGBHopPathGraphStateWrapper(
        in_channels=9,
        edge_dim=3,
        hidden_channels=192,
        num_layers=9,
        num_heads=4,
        dropout=0.1,
        n_targets=1,
        pooling="mean",
        rwse_dim=16,
        edge_state_channels=64,
        wedge_channels=16,
        geometry_basis_channels=16,
        graph_state_channels=64,
        graph_exchange_rank=32,
        path_channels=32,
    )
