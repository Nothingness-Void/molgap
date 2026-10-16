"""Bounded local-state additions for PCQM GraphState architecture screens."""
from __future__ import annotations

import torch
from torch import nn

from .pcqm_gap_architecture import (
    OGBLocalGlobalGeometrySparseTriangleEdgeStateGPSWrapper,
)


BASELINE = "ogb_distance_angle_triangle_edge_state_graph_state9"
PNA_CANDIDATE = (
    "ogb_distance_angle_pna_statistics_triangle_edge_state_graph_state9"
)
RETENTION_CANDIDATE = (
    "ogb_distance_angle_retention_triangle_edge_state_graph_state9"
)


class _SharedPNANeighborhoodState(nn.Module):
    """Shared mean/max/min/std neighborhood path with a zero-start return."""

    def __init__(self, atom_channels: int, edge_channels: int, rank: int) -> None:
        super().__init__()
        message_channels = edge_channels
        self.message_norm = nn.LayerNorm(atom_channels + edge_channels)
        self.message = nn.Linear(atom_channels + edge_channels, message_channels)
        statistic_channels = 4 * message_channels + 1
        self.statistics_norm = nn.LayerNorm(statistic_channels)
        self.down = nn.Linear(statistic_channels, rank)
        self.value = nn.Linear(rank, atom_channels)
        self.gate = nn.Linear(rank, atom_channels)
        nn.init.zeros_(self.value.weight)
        nn.init.zeros_(self.value.bias)

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_state: torch.Tensor,
    ) -> torch.Tensor:
        source, destination = edge_index
        node_count = h.shape[0]
        channels = edge_state.shape[1]
        messages = self.message(
            self.message_norm(torch.cat([h[source], edge_state], dim=-1))
        )
        counts = messages.new_zeros((node_count, 1))
        sums = messages.new_zeros((node_count, channels))
        squares = messages.new_zeros((node_count, channels))
        ones = messages.new_ones((messages.shape[0], 1))
        counts.index_add_(0, destination, ones)
        sums.index_add_(0, destination, messages)
        squares.index_add_(0, destination, messages.square())
        denominator = counts.clamp_min(1.0)
        mean = sums / denominator
        variance = (squares / denominator - mean.square()).clamp_min(0.0)
        standard_deviation = torch.sqrt(variance + 1.0e-8)

        maximum = messages.new_full((node_count, channels), -torch.inf)
        minimum = messages.new_full((node_count, channels), torch.inf)
        if messages.shape[0]:
            expanded = destination.view(-1, 1).expand(-1, channels)
            maximum.scatter_reduce_(0, expanded, messages, "amax", include_self=True)
            minimum.scatter_reduce_(0, expanded, messages, "amin", include_self=True)
        empty = counts == 0
        standard_deviation = torch.where(
            empty, torch.zeros_like(standard_deviation), standard_deviation
        )
        maximum = torch.where(empty, torch.zeros_like(maximum), maximum)
        minimum = torch.where(empty, torch.zeros_like(minimum), minimum)
        degree = torch.log1p(counts)
        statistics = torch.cat(
            [mean, maximum, minimum, standard_deviation, degree], dim=-1
        )
        latent = torch.nn.functional.silu(
            self.down(self.statistics_norm(statistics))
        )
        return h + self.value(latent) * torch.sigmoid(self.gate(latent))


class OGBPNAStatisticsGraphStateWrapper(
    OGBLocalGlobalGeometrySparseTriangleEdgeStateGPSWrapper
):
    """GraphState9 plus one shared PNA-style neighborhood statistics path."""

    def __init__(self, *args, pna_rank: int = 64, **kwargs) -> None:
        super().__init__(*args, global_mode="graph_state", **kwargs)
        self.pna_statistics = _SharedPNANeighborhoodState(
            self.head[0].in_features,
            self.edge_state_channels,
            pna_rank,
        )

    def _update_geometry_auxiliary(
        self,
        layer,
        h,
        edge_index,
        edge_state,
        batch,
        auxiliary_state,
    ):
        h, edge_state, auxiliary_state = super()._update_geometry_auxiliary(
            layer, h, edge_index, edge_state, batch, auxiliary_state
        )
        h = self.pna_statistics(h, edge_index, edge_state)
        return h, edge_state, auxiliary_state


class _GatedPersistentEdgeUpdate(nn.Module):
    """Preserve the baseline update and learn a zero-start retention correction."""

    def __init__(self, base: nn.Module, edge_channels: int, rank: int) -> None:
        super().__init__()
        # Keep baseline names stable so paired initialization can be verified.
        self.source = base.source
        self.target = base.target
        self.update = base.update
        self.output_norm = base.output_norm
        self.retention_norm = nn.LayerNorm(2 * edge_channels)
        self.retention_down = nn.Linear(2 * edge_channels, rank)
        self.retention_value = nn.Linear(rank, edge_channels)
        self.retention_gate = nn.Linear(rank, edge_channels)
        nn.init.zeros_(self.retention_value.weight)
        nn.init.zeros_(self.retention_value.bias)

    def forward(self, h, edge_index, edge_state):
        source, target = edge_index
        context = edge_state + self.source(h[source]) + self.target(h[target])
        proposal = self.output_norm(edge_state + self.update(context))
        latent = torch.nn.functional.silu(
            self.retention_down(
                self.retention_norm(torch.cat([edge_state, proposal], dim=-1))
            )
        )
        correction = self.retention_value(latent) * torch.sigmoid(
            self.retention_gate(latent)
        )
        return proposal + correction


class OGBRetentionGraphStateWrapper(
    OGBLocalGlobalGeometrySparseTriangleEdgeStateGPSWrapper
):
    """GraphState9 with a learned gated correction on persistent bond memory."""

    def __init__(self, *args, retention_rank: int = 32, **kwargs) -> None:
        super().__init__(*args, global_mode="graph_state", **kwargs)
        self.edge_updates = nn.ModuleList(
            _GatedPersistentEdgeUpdate(
                update,
                self.edge_state_channels,
                retention_rank,
            )
            for update in self.edge_updates
        )


def make_local_statistics_encoder(candidate: str):
    common = {
        "in_channels": 9,
        "edge_dim": 3,
        "hidden_channels": 192,
        "num_layers": 9,
        "num_heads": 4,
        "dropout": 0.1,
        "n_targets": 1,
        "pooling": "mean",
        "rwse_dim": 16,
        "edge_state_channels": 64,
        "wedge_channels": 16,
        "geometry_basis_channels": 16,
        "graph_state_channels": 64,
        "graph_exchange_rank": 32,
    }
    if candidate == PNA_CANDIDATE:
        return OGBPNAStatisticsGraphStateWrapper(**common, pna_rank=64)
    if candidate == RETENTION_CANDIDATE:
        return OGBRetentionGraphStateWrapper(**common, retention_rank=32)
    raise ValueError(f"Unknown local-statistics candidate: {candidate}")
