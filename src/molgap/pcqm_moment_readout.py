"""Compact projected-moment readout for the frozen PCQM GraphState encoder."""
from __future__ import annotations

import torch
import torch.nn as nn

from .pcqm_gap_architecture import (
    OGBLocalGlobalGeometrySparseTriangleEdgeStateGPSWrapper,
)


BASELINE_CANDIDATE_ID = (
    "ogb_distance_angle_triangle_edge_state_graph_state9"
)
CANDIDATE_ID = (
    "ogb_distance_angle_triangle_edge_state_graph_state9_moment_readout32"
)
BASELINE_PARAMETER_COUNT = 3_665_809
MOMENT_CHANNELS = 32
MOMENT_PARAMETER_DELTA = (
    2 * 192                 # node LayerNorm
    + 192 * MOMENT_CHANNELS  # bias-free nonlinear projection
    + 2 * (2 * MOMENT_CHANNELS)  # moment LayerNorm
    + (2 * MOMENT_CHANNELS) * 192  # zero-initialized return
)
MOMENT_PARAMETER_COUNT = BASELINE_PARAMETER_COUNT + MOMENT_PARAMETER_DELTA
PARAMETER_BUDGET = 4_000_000

COMMON_KWARGS = {
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


class OGBProjectedMomentReadoutGraphStateWrapper(
    OGBLocalGlobalGeometrySparseTriangleEdgeStateGPSWrapper
):
    """GraphState9 with one compact final atom-distribution readout.

    The complete encoder and its internal GraphState summaries are unchanged.
    Only the final graph embedding receives nonlinear projected first and
    centered-second moments. The return is zero initialized so the candidate
    begins exactly at the mean-readout control.
    """

    def __init__(self, *args, moment_channels: int = MOMENT_CHANNELS, **kwargs):
        if int(moment_channels) != MOMENT_CHANNELS:
            raise ValueError("the bounded candidate requires moment_channels=32")
        super().__init__(*args, global_mode="graph_state", **kwargs)
        hidden_channels = self.head[0].in_features
        self.moment_channels = MOMENT_CHANNELS
        self.moment_node_norm = nn.LayerNorm(hidden_channels)
        self.moment_projection = nn.Linear(
            hidden_channels, self.moment_channels, bias=False
        )
        self.moment_norm = nn.LayerNorm(2 * self.moment_channels)
        self.moment_return = nn.Linear(
            2 * self.moment_channels, hidden_channels, bias=False
        )
        nn.init.zeros_(self.moment_return.weight)

    def _pool(self, h: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        from torch_geometric.nn import global_mean_pool

        mean_embedding = global_mean_pool(h, batch)
        projected = torch.nn.functional.silu(
            self.moment_projection(self.moment_node_norm(h))
        )
        first = global_mean_pool(projected, batch)
        raw_second = global_mean_pool(projected.square(), batch)
        centered_second = (raw_second - first.square()).clamp_min(0.0)
        moments = self.moment_norm(torch.cat([first, centered_second], dim=-1))
        return mean_embedding + self.moment_return(moments)


def make_moment_readout_encoder():
    """Build the bounded K2 candidate with frozen GraphState9 settings."""
    return OGBProjectedMomentReadoutGraphStateWrapper(
        **COMMON_KWARGS,
        moment_channels=MOMENT_CHANNELS,
    )


__all__ = [
    "BASELINE_CANDIDATE_ID",
    "BASELINE_PARAMETER_COUNT",
    "CANDIDATE_ID",
    "MOMENT_CHANNELS",
    "MOMENT_PARAMETER_COUNT",
    "PARAMETER_BUDGET",
    "OGBProjectedMomentReadoutGraphStateWrapper",
    "make_moment_readout_encoder",
]
