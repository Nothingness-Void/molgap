"""Bounded PCQM vector-state GraphState candidate.

The candidate keeps the accepted OGB distance/angle GraphState9 encoder intact
and adds one shared, node-local polar-vector state cell.  Coordinates enter
only as directed real-bond displacements; no relation or target is added.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .pcqm_gap_architecture import (
    OGBLocalGlobalGeometrySparseTriangleEdgeStateGPSWrapper,
)


BASELINE_CANDIDATE_ID = "ogb_distance_angle_triangle_edge_state_graph_state9"
CANDIDATE_ID = "ogb_distance_angle_vector_state_triangle_edge_state_graph_state9"

VECTOR_CHANNELS = 16
VECTOR_UPDATE_BLOCKS = (2, 4, 6, 8)

# The parent count is frozen by the accepted GraphState9 record.  The added
# arithmetic mirrors _PolarVectorCell and scalar_return exactly, without
# needing to instantiate torch modules on the coordinator host.
BASELINE_PARAMETER_COUNT = 3_665_809
VECTOR_STATE_PARAMETER_DELTA = (
    2 * (2 * 192 + 64 + 16)       # context_norm: LayerNorm(464)
    + 2 * (2 * 192 + 64 + 16) * 16  # edge coeff/gate: 464 -> 16, bias-free
    + 2 * (192 + 64 + 16)          # node_norm: LayerNorm(272)
    + (192 + 64 + 16) * 16         # node gate: 272 -> 16, bias-free
    + 2 * 16 * 16                  # two bias-free channel-linear transforms
    + (3 * 16) * 192               # zero-initialized scalar return: 48 -> 192
)
VECTOR_STATE_PARAMETER_COUNT = (
    BASELINE_PARAMETER_COUNT + VECTOR_STATE_PARAMETER_DELTA
)
PARAMETER_BUDGET = 4_000_000

VECTOR_STATE_KWARGS = {
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


class _PolarVectorCell(nn.Module):
    """One shared scalar-coefficient/vector-state update cell.

    All learned maps that act on vector channels are bias-free and operate on
    the channel axis only.  The coordinate axis is touched only by scalar
    multiplication, sums, squared norms, and dot products, so the state is
    covariant under every orthogonal transformation, including reflections.
    """

    EPS = 1.0e-8

    def __init__(
        self,
        atom_channels: int,
        edge_channels: int,
        vector_channels: int,
    ) -> None:
        super().__init__()
        self.vector_channels = int(vector_channels)
        context_channels = 2 * int(atom_channels) + int(edge_channels) + 16
        node_context_channels = int(atom_channels) + int(edge_channels) + 16

        # These are scalar networks: their outputs are coefficients/gates,
        # never Cartesian vectors.
        self.context_norm = nn.LayerNorm(context_channels)
        self.edge_coeff = nn.Linear(
            context_channels, self.vector_channels, bias=False
        )
        self.edge_gate = nn.Linear(
            context_channels, self.vector_channels, bias=False
        )
        self.node_norm = nn.LayerNorm(node_context_channels)
        self.node_gate = nn.Linear(
            node_context_channels, self.vector_channels, bias=False
        )

        # These transforms mix vector channels independently for each of the
        # three Cartesian components; a bias would create a non-covariant
        # constant vector contribution.
        self.message_mix = nn.Linear(
            self.vector_channels, self.vector_channels, bias=False
        )
        self.state_mix = nn.Linear(
            self.vector_channels, self.vector_channels, bias=False
        )

    @staticmethod
    def _mean_to_nodes(
        values: torch.Tensor,
        destinations: torch.Tensor,
        weights: torch.Tensor,
        node_count: int,
    ) -> torch.Tensor:
        """Scatter a masked directed-edge scalar payload to destination nodes."""
        result = values.new_zeros((node_count, values.shape[-1]))
        counts = values.new_zeros((node_count, 1))
        if values.shape[0]:
            result.index_add_(0, destinations, values * weights)
            counts.index_add_(0, destinations, weights)
        return result / counts.clamp_min_(1.0)

    @staticmethod
    def _channel_linear(
        transform: nn.Linear, values: torch.Tensor
    ) -> torch.Tensor:
        """Apply a channel-only linear map to ``[N, C, 3]`` values."""
        return transform(values.transpose(1, 2)).transpose(1, 2)

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_state: torch.Tensor,
        vector_state: torch.Tensor,
        positions: torch.Tensor,
        node_mask: torch.Tensor,
        distance_basis: nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        source, destination = edge_index
        displacement = positions[destination] - positions[source]
        edge_mask = node_mask[source] * node_mask[destination]

        # No division by distance is used.  Thus coincident endpoints remain
        # finite and still provide a well-defined zero polar displacement.
        squared_distance = displacement.square().sum(dim=-1, keepdim=True)
        distance = torch.sqrt(squared_distance + self.EPS)
        distance_features = distance_basis(distance) * edge_mask

        edge_context = torch.cat(
            [h[source], h[destination], edge_state, distance_features], dim=-1
        )
        edge_context = self.context_norm(edge_context)
        coefficient = torch.tanh(self.edge_coeff(edge_context))
        edge_gate = torch.sigmoid(self.edge_gate(edge_context))
        edge_vector = (
            coefficient.unsqueeze(-1)
            * edge_gate.unsqueeze(-1)
            * displacement.unsqueeze(1)
        )
        edge_vector = self._channel_linear(self.message_mix, edge_vector)
        edge_vector = edge_vector * edge_mask.view(-1, 1, 1)

        node_edge_context = self._mean_to_nodes(
            torch.cat([edge_state, distance_features], dim=-1),
            destination,
            edge_mask,
            h.shape[0],
        )
        node_context = self.node_norm(
            torch.cat([h, node_edge_context], dim=-1)
        )
        node_gate = torch.sigmoid(self.node_gate(node_context))

        message = vector_state.new_zeros(vector_state.shape)
        counts = vector_state.new_zeros((h.shape[0], 1))
        if edge_vector.shape[0]:
            message.index_add_(0, destination, edge_vector)
            counts.index_add_(0, destination, edge_mask)
        message = message / counts.clamp_min_(1.0).unsqueeze(-1)

        state_component = self._channel_linear(self.state_mix, vector_state)
        proposal = state_component + message
        proposal_rms = torch.sqrt(
            proposal.square().mean(dim=(1, 2), keepdim=True) + self.EPS
        )
        updated = vector_state + node_gate.unsqueeze(-1) * (
            proposal / (1.0 + proposal_rms)
        )
        updated = updated * node_mask.view(-1, 1, 1)

        # These are the only vector-to-scalar observations.  Subtracting the
        # zero-state value keeps edgeless graphs exactly at zero while the
        # epsilon keeps the derivative finite at a zero vector.
        zero_norm = self.EPS**0.5
        old_norm = torch.sqrt(
            vector_state.square().sum(dim=-1) + self.EPS
        ) - zero_norm
        new_norm = torch.sqrt(updated.square().sum(dim=-1) + self.EPS) - zero_norm
        dot_product = (vector_state * updated).sum(dim=-1)
        invariants = torch.cat([old_norm, new_norm, dot_product], dim=-1)
        invariants = invariants * node_mask
        return updated, invariants


class OGBVectorStateGraphStateWrapper(
    OGBLocalGlobalGeometrySparseTriangleEdgeStateGPSWrapper
):
    """GraphState9 plus one shared persistent polar vector state."""

    VECTOR_BLOCKS = VECTOR_UPDATE_BLOCKS

    def __init__(
        self,
        *args,
        vector_channels: int = VECTOR_CHANNELS,
        **kwargs,
    ) -> None:
        if int(vector_channels) != VECTOR_CHANNELS:
            raise ValueError("the bounded candidate requires vector_channels=16")
        # Construct the complete accepted parent first.  This preserves every
        # shared initialization draw under seed 42 before new modules exist.
        super().__init__(*args, global_mode="graph_state", **kwargs)
        hidden_channels = self.head[0].in_features
        self.vector_channels = VECTOR_CHANNELS
        self.vector_cell = _PolarVectorCell(
            hidden_channels,
            self.edge_state_channels,
            self.vector_channels,
        )
        self.scalar_return = nn.Linear(
            3 * self.vector_channels, hidden_channels, bias=False
        )
        nn.init.zeros_(self.scalar_return.weight)

    @staticmethod
    def _sanitize_geometry_valid(
        geometry_valid: torch.Tensor, batch: torch.Tensor
    ) -> torch.Tensor:
        if geometry_valid is None:
            raise ValueError("geometry_valid is required")
        if batch.ndim != 1:
            raise ValueError("batch must have shape [N]")
        valid = geometry_valid.reshape(-1).to(
            device=batch.device, dtype=torch.float32
        )
        if batch.numel() and int(batch.min()) < 0:
            raise ValueError("batch contains a negative graph id")
        if batch.numel() and (
            valid.numel() == 0 or int(batch.max()) >= valid.shape[0]
        ):
            raise ValueError("geometry_valid does not cover the batched graphs")
        finite = torch.isfinite(valid)
        return (finite & (valid > 0)).to(dtype=valid.dtype)

    @staticmethod
    def _sanitize_geometry_values(value: torch.Tensor) -> torch.Tensor:
        if value is None:
            raise ValueError("geometry scalar payload is required")
        return torch.where(torch.isfinite(value), value, torch.zeros_like(value))

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
        pos,
    ):
        embedding = self.encode(
            x,
            edge_index,
            edge_attr,
            batch,
            random_walk_pe,
            wedge_edge_ids,
            edge_distance,
            wedge_angle_cos,
            geometry_valid,
            pos,
        )
        return self.head(embedding)

    def encode(
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
        pos,
    ):
        geometry_valid = self._sanitize_geometry_valid(geometry_valid, batch)
        edge_distance = self._sanitize_geometry_values(edge_distance)
        wedge_angle_cos = self._sanitize_geometry_values(wedge_angle_cos)
        # Keep this two-item payload explicit: the parent owns all graph and
        # geometry ordering, while the candidate adds only its vector state.
        return self._encode_geometry(
            x,
            edge_index,
            edge_attr,
            batch,
            random_walk_pe,
            wedge_edge_ids,
            edge_distance,
            wedge_angle_cos,
            geometry_valid,
            auxiliary_payload=(pos, geometry_valid),
            pos=pos,
        )

    def _initialize_geometry_auxiliary(
        self,
        h: torch.Tensor,
        batch: torch.Tensor,
        auxiliary_payload,
    ):
        if (
            not isinstance(auxiliary_payload, tuple)
            or len(auxiliary_payload) != 2
        ):
            raise ValueError("vector-state payload must be (pos, geometry_valid)")
        pos, geometry_valid = auxiliary_payload
        if pos is None or pos.ndim != 2 or tuple(pos.shape) != (h.shape[0], 3):
            raise ValueError("pos must align to batched atom coordinates")

        positions = pos.to(device=h.device, dtype=h.dtype)
        finite_nodes = torch.isfinite(positions).all(dim=1, keepdim=True)
        safe_positions = torch.where(
            finite_nodes, positions, torch.zeros_like(positions)
        )
        valid = geometry_valid.reshape(-1).to(device=h.device, dtype=h.dtype)
        node_mask = valid[batch].view(-1, 1) * finite_nodes.to(dtype=h.dtype)

        # The parent is deliberately called with None because its GraphState
        # hook receives a tensor, not this candidate's auxiliary dictionary.
        graph_state = super()._initialize_geometry_auxiliary(h, batch, None)
        return {
            "vectors": h.new_zeros((h.shape[0], self.vector_channels, 3)),
            "graph_state": graph_state,
            "positions": safe_positions,
            "node_mask": node_mask,
        }

    def _update_geometry_auxiliary(
        self,
        layer: int,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_state: torch.Tensor,
        batch: torch.Tensor,
        auxiliary_state,
    ):
        if not isinstance(auxiliary_state, dict):
            raise ValueError("vector-state auxiliary state must be a dictionary")

        # Keep the parent's GraphState update first.  In particular, block 6
        # has the same graph-context ordering as the accepted baseline.
        h, edge_state, graph_state = super()._update_geometry_auxiliary(
            layer,
            h,
            edge_index,
            edge_state,
            batch,
            auxiliary_state["graph_state"],
        )
        auxiliary_state["graph_state"] = graph_state

        if layer + 1 in self.VECTOR_BLOCKS:
            vectors, invariants = self.vector_cell(
                h,
                edge_index,
                edge_state,
                auxiliary_state["vectors"],
                auxiliary_state["positions"],
                auxiliary_state["node_mask"],
                self.distance_basis,
            )
            auxiliary_state["vectors"] = vectors
            h = h + self.scalar_return(invariants) * auxiliary_state["node_mask"]
        return h, edge_state, auxiliary_state


def make_vector_state_encoder():
    """Build the bounded vector-state candidate with the frozen screen kwargs."""
    return OGBVectorStateGraphStateWrapper(**VECTOR_STATE_KWARGS)


__all__ = [
    "BASELINE_CANDIDATE_ID",
    "BASELINE_PARAMETER_COUNT",
    "CANDIDATE_ID",
    "PARAMETER_BUDGET",
    "VECTOR_STATE_KWARGS",
    "VECTOR_STATE_PARAMETER_COUNT",
    "VECTOR_STATE_PARAMETER_DELTA",
    "OGBVectorStateGraphStateWrapper",
    "make_vector_state_encoder",
]
