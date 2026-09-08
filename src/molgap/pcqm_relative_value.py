"""Sparse relative-value path communication for the PCQM GraphState anchor."""
from __future__ import annotations

import math

import torch
from torch import nn

from .pcqm_gap_architecture import (
    OGBLocalGlobalGeometrySparseTriangleEdgeStateGPSWrapper,
)
from .pcqm_hop_path import PATH_BLOCKS, PATH_FEATURE_DIM


BASELINE = "ogb_distance_angle_triangle_edge_state_graph_state9"
RELATIVE_VALUE_CANDIDATE = (
    "ogb_distance_angle_relative_value_triangle_edge_state_graph_state9"
)


class _SparseRelativeValuePathAttention(nn.Module):
    """Attend over exact 2/3-hop relations with path-conditioned keys/values.

    The earlier hop-path screen averaged every relation.  This block instead
    lets each target atom select incoming paths and, following GRPE's useful
    ablation, places path context in the value representation as well as the
    attention score.  A relation-count input lets the return gate suppress
    unreliable updates on sparsely supported atoms.
    """

    def __init__(
        self,
        atom_channels: int,
        path_channels: int = 32,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if path_channels <= 0 or path_channels % num_heads:
            raise ValueError("path_channels must be positive and divisible by heads")
        self.path_channels = int(path_channels)
        self.num_heads = int(num_heads)
        self.head_channels = self.path_channels // self.num_heads
        self.dropout = nn.Dropout(float(dropout))

        self.atom_norm = nn.LayerNorm(atom_channels)
        self.path_norm = nn.LayerNorm(PATH_FEATURE_DIM)
        self.query = nn.Linear(atom_channels, self.path_channels, bias=False)
        self.key = nn.Linear(atom_channels, self.path_channels, bias=False)
        self.value = nn.Linear(atom_channels, self.path_channels, bias=False)
        self.path_key = nn.Linear(PATH_FEATURE_DIM, self.path_channels, bias=False)
        self.path_value = nn.Linear(
            PATH_FEATURE_DIM, self.path_channels, bias=False
        )
        self.path_bias = nn.Linear(PATH_FEATURE_DIM, self.num_heads, bias=False)
        self.aggregate_norm = nn.LayerNorm(self.path_channels)
        self.output_value = nn.Linear(self.path_channels, atom_channels)
        self.output_gate = nn.Linear(
            atom_channels + self.path_channels + 1,
            atom_channels,
        )
        nn.init.zeros_(self.output_value.weight)
        nn.init.zeros_(self.output_value.bias)

    def forward(
        self,
        h: torch.Tensor,
        hop_path_edge_index: torch.Tensor,
        hop_path_features: torch.Tensor,
    ) -> torch.Tensor:
        from torch_geometric.utils import softmax

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
        atom = self.atom_norm(h)
        path = self.path_norm(hop_path_features.float())
        shape = (-1, self.num_heads, self.head_channels)
        query = self.query(atom[target]).view(*shape)
        key = (
            self.key(atom[source]) + self.path_key(path)
        ).view(*shape)
        value = (
            self.value(atom[source]) + self.path_value(path)
        ).view(*shape)
        score = (query * key).sum(dim=-1) / math.sqrt(self.head_channels)
        score = score + self.path_bias(path)
        weight = softmax(score, target, num_nodes=h.shape[0])
        message = self.dropout(weight).unsqueeze(-1) * value

        aggregate = h.new_zeros(
            (h.shape[0], self.num_heads, self.head_channels)
        )
        aggregate.index_add_(0, target, message)
        aggregate = self.aggregate_norm(aggregate.flatten(start_dim=1))

        count = h.new_zeros((h.shape[0], 1))
        count.index_add_(0, target, h.new_ones((target.shape[0], 1)))
        support = torch.log1p(count)
        gate = torch.sigmoid(
            self.output_gate(torch.cat([atom, aggregate, support], dim=-1))
        )
        return h + gate * self.output_value(aggregate)


class OGBRelativeValuePathGraphStateWrapper(
    OGBLocalGlobalGeometrySparseTriangleEdgeStateGPSWrapper
):
    """GraphState9 with one shared sparse relative-value path attention."""

    def __init__(
        self,
        *args,
        path_channels: int = 32,
        path_heads: int = 4,
        **kwargs,
    ) -> None:
        dropout = float(kwargs.get("dropout", 0.1))
        super().__init__(*args, global_mode="graph_state", **kwargs)
        self.hop_path_mixer = _SparseRelativeValuePathAttention(
            self.head[0].in_features,
            path_channels=path_channels,
            num_heads=path_heads,
            dropout=dropout,
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
            raise ValueError("Relative-value GraphState requires path relations")
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


def make_relative_value_encoder(candidate: str):
    if candidate != RELATIVE_VALUE_CANDIDATE:
        raise ValueError(f"Unknown relative-value candidate: {candidate}")
    return OGBRelativeValuePathGraphStateWrapper(
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
        path_heads=4,
    )


__all__ = [
    "BASELINE",
    "RELATIVE_VALUE_CANDIDATE",
    "OGBRelativeValuePathGraphStateWrapper",
    "make_relative_value_encoder",
]
