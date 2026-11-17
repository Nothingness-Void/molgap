"""Richer single-A100 TGT/EGT hybrid for the QM9 Route-C screen.

This candidate composes the best bounded ingredients measured so far: a
pure-2D persistent pair/triplet encoder with multi-scale walk descriptors and
an ETKDG pair-state encoder with explicit bond channels.  The two views are
fused only at molecule level, so the topology view remains aligned to the
heavy-atom graph while the geometry view remains aligned to explicit-H
coordinates.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .pair_triplet_2d_rich import PairTriplet2DRichWrapper
from .tgt_egt_hybrid import (
    _GeometryPairEncoder,
    _offset_edges,
    _counts_to_batch,
)


class TGTEGTRichWrapper(nn.Module):
    """Rich 2D pair/triplet encoder fused with ETKDG pair geometry."""

    def __init__(
        self,
        in_channels: int = 11,
        edge_dim: int = 4,
        hidden_channels: int = 192,
        pair_channels: int = 64,
        num_layers: int = 8,
        num_heads: int = 4,
        num_rbf: int = 64,
        cutoff: float = 12.0,
        dropout: float = 0.05,
        topology_layers: int = 8,
        topology_hidden_channels: int = 256,
        topology_pair_channels: int = 96,
        topology_heads: int = 8,
        topology_path_steps: int = 5,
        topology_triplet_rank: int = 16,
        n_targets: int = 3,
        learning_rate: float | None = None,
        amp: bool | None = None,
    ):
        super().__init__()
        del learning_rate, amp  # training-only config fields
        self.hidden_channels = int(hidden_channels)
        self.edge_dim = int(edge_dim)

        self.topology = PairTriplet2DRichWrapper(
            in_channels=in_channels,
            edge_dim=edge_dim,
            hidden_channels=topology_hidden_channels,
            pair_channels=topology_pair_channels,
            num_layers=topology_layers,
            num_heads=topology_heads,
            dropout=dropout,
            pooling="mean_max",
            path_steps=topology_path_steps,
            triplet_rank=topology_triplet_rank,
        )
        # The composed encoder returns its embedding; its prediction head is
        # not part of this model and must not create unused trainable weights.
        self.topology.head = nn.Identity()
        self.topology_projection = nn.Sequential(
            nn.LayerNorm(topology_hidden_channels),
            nn.Linear(topology_hidden_channels, hidden_channels),
            nn.SiLU(),
        )
        self.geometry = _GeometryPairEncoder(
            hidden_channels,
            pair_channels,
            num_heads,
            num_layers,
            num_rbf,
            cutoff,
            edge_dim,
            dropout,
        )
        self.fusion_gate = nn.Sequential(
            nn.LayerNorm(hidden_channels * 2),
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.Sigmoid(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.SiLU(),
            nn.Linear(hidden_channels // 2, n_targets),
        )

    def _topology_encode(
        self,
        topology_x,
        topology_edges,
        topology_edge_attr,
        topology_node_counts,
        topology_edge_counts,
    ):
        topology_batch = _counts_to_batch(topology_node_counts)
        global_edges = _offset_edges(
            topology_edges,
            topology_node_counts,
            topology_edge_counts,
        )
        edge_index = global_edges.contiguous()
        embedding = self.topology.encode(
            topology_x.float(),
            edge_index,
            topology_edge_attr.float(),
            topology_batch,
        )
        return self.topology_projection(embedding)

    def encode(
        self,
        x,
        edge_index,
        edge_attr,
        z,
        pos,
        batch,
        *,
        topology_x=None,
        topology_edges=None,
        topology_edge_attr=None,
        topology_node_counts=None,
        topology_edge_counts=None,
        geometry_node_counts=None,
    ):
        if topology_x is None:
            topology_x = x
        if topology_edges is None:
            topology_edges = edge_index.t().contiguous()
        if topology_edge_attr is None:
            topology_edge_attr = edge_attr
        if topology_node_counts is None:
            topology_node_counts = torch.bincount(batch)
        if topology_edge_counts is None:
            topology_edge_counts = torch.bincount(
                torch.zeros(
                    topology_edges.shape[0],
                    dtype=torch.long,
                    device=topology_edges.device,
                ),
                minlength=topology_node_counts.numel(),
            )
        if geometry_node_counts is None:
            geometry_node_counts = torch.bincount(batch)
        geometry_batch = _counts_to_batch(geometry_node_counts)

        topology = self._topology_encode(
            topology_x,
            topology_edges,
            topology_edge_attr,
            topology_node_counts,
            topology_edge_counts,
        )
        geometry = self.geometry.encode(
            z,
            pos,
            geometry_batch,
            topology_edges=topology_edges,
            topology_edge_attr=topology_edge_attr,
            topology_node_counts=topology_node_counts,
            topology_edge_counts=topology_edge_counts,
        )
        gate = self.fusion_gate(torch.cat((topology, geometry), dim=-1))
        return gate * topology + (1.0 - gate) * geometry

    def forward(self, x, edge_index, edge_attr, z, pos, batch, **kwargs):
        return self.head(
            self.encode(x, edge_index, edge_attr, z, pos, batch, **kwargs)
        )
