"""A robust TGT-lite plus edge-aware global-attention hybrid."""
from __future__ import annotations

import torch
import torch.nn as nn

from .tgt_lite import TGTLiteWrapper
from .topology_attention import TopologyAttentionWrapper


class TGTLiteHybridV2Wrapper(nn.Module):
    """GINE-free topology attention plus ETKDG TGT-lite geometry."""

    def __init__(
        self,
        in_channels: int = 11,
        edge_dim: int = 4,
        hidden_channels: int = 192,
        pair_channels: int = 48,
        num_layers: int = 8,
        num_heads: int = 4,
        num_rbf: int = 32,
        cutoff: float = 12.0,
        dropout: float = 0.05,
        topology_layers: int = 6,
        n_targets: int = 3,
    ):
        super().__init__()
        self.topology = TopologyAttentionWrapper(
            in_channels=in_channels,
            edge_dim=edge_dim,
            hidden_channels=hidden_channels,
            pair_channels=pair_channels,
            num_layers=topology_layers,
            num_heads=num_heads,
            dropout=dropout,
            n_targets=n_targets,
        )
        self.geometry = TGTLiteWrapper(
            in_channels=in_channels,
            edge_dim=edge_dim,
            hidden_channels=hidden_channels,
            pair_channels=pair_channels,
            num_layers=num_layers,
            num_heads=num_heads,
            num_rbf=num_rbf,
            cutoff=cutoff,
            dropout=dropout,
            n_targets=n_targets,
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(hidden_channels * 2),
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.SiLU(),
            nn.Linear(hidden_channels // 2, n_targets),
        )

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
        topology = self.topology.encode(
            topology_x if topology_x is not None else x,
            edge_index,
            edge_attr,
            batch,
            topology_edges=topology_edges,
            topology_edge_attr=topology_edge_attr,
            topology_node_counts=topology_node_counts,
            topology_edge_counts=topology_edge_counts,
        )
        geometry_batch = batch
        if geometry_node_counts is not None:
            geometry_counts = geometry_node_counts.view(-1).long()
            if int(geometry_counts.sum()) != int(z.shape[0]):
                raise ValueError("geometry node counts do not match z")
            geometry_batch = torch.repeat_interleave(
                torch.arange(geometry_counts.numel(), device=z.device),
                geometry_counts,
            )
        geometry = self.geometry.encode(
            x, edge_index, edge_attr, z, pos, geometry_batch
        )
        return self.fusion(torch.cat((topology, geometry), dim=-1))

    def forward(self, x, edge_index, edge_attr, z, pos, batch, **kwargs):
        return self.head(self.encode(x, edge_index, edge_attr, z, pos, batch, **kwargs))
