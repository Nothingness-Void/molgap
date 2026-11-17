"""A topology-preserving GPS plus TGT-lite graph-level hybrid.

The TGT-lite screen deliberately used only explicit-H geometry because the
legacy cache can have a heavy-atom-only 2D view.  This wrapper keeps that
alignment-safe geometry branch, but restores the transferable 2D topology
signal through GPS and fuses the two molecule-level representations.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .gps import GPSWrapper
from .tgt_lite import TGTLiteWrapper


class TGTLiteHybridWrapper(nn.Module):
    """GPS topology encoder plus TGT-lite ETKDG encoder."""

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
        topology_layers: int = 9,
        n_targets: int = 3,
    ):
        super().__init__()
        self.topology = GPSWrapper(
            in_channels=in_channels,
            edge_dim=edge_dim,
            hidden_channels=hidden_channels,
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

    def encode(self, x, edge_index, edge_attr, z, pos, batch):
        topology = self.topology.encode(x, edge_index, edge_attr, batch)
        geometry = self.geometry.encode(x, edge_index, edge_attr, z, pos, batch)
        return self.fusion(torch.cat((topology, geometry), dim=-1))

    def forward(self, x, edge_index, edge_attr, z, pos, batch):
        return self.head(self.encode(x, edge_index, edge_attr, z, pos, batch))
