"""Local-message-passing GINE encoder for molecular regression."""
from __future__ import annotations

import torch
import torch.nn as nn


class GINEWrapper(nn.Module):
    """GINE baseline with the same molecular readout contract as GPSWrapper."""

    def __init__(
        self,
        in_channels: int = 9,
        edge_dim: int = 4,
        hidden_channels: int = 192,
        num_layers: int = 6,
        dropout: float = 0.05,
        n_targets: int = 3,
    ) -> None:
        super().__init__()
        from torch_geometric.nn import GINEConv

        self.node_emb = nn.Linear(in_channels, hidden_channels)
        self.edge_emb = nn.Linear(edge_dim, hidden_channels)
        self.convs = nn.ModuleList([
            GINEConv(
                nn.Sequential(
                    nn.Linear(hidden_channels, hidden_channels),
                    nn.SiLU(),
                    nn.Linear(hidden_channels, hidden_channels),
                ),
                edge_dim=hidden_channels,
            )
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_channels) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.pool_proj = nn.Linear(hidden_channels * 2, hidden_channels)
        self.head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.SiLU(),
            nn.Linear(hidden_channels // 2, n_targets),
        )

    def encode(self, x, edge_index, edge_attr, batch):
        from torch_geometric.nn import global_max_pool, global_mean_pool

        h = self.node_emb(x.float())
        edge = self.edge_emb(edge_attr.float())
        for conv, norm in zip(self.convs, self.norms):
            update = conv(h, edge_index, edge_attr=edge)
            h = h + self.dropout(norm(update))
        pooled = torch.cat([
            global_mean_pool(h, batch),
            global_max_pool(h, batch),
        ], dim=-1)
        return self.pool_proj(pooled)

    def forward(self, x, edge_index, edge_attr, batch):
        return self.head(self.encode(x, edge_index, edge_attr, batch))
