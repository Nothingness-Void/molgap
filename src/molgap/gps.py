"""GPS (General Powerful Scalable) Graph Transformer for 2D molecular graphs."""
from __future__ import annotations

import torch
import torch.nn as nn


class GPSWrapper(nn.Module):
    """GPS Graph Transformer operating on 2D bond-topology graphs.

    Input: PyG Data with x (atom features), edge_index (bonds),
           edge_attr (bond features), batch.
    """

    def __init__(self, in_channels=9, edge_dim=4, hidden_channels=128,
                 num_layers=6, num_heads=8, dropout=0.1, n_targets=3):
        super().__init__()
        from torch_geometric.nn import GPSConv, GINEConv

        self.node_emb = nn.Linear(in_channels, hidden_channels)
        self.edge_emb = nn.Linear(edge_dim, hidden_channels)

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            gin = GINEConv(
                nn.Sequential(
                    nn.Linear(hidden_channels, hidden_channels),
                    nn.SiLU(),
                    nn.Linear(hidden_channels, hidden_channels),
                ),
                edge_dim=hidden_channels,
            )
            gps = GPSConv(
                channels=hidden_channels,
                conv=gin,
                heads=num_heads,
                dropout=dropout,
                act="silu",
                norm="batch_norm",
                attn_type="multihead",
            )
            self.convs.append(gps)

        self.head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.SiLU(),
            nn.Linear(hidden_channels // 2, n_targets),
        )

    def forward(self, x, edge_index, edge_attr, batch):
        h = self.encode(x, edge_index, edge_attr, batch)
        return self.head(h)

    def encode(self, x, edge_index, edge_attr, batch):
        """Return molecule-level embeddings [num_molecules, hidden_channels]."""
        from torch_geometric.nn import global_mean_pool

        h = self.node_emb(x.float())
        e = self.edge_emb(edge_attr.float())

        for conv in self.convs:
            h = conv(h, edge_index, batch, edge_attr=e)

        return global_mean_pool(h, batch)
