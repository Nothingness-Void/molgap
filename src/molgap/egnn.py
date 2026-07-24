"""Lightweight E(n)-equivariant molecular graph encoder."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch_scatter import scatter_mean
from torch_geometric.nn import global_max_pool, global_mean_pool
from torch_cluster import radius_graph


class _EGNNLayer(nn.Module):
    def __init__(self, hidden_channels: int, num_rbf: int, coordinate_scale: float):
        super().__init__()
        self.coordinate_scale = coordinate_scale
        self.message = nn.Sequential(
            nn.Linear(hidden_channels * 2 + num_rbf, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
        )
        self.coordinate = nn.Linear(hidden_channels, 1, bias=False)
        self.update = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.norm = nn.LayerNorm(hidden_channels)

    def forward(self, h, pos, edge_index, radial):
        row, col = edge_index
        message = self.message(torch.cat((h[row], h[col], radial), dim=-1))
        aggregated = scatter_mean(message, row, dim=0, dim_size=h.size(0))
        h = self.norm(h + self.update(torch.cat((h, aggregated), dim=-1)))

        direction = pos[row] - pos[col]
        distance = direction.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        weight = self.coordinate(message).tanh()
        delta = scatter_mean(
            direction / distance * weight,
            row,
            dim=0,
            dim_size=pos.size(0),
        )
        return h, pos + self.coordinate_scale * delta


class EGNNWrapper(nn.Module):
    """EGNN with an invariant pooled embedding for shared fusion heads."""

    def __init__(
        self,
        hidden_channels: int = 128,
        num_layers: int = 4,
        num_rbf: int = 32,
        cutoff: float = 5.0,
        dropout: float = 0.05,
        n_targets: int = 3,
        max_num_neighbors: int = 32,
        coordinate_scale: float = 0.1,
    ):
        super().__init__()
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
        self.embedding = nn.Embedding(100, hidden_channels)
        self.register_buffer("rbf_centers", torch.linspace(0.0, cutoff, num_rbf))
        self.rbf_gamma = float(num_rbf) / cutoff
        self.layers = nn.ModuleList(
            _EGNNLayer(hidden_channels, num_rbf, coordinate_scale)
            for _ in range(num_layers)
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.SiLU(),
            nn.Linear(hidden_channels // 2, n_targets),
        )

    def encode(self, z, pos, batch):
        h = self.embedding(z)
        edge_index = radius_graph(
            pos,
            r=self.cutoff,
            batch=batch,
            max_num_neighbors=self.max_num_neighbors,
        )
        for layer in self.layers:
            row, col = edge_index
            distance = (pos[row] - pos[col]).norm(dim=-1, keepdim=True)
            radial = torch.exp(
                -self.rbf_gamma * (distance - self.rbf_centers.view(1, -1)).square()
            )
            h, pos = layer(h, pos, edge_index, radial)
        return torch.cat(
            (global_mean_pool(h, batch), global_max_pool(h, batch)),
            dim=-1,
        )

    def forward(self, z, pos, batch):
        return self.head(self.encode(z, pos, batch))
