"""Shared SchNet wrapper with optional Gasteiger charge injection and 2D descriptor fusion."""
from __future__ import annotations

import torch
import torch.nn as nn


class SchNetWrapper(nn.Module):
    """SchNet with multi-target output head, optional charge features, and optional 2D descriptor fusion."""

    def __init__(self, hidden_channels, num_filters, num_interactions,
                 num_gaussians, cutoff, dropout=0.1, n_targets=3,
                 use_charges=False, n_desc=0):
        super().__init__()
        from torch_geometric.nn.models import SchNet

        self.schnet = SchNet(
            hidden_channels=hidden_channels,
            num_filters=num_filters,
            num_interactions=num_interactions,
            num_gaussians=num_gaussians,
            cutoff=cutoff,
        )
        self.use_charges = use_charges
        if use_charges:
            self.charge_proj = nn.Linear(1, hidden_channels)

        self.n_desc = n_desc
        if n_desc > 0:
            self.desc_proj = nn.Sequential(
                nn.Linear(n_desc, hidden_channels // 2),
                nn.SiLU(),
                nn.Dropout(dropout),
            )
            head_in = hidden_channels + hidden_channels // 2
        else:
            head_in = hidden_channels

        self.head = nn.Sequential(
            nn.Linear(head_in, hidden_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.SiLU(),
            nn.Linear(hidden_channels // 2, n_targets),
        )

    def forward(self, z, pos, batch, charges=None, desc=None):
        return self.head(self.encode(z, pos, batch, charges=charges, desc=desc))

    def encode(self, z, pos, batch, charges=None, desc=None):
        """Pooled molecular embedding (pre-head). Used for hybrid 2D+3D fusion."""
        from torch_geometric.nn import global_mean_pool

        h = self.schnet.embedding(z)

        if self.use_charges and charges is not None:
            h = h + self.charge_proj(charges.unsqueeze(-1))

        edge_index, edge_weight = self._radius_graph(pos, batch)
        edge_attr = self.schnet.distance_expansion(edge_weight)

        for interaction in self.schnet.interactions:
            h = h + interaction(h, edge_index, edge_weight, edge_attr)

        h = global_mean_pool(h, batch)

        if self.n_desc > 0 and desc is not None:
            desc = desc.view(h.size(0), self.n_desc)
            h = torch.cat([h, self.desc_proj(desc)], dim=-1)

        return h

    def _radius_graph(self, pos, batch):
        from torch_geometric.nn.models.schnet import radius_graph
        edge_index = radius_graph(pos, r=self.schnet.cutoff, batch=batch,
                                  max_num_neighbors=32)
        row, col = edge_index
        edge_weight = (pos[row] - pos[col]).norm(dim=-1)
        return edge_index, edge_weight
