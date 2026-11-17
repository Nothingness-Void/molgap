"""End-to-end pure-2D GPS with dynamic bond states.

The model keeps the same atom/bond graph contract as :class:`GPSWrapper`, but
does not treat bond embeddings as a fixed side input. At every block it
rebuilds a symmetric bond state from the current endpoint states and feeds
that state into the next local/global GPS block. The prediction head is a
direct three-target regressor; no checkpoint, prediction, or target residual
from another model is used.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import GPSConv, GINEConv
from torch_geometric.nn import global_add_pool, global_mean_pool
from torch_geometric.utils import degree, softmax


class DynamicEdgeGPSWrapper(nn.Module):
    """GPS backbone with learned topology-dependent bond states."""

    def __init__(
        self,
        in_channels: int = 9,
        edge_dim: int = 4,
        hidden_channels: int = 160,
        num_layers: int = 11,
        num_heads: int = 4,
        dropout: float = 0.05,
        n_targets: int = 3,
        max_degree: int = 8,
    ) -> None:
        super().__init__()
        if hidden_channels % num_heads:
            raise ValueError("hidden_channels must be divisible by num_heads")
        if num_layers < 1:
            raise ValueError("num_layers must be positive")

        self.hidden_channels = hidden_channels
        self.max_degree = max_degree
        self.node_emb = nn.Linear(in_channels, hidden_channels)
        self.edge_emb = nn.Linear(edge_dim, hidden_channels)
        self.degree_emb = nn.Embedding(max_degree + 1, hidden_channels)

        # A bottleneck keeps the parameter budget close to GPS11-160 while
        # allowing every layer to recompute bond context from node states.
        edge_bottleneck = max(hidden_channels // 2, 32)
        self.edge_updates = nn.ModuleList()
        self.edge_norms = nn.ModuleList()
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.edge_updates.append(
                nn.Sequential(
                    nn.Linear(hidden_channels * 3, edge_bottleneck),
                    nn.SiLU(),
                    nn.Linear(edge_bottleneck, hidden_channels),
                    nn.Dropout(dropout),
                )
            )
            self.edge_norms.append(nn.LayerNorm(hidden_channels))
            gin = GINEConv(
                nn.Sequential(
                    nn.Linear(hidden_channels, hidden_channels),
                    nn.SiLU(),
                    nn.Linear(hidden_channels, hidden_channels),
                ),
                edge_dim=hidden_channels,
            )
            self.convs.append(
                GPSConv(
                    channels=hidden_channels,
                    conv=gin,
                    heads=num_heads,
                    dropout=dropout,
                    act="silu",
                    norm="batch_norm",
                    attn_type="multihead",
                )
            )

        # Learn where the orbital signal is concentrated within a molecule.
        # The zero initialization makes the first forward pass exactly mean
        # pooling, then lets training depart from it when useful.
        self.pool_gate = nn.Sequential(
            nn.Linear(hidden_channels, max(hidden_channels // 2, 32)),
            nn.SiLU(),
            nn.Linear(max(hidden_channels // 2, 32), 1),
        )
        nn.init.zeros_(self.pool_gate[-1].weight)
        nn.init.zeros_(self.pool_gate[-1].bias)
        self.pool_proj = nn.Linear(hidden_channels * 2, hidden_channels)
        with torch.no_grad():
            self.pool_proj.weight.zero_()
            self.pool_proj.weight[:, :hidden_channels].copy_(
                torch.eye(hidden_channels)
            )
            self.pool_proj.bias.zero_()

        self.head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.SiLU(),
            nn.Linear(hidden_channels // 2, n_targets),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        return self.head(self.encode(x, edge_index, edge_attr, batch))

    def _pool(self, h: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        mean = global_mean_pool(h, batch)
        scores = self.pool_gate(h)
        weights = softmax(scores, batch)
        attended = global_add_pool(h * weights, batch)
        return self.pool_proj(torch.cat([mean, attended], dim=-1))

    def encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """Return molecule-level embeddings with the direct 2D encoder path."""
        h = self.node_emb(x.float())
        node_degree = degree(
            edge_index[0], num_nodes=x.size(0), dtype=torch.long
        ).clamp_max(self.max_degree)
        h = h + self.degree_emb(node_degree)
        edge_state = self.edge_emb(edge_attr.float())
        source, target = edge_index

        for edge_update, edge_norm, conv in zip(
            self.edge_updates, self.edge_norms, self.convs
        ):
            endpoint_sum = h[source] + h[target]
            endpoint_difference = (h[source] - h[target]).abs()
            context = torch.cat(
                [edge_state, endpoint_sum, endpoint_difference], dim=-1
            )
            edge_state = edge_norm(edge_update(context))
            h = conv(h, edge_index, batch, edge_attr=edge_state)

        return self._pool(h, batch)

