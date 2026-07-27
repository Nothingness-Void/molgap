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
                 num_layers=6, num_heads=8, dropout=0.1, n_targets=3,
                 pooling="mean"):
        super().__init__()
        from torch_geometric.nn import GPSConv, GINEConv

        if pooling not in {"mean", "mean_max"}:
            raise ValueError(f"Unsupported GPS pooling: {pooling}")
        self.pooling = pooling

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

        if pooling == "mean_max":
            self.pool_proj = nn.Linear(hidden_channels * 2, hidden_channels)
            # Start exactly at mean pooling; training can add max-pooled signal.
            with torch.no_grad():
                self.pool_proj.weight.zero_()
                self.pool_proj.weight[:, :hidden_channels].copy_(torch.eye(hidden_channels))
                self.pool_proj.bias.zero_()

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

    def _pool(self, h, batch):
        from torch_geometric.nn import global_max_pool, global_mean_pool

        mean = global_mean_pool(h, batch)
        if self.pooling == "mean":
            return mean
        maximum = global_max_pool(h, batch)
        return self.pool_proj(torch.cat([mean, maximum], dim=-1))

    def encode(self, x, edge_index, edge_attr, batch):
        """Return molecule-level embeddings [num_molecules, hidden_channels]."""
        h = self.node_emb(x.float())
        e = self.edge_emb(edge_attr.float())

        for conv in self.convs:
            h = conv(h, edge_index, batch, edge_attr=e)

        return self._pool(h, batch)

    def encode_layers(self, x, edge_index, edge_attr, batch, layers=(2, 4, -1)):
        """Return concatenated pooled embeddings from selected GPS layers.

        Layer indices are 1-based after each GPSConv. ``-1`` means the final
        layer. This supports lightweight layer-fusion probes without changing
        the normal production ``encode`` path.
        """
        n_layers = len(self.convs)
        wanted = {n_layers if layer == -1 else int(layer) for layer in layers}
        invalid = [layer for layer in wanted if layer < 1 or layer > n_layers]
        if invalid:
            raise ValueError(f"GPS layer index out of range: {invalid}")

        h = self.node_emb(x.float())
        e = self.edge_emb(edge_attr.float())
        pooled = []
        for i, conv in enumerate(self.convs, start=1):
            h = conv(h, edge_index, batch, edge_attr=e)
            if i in wanted:
                pooled.append(self._pool(h, batch))
        return torch.cat(pooled, dim=-1)
