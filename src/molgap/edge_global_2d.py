"""Compact EGT/GPTrans-style pure-2D molecular encoder.

The encoder keeps a learnable state on each bond edge, updates that state from
both endpoint nodes, and exposes it as a bias to molecule-local global
attention.  Unlike the first TGT transfer screen it does not materialize a
dense pair feature tensor or require a conformer, and it avoids optional
torch-cluster/GINE CUDA kernels.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class _EdgeGlobalBlock(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        edge_channels: int,
        edge_dim: int,
        num_heads: int,
        dropout: float,
    ):
        super().__init__()
        if hidden_channels % num_heads:
            raise ValueError("hidden_channels must be divisible by num_heads")
        self.num_heads = int(num_heads)
        self.head_channels = hidden_channels // num_heads
        self.node_norm = nn.LayerNorm(hidden_channels)
        self.qkv = nn.Linear(hidden_channels, hidden_channels * 3)
        self.attn_out = nn.Linear(hidden_channels, hidden_channels)
        self.edge_bias = nn.Linear(edge_channels, num_heads)
        self.edge_norm = nn.LayerNorm(edge_channels)
        self.edge_input = nn.Linear(edge_dim, edge_channels)
        self.edge_update = nn.Sequential(
            nn.Linear(edge_channels + hidden_channels * 2 + edge_channels, edge_channels * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(edge_channels * 2, edge_channels),
        )
        self.edge_message = nn.Linear(edge_channels, hidden_channels)
        self.local_out = nn.Linear(hidden_channels, hidden_channels)
        self.dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(hidden_channels)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels * 4, hidden_channels),
        )

    @staticmethod
    def _mean_edge_messages(
        message: torch.Tensor,
        edge_index: torch.Tensor,
        node_count: int,
    ) -> torch.Tensor:
        src, dst = edge_index
        result = message.new_zeros((node_count, message.shape[-1]))
        result.index_add_(0, src, message)
        result.index_add_(0, dst, message)
        degree = message.new_zeros((node_count, 1))
        ones = message.new_ones((message.shape[0], 1))
        degree.index_add_(0, src, ones)
        degree.index_add_(0, dst, ones)
        return result / degree.clamp_min(1.0)

    def forward(
        self,
        node: torch.Tensor,
        edge: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from torch_geometric.utils import to_dense_batch

        src, dst = edge_index
        normalized_node = self.node_norm(node)
        normalized_edge = self.edge_norm(edge)
        edge_input = self.edge_input(edge_attr.float())
        edge_delta = self.edge_update(
            torch.cat((normalized_edge, normalized_node[src], normalized_node[dst], edge_input), dim=-1)
        )
        edge = normalized_edge + self.dropout(edge_delta)

        q, k, value = self.qkv(normalized_node).chunk(3, dim=-1)
        dense_q, node_mask = to_dense_batch(q, batch)
        dense_k, _ = to_dense_batch(k, batch)
        dense_value, _ = to_dense_batch(value, batch)
        batch_size, max_nodes = dense_q.shape[:2]
        dense_q = dense_q.view(batch_size, max_nodes, self.num_heads, self.head_channels)
        dense_k = dense_k.view(batch_size, max_nodes, self.num_heads, self.head_channels)
        dense_value = dense_value.view(
            batch_size, max_nodes, self.num_heads, self.head_channels
        )
        scores = torch.einsum("bihd,bjhd->bhij", dense_q, dense_k)
        scores = scores / math.sqrt(self.head_channels)
        # Avoid torch_geometric.to_dense_adj here: the IMS vendor/shim stack
        # can enter an optional torch-cluster path that stalls even for tiny
        # graphs.  Node batches are contiguous, so local positions are cheap
        # to reconstruct and index_put_ gives the same accumulated edge bias.
        batch_size = dense_q.shape[0]
        counts = torch.bincount(batch, minlength=batch_size)
        offsets = torch.cat((counts.new_zeros(1), counts.cumsum(dim=0)[:-1]))
        local_index = torch.arange(node.shape[0], device=node.device) - offsets[batch]
        dense_edge_bias = edge.new_zeros(
            (batch_size, max_nodes, max_nodes, self.num_heads)
        )
        dense_edge_bias.index_put_(
            (
                batch[edge_index[0]],
                local_index[edge_index[0]],
                local_index[edge_index[1]],
            ),
            self.edge_bias(edge).to(dtype=dense_edge_bias.dtype),
            accumulate=True,
        )
        dense_edge_bias = dense_edge_bias.permute(0, 3, 1, 2)
        scores = scores + dense_edge_bias
        scores = scores.masked_fill(
            ~node_mask[:, None, None, :], torch.finfo(scores.dtype).min
        )
        attention = torch.softmax(scores, dim=-1)
        dense_attended = torch.einsum("bhij,bjhd->bihd", attention, dense_value)
        attended = dense_attended.reshape(batch_size, max_nodes, -1)[node_mask]

        local = self._mean_edge_messages(self.edge_message(edge), edge_index, node.shape[0])
        node = node + self.dropout(self.attn_out(attended))
        node = node + self.dropout(self.local_out(local))
        node = node + self.dropout(self.ffn(self.ffn_norm(node)))
        return node, edge


class EdgeGlobal2DWrapper(nn.Module):
    """Pure-2D edge-state/global-attention predictor for QM9 and PCQM."""

    def __init__(
        self,
        in_channels: int = 11,
        edge_dim: int = 4,
        hidden_channels: int = 192,
        edge_channels: int = 64,
        num_layers: int = 8,
        num_heads: int = 4,
        dropout: float = 0.05,
        n_targets: int = 3,
        pooling: str = "mean_max",
    ):
        super().__init__()
        if pooling not in {"mean", "mean_max"}:
            raise ValueError(f"Unsupported pooling: {pooling}")
        self.pooling = pooling
        self.node_input = nn.Linear(in_channels, hidden_channels)
        self.edge_input = nn.Sequential(
            nn.Linear(edge_dim, edge_channels),
            nn.SiLU(),
            nn.Linear(edge_channels, edge_channels),
        )
        self.layers = nn.ModuleList(
            [
                _EdgeGlobalBlock(
                    hidden_channels,
                    edge_channels,
                    edge_dim,
                    num_heads,
                    dropout,
                )
                for _ in range(num_layers)
            ]
        )
        pooled_channels = hidden_channels * (2 if pooling == "mean_max" else 1)
        self.pool_proj = (
            nn.Linear(pooled_channels, hidden_channels)
            if pooled_channels != hidden_channels
            else nn.Identity()
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
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        from torch_geometric.nn import global_max_pool, global_mean_pool

        node = self.node_input(x.float())
        edge = self.edge_input(edge_attr.float())
        for layer in self.layers:
            node, edge = layer(node, edge, edge_index, edge_attr, batch)
        pooled = global_mean_pool(node, batch)
        if self.pooling == "mean_max":
            pooled = torch.cat((pooled, global_max_pool(node, batch)), dim=-1)
        return self.pool_proj(pooled)

    def forward(self, x, edge_index, edge_attr, batch):
        return self.head(self.encode(x, edge_index, edge_attr, batch))
