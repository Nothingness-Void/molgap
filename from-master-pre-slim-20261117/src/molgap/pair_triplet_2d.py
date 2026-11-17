"""Pure-2D pair/triplet graph transformer for the QM9 architecture screen.

This keeps the most portable idea from TGT/GEM-2/GPTrans: every molecular
node pair has a persistent state, and pair states communicate through a
low-rank triplet update.  The initial pair channels contain only bond
features and short 2D walk indicators, so no conformer or optional geometry
kernel is involved.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class _PairTriplet2DBlock(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        pair_channels: int,
        num_heads: int,
        dropout: float,
        triplet_rank: int,
    ):
        super().__init__()
        if hidden_channels % num_heads:
            raise ValueError("hidden_channels must be divisible by num_heads")
        self.num_heads = int(num_heads)
        self.head_channels = hidden_channels // num_heads
        self.norm_attn = nn.LayerNorm(hidden_channels)
        self.qkv = nn.Linear(hidden_channels, hidden_channels * 3)
        self.attn_out = nn.Linear(hidden_channels, hidden_channels)
        self.pair_bias = nn.Linear(pair_channels, num_heads)
        self.dropout = nn.Dropout(dropout)

        self.local_out = nn.Linear(hidden_channels, hidden_channels)
        self.norm_ffn = nn.LayerNorm(hidden_channels)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels * 4, hidden_channels),
        )

        self.node_to_pair = nn.Linear(hidden_channels, pair_channels)
        self.triplet_left = nn.Linear(pair_channels, triplet_rank, bias=False)
        self.triplet_right = nn.Linear(pair_channels, triplet_rank, bias=False)
        self.triplet_out = nn.Linear(triplet_rank, pair_channels)
        self.norm_pair = nn.LayerNorm(pair_channels)
        self.pair_update = nn.Sequential(
            nn.Linear(pair_channels * 3, pair_channels * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(pair_channels * 2, pair_channels),
        )

    def forward(
        self,
        node: torch.Tensor,
        pair: torch.Tensor,
        bond_mask: torch.Tensor,
        node_mask: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        residual = node
        normalized = self.norm_attn(node)
        qkv = self.qkv(normalized).view(
            normalized.shape[0], normalized.shape[1], 3,
            self.num_heads, self.head_channels,
        )
        q, k, value = qkv.unbind(dim=2)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)
        scores = torch.matmul(q, k.transpose(-1, -2))
        scores = scores / math.sqrt(self.head_channels)
        scores = scores + self.pair_bias(pair).permute(0, 3, 1, 2)
        scores = scores.masked_fill(
            ~node_mask[:, None, None, :], torch.finfo(scores.dtype).min
        )
        attention = torch.softmax(scores, dim=-1)
        attended = torch.matmul(attention, value).transpose(1, 2).reshape_as(node)
        node = residual + self.dropout(self.attn_out(attended))

        # Keep an explicit local bond path alongside global attention.  This
        # is the GPS/GPTrans inductive bias that the edge-only screen lacked.
        degree = bond_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        local = torch.matmul(bond_mask, node) / degree
        node = node + self.dropout(self.local_out(local))
        node = node * node_mask.unsqueeze(-1)
        node = node + self.dropout(self.ffn(self.norm_ffn(node)))
        node = node * node_mask.unsqueeze(-1)

        node_pair = self.node_to_pair(node)
        node_pair = node_pair.unsqueeze(2) + node_pair.unsqueeze(1)
        left = self.triplet_left(pair)
        right = self.triplet_right(pair)
        triplet = torch.einsum("bikd,bkjd->bijd", left, right)
        triplet = self.triplet_out(triplet / math.sqrt(left.shape[-1]))
        pair_input = torch.cat((pair, triplet, node_pair), dim=-1)
        pair = self.norm_pair(pair + self.pair_update(pair_input))
        pair = pair * pair_mask.unsqueeze(-1)
        return node, pair


class PairTriplet2DWrapper(nn.Module):
    """Pure-2D all-pairs transformer with local bonds and triplet mixing."""

    def __init__(
        self,
        in_channels: int = 11,
        edge_dim: int = 4,
        hidden_channels: int = 192,
        pair_channels: int = 48,
        num_layers: int = 6,
        num_heads: int = 4,
        dropout: float = 0.05,
        n_targets: int = 3,
        pooling: str = "mean_max",
        path_steps: int = 3,
        triplet_rank: int = 8,
    ):
        super().__init__()
        if pooling not in {"mean", "mean_max"}:
            raise ValueError(f"Unsupported pooling: {pooling}")
        if path_steps < 1:
            raise ValueError("path_steps must be positive")
        self.pooling = pooling
        self.edge_dim = int(edge_dim)
        self.path_steps = int(path_steps)
        self.node_input = nn.Linear(in_channels, hidden_channels)
        # ``path_features`` contains bond reachability plus the remaining
        # walk powers, so its width is exactly ``path_steps``.  The bond
        # indicator is already the first path channel; do not count it twice.
        pair_input_dim = edge_dim + path_steps
        self.pair_input = nn.Sequential(
            nn.Linear(pair_input_dim, pair_channels),
            nn.SiLU(),
            nn.Linear(pair_channels, pair_channels),
        )
        self.layers = nn.ModuleList(
            [
                _PairTriplet2DBlock(
                    hidden_channels,
                    pair_channels,
                    num_heads,
                    dropout,
                    triplet_rank,
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

    @staticmethod
    def _local_edge_index(edge_index, batch, node_count: int):
        batch_size = int(batch.max().item()) + 1 if batch.numel() else 0
        counts = torch.bincount(batch, minlength=batch_size)
        offsets = torch.cat((counts.new_zeros(1), counts.cumsum(dim=0)[:-1]))
        local = torch.arange(node_count, device=batch.device) - offsets[batch]
        return batch[edge_index[0]], local[edge_index[0]], local[edge_index[1]]

    def _dense_inputs(self, x, edge_index, edge_attr, batch):
        from torch_geometric.utils import to_dense_batch

        node, node_mask = to_dense_batch(x.float(), batch)
        batch_size, max_nodes = node.shape[:2]
        dense_edges = x.new_zeros((batch_size, max_nodes, max_nodes, self.edge_dim))
        edge_batch, edge_src, edge_dst = self._local_edge_index(
            edge_index, batch, int(x.shape[0])
        )
        dense_edges.index_put_(
            (edge_batch, edge_src, edge_dst),
            edge_attr.float().to(dtype=dense_edges.dtype),
            accumulate=True,
        )
        bond = x.new_zeros((batch_size, max_nodes, max_nodes))
        bond.index_put_(
            (edge_batch, edge_src, edge_dst),
            torch.ones(edge_batch.shape[0], device=x.device, dtype=bond.dtype),
            accumulate=True,
        )
        bond = (bond > 0).to(dtype=x.dtype)
        path_features = [bond]
        walk = bond
        for _ in range(1, self.path_steps):
            walk = torch.bmm(walk, bond).clamp(max=1.0)
            path_features.append(walk)
        pair_features = torch.cat(
            (dense_edges, torch.stack(path_features, dim=-1)), dim=-1
        )
        pair_mask = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
        pair = self.pair_input(pair_features) * pair_mask.unsqueeze(-1)
        return node, pair, bond, node_mask, pair_mask

    def encode(self, x, edge_index, edge_attr, batch):
        node, pair, bond, node_mask, pair_mask = self._dense_inputs(
            x, edge_index, edge_attr, batch
        )
        node = self.node_input(node) * node_mask.unsqueeze(-1)
        for layer in self.layers:
            node, pair = layer(node, pair, bond, node_mask, pair_mask)
        masked = node * node_mask.unsqueeze(-1)
        mean = masked.sum(dim=1) / node_mask.sum(dim=1, keepdim=True).clamp_min(1)
        if self.pooling == "mean":
            return mean
        maximum = masked.masked_fill(
            ~node_mask.unsqueeze(-1), torch.finfo(masked.dtype).min
        ).amax(dim=1)
        return self.pool_proj(torch.cat((mean, maximum), dim=-1))

    def forward(self, x, edge_index, edge_attr, batch):
        return self.head(self.encode(x, edge_index, edge_attr, batch))
