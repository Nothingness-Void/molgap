"""A GINE-free edge-aware global attention encoder for QM9 screening.

This is a small GPS/EGT-inspired topology branch.  It keeps a persistent
pair channel from the 2D bond graph and uses it as an attention bias, while
avoiding the optional PyG GINEConv CUDA path used by the first hybrid screen.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class _TopologyAttentionBlock(nn.Module):
    def __init__(self, hidden_channels: int, pair_channels: int, num_heads: int, dropout: float):
        super().__init__()
        if hidden_channels % num_heads:
            raise ValueError("hidden_channels must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_channels = hidden_channels // num_heads
        self.norm_attn = nn.LayerNorm(hidden_channels)
        self.qkv = nn.Linear(hidden_channels, hidden_channels * 3)
        self.attn_out = nn.Linear(hidden_channels, hidden_channels)
        self.pair_bias = nn.Linear(pair_channels, num_heads)
        self.dropout = nn.Dropout(dropout)
        self.norm_ffn = nn.LayerNorm(hidden_channels)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels * 4, hidden_channels),
        )
        self.node_to_pair = nn.Linear(hidden_channels, pair_channels)
        self.pair_update = nn.Sequential(
            nn.Linear(pair_channels * 3, pair_channels * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(pair_channels * 2, pair_channels),
        )
        self.norm_pair = nn.LayerNorm(pair_channels)

    def forward(
        self,
        node: torch.Tensor,
        pair: torch.Tensor,
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
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_channels)
        scores = scores + self.pair_bias(pair).permute(0, 3, 1, 2)
        scores = scores.masked_fill(~node_mask[:, None, None, :], torch.finfo(scores.dtype).min)
        attention = torch.softmax(scores, dim=-1)
        attended = torch.matmul(attention, value).transpose(1, 2).reshape_as(node)
        node = residual + self.dropout(self.attn_out(attended))
        node = node * node_mask.unsqueeze(-1)
        node = node + self.dropout(self.ffn(self.norm_ffn(node)))
        node = node * node_mask.unsqueeze(-1)

        pair_residual = pair
        node_pair = self.node_to_pair(node)
        node_pair = node_pair.unsqueeze(2) + node_pair.unsqueeze(1)
        triplet = torch.einsum("bikd,bkjd->bijd", pair, pair)
        triplet = triplet / math.sqrt(max(pair.shape[-1], 1))
        pair_input = torch.cat((pair, triplet, node_pair), dim=-1)
        pair = self.norm_pair(pair_residual + self.pair_update(pair_input))
        pair = pair * pair_mask.unsqueeze(-1)
        return node, pair


class TopologyAttentionWrapper(nn.Module):
    """Bond-topology encoder with global attention and persistent pair state."""

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
    ):
        super().__init__()
        self.edge_dim = int(edge_dim)
        self.node_features = nn.Linear(in_channels, hidden_channels)
        self.pair_input = nn.Sequential(
            nn.Linear(edge_dim + 1, pair_channels),
            nn.SiLU(),
            nn.Linear(pair_channels, pair_channels),
        )
        self.layers = nn.ModuleList(
            [
                _TopologyAttentionBlock(hidden_channels, pair_channels, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.SiLU(),
            nn.Linear(hidden_channels // 2, n_targets),
        )

    def _dense_inputs(
        self,
        x,
        edge_index,
        edge_attr,
        batch,
        *,
        topology_edges=None,
        topology_edge_attr=None,
        topology_node_counts=None,
        topology_edge_counts=None,
    ):
        from torch_geometric.utils import to_dense_adj, to_dense_batch

        if topology_node_counts is not None:
            node_counts = topology_node_counts.view(-1).long()
            if int(node_counts.sum()) != int(x.shape[0]):
                raise ValueError("topology node counts do not match topology_x")
            topology_batch = torch.repeat_interleave(
                torch.arange(node_counts.numel(), device=x.device), node_counts
            )
            if topology_edges is not None and topology_edge_counts is not None:
                edge_counts = topology_edge_counts.view(-1).long()
                if int(edge_counts.sum()) != int(topology_edges.shape[0]):
                    raise ValueError("topology edge counts do not match topology_edges")
                node_offsets = torch.cat(
                    (
                        node_counts.new_zeros(1),
                        node_counts.cumsum(dim=0)[:-1],
                    )
                )
                edge_offsets = torch.repeat_interleave(node_offsets, edge_counts)
                edge_index = topology_edges.t().long() + edge_offsets.unsqueeze(0)
                edge_attr = topology_edge_attr
            elif topology_edges is not None:
                edge_index = topology_edges.t().long()
                edge_attr = topology_edge_attr
            batch = topology_batch
        node, node_mask = to_dense_batch(x.float(), batch)
        max_nodes = node.shape[1]
        dense_edges = to_dense_adj(
            edge_index,
            batch,
            edge_attr=edge_attr.float(),
            max_num_nodes=max_nodes,
        )
        bond_mask = dense_edges.abs().sum(dim=-1, keepdim=True).clamp(max=1.0)
        pair_mask = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
        pair = self.pair_input(torch.cat((dense_edges, bond_mask), dim=-1))
        return node, pair * pair_mask.unsqueeze(-1), node_mask, pair_mask

    def encode(
        self,
        x,
        edge_index,
        edge_attr,
        batch,
        *,
        topology_edges=None,
        topology_edge_attr=None,
        topology_node_counts=None,
        topology_edge_counts=None,
    ):
        node, pair, node_mask, pair_mask = self._dense_inputs(
            x,
            edge_index,
            edge_attr,
            batch,
            topology_edges=topology_edges,
            topology_edge_attr=topology_edge_attr,
            topology_node_counts=topology_node_counts,
            topology_edge_counts=topology_edge_counts,
        )
        node = self.node_features(node) * node_mask.unsqueeze(-1)
        for layer in self.layers:
            node, pair = layer(node, pair, node_mask, pair_mask)
        masked = node * node_mask.unsqueeze(-1)
        return masked.sum(dim=1) / node_mask.sum(dim=1, keepdim=True).clamp_min(1)

    def forward(self, x, edge_index, edge_attr, batch):
        return self.head(self.encode(x, edge_index, edge_attr, batch))
