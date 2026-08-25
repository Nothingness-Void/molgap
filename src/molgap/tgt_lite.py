"""A bounded TGT-inspired pair/triplet graph transformer.

The official Triplet Graph Transformer is a large two-stage PCQM model.  This
module keeps the transferable idea rather than copying its scale: molecular
nodes attend globally, an explicit pair channel carries geometry/topology, and
pair channels receive a low-rank triplet update.  It is intended for the
deployment-matched ETKDG QM9 screen, not as a claim of exact TGT parity.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


def _aligned_atomic_features(z: torch.Tensor, channels: int = 11) -> torch.Tensor:
    """Build node features in the same explicit-H order as the 3D cache."""
    if channels != 11:
        raise ValueError("TGTLite atomic feature schema is fixed at 11 channels")
    features = torch.zeros(
        (z.numel(), channels), dtype=torch.float32, device=z.device
    )
    # Keep the schema small but cover QM9 and the common expanded-PCQM atoms.
    atomic_columns = {
        1: 0,
        6: 1,
        7: 2,
        8: 3,
        9: 4,
        15: 5,
        16: 6,
        17: 7,
        34: 8,
        35: 9,
    }
    for atomic_number, column in atomic_columns.items():
        features[:, column] = (z == atomic_number).to(features.dtype)
    features[:, 10] = z.to(features.dtype).clamp(min=0, max=118) / 118.0
    return features


class _PairTripletBlock(nn.Module):
    def __init__(self, hidden_channels: int, pair_channels: int, num_heads: int, dropout: float):
        super().__init__()
        if hidden_channels % num_heads:
            raise ValueError("hidden_channels must be divisible by num_heads")
        self.hidden_channels = hidden_channels
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
        triplet = torch.einsum("bikd,bkjd->bijd", pair, pair)
        triplet = triplet / math.sqrt(max(pair.shape[-1], 1))
        node_pair = self.node_to_pair(node)
        node_pair = node_pair.unsqueeze(2) + node_pair.unsqueeze(1)
        pair_input = torch.cat((pair, triplet, node_pair), dim=-1)
        pair = self.norm_pair(pair_residual + self.pair_update(pair_input))
        pair = pair * pair_mask.unsqueeze(-1)
        return node, pair


class TGTLiteWrapper(nn.Module):
    """ETKDG-compatible global pair/triplet transformer for QM9 screening."""

    def __init__(
        self,
        in_channels: int = 11,
        edge_dim: int = 4,
        hidden_channels: int = 128,
        pair_channels: int = 32,
        num_layers: int = 6,
        num_heads: int = 4,
        num_rbf: int = 32,
        cutoff: float = 12.0,
        dropout: float = 0.05,
        n_targets: int = 3,
    ):
        super().__init__()
        self.cutoff = float(cutoff)
        self.num_rbf = int(num_rbf)
        self.edge_dim = int(edge_dim)
        # Atomic numbers are used directly so the same model can consume the
        # expanded PCQM Route B element schema (including Br/Se), not only QM9.
        self.node_embedding = nn.Embedding(119, hidden_channels)
        self.node_features = nn.Linear(in_channels, hidden_channels)
        centers = torch.linspace(0.0, self.cutoff, self.num_rbf)
        self.register_buffer("rbf_centers", centers)
        self.rbf_width = nn.Parameter(torch.tensor(0.35))
        self.pair_input = nn.Sequential(
            nn.Linear(self.num_rbf + edge_dim + 1, pair_channels),
            nn.SiLU(),
            nn.Linear(pair_channels, pair_channels),
        )
        self.layers = nn.ModuleList(
            [
                _PairTripletBlock(hidden_channels, pair_channels, num_heads, dropout)
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

    def _dense_inputs(self, x, edge_index, edge_attr, z, pos, batch):
        from torch_geometric.utils import to_dense_batch

        # The QM9 ETKDG cache stores explicit-H coordinates, while its legacy
        # 2D x/edge view is heavy-atom-only for many molecules.  The candidate
        # must therefore derive every node input from z/pos, whose batch
        # alignment is the immutable geometry contract.  Pair geometry and
        # triplet updates remain the transferable TGT ideas under test.
        aligned_features = _aligned_atomic_features(z, self.node_features.in_features)
        node_features, node_mask = to_dense_batch(aligned_features, batch)
        dense_z, _ = to_dense_batch(z, batch, fill_value=0)
        dense_pos, _ = to_dense_batch(pos, batch, fill_value=0.0)
        max_nodes = node_features.shape[1]
        dense_edges = dense_pos.new_zeros(
            (dense_pos.shape[0], max_nodes, max_nodes, self.edge_dim)
        )
        pair_mask = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
        distance = torch.cdist(dense_pos, dense_pos).clamp(max=self.cutoff)
        scaled_width = self.rbf_width.abs().clamp_min(0.05)
        rbf = torch.exp(-scaled_width * (distance.unsqueeze(-1) - self.rbf_centers) ** 2)
        bond_mask = dense_edges.sum(dim=-1, keepdim=True).clamp(max=1.0)
        pair_input = torch.cat((rbf, dense_edges, bond_mask), dim=-1)
        return node_features, dense_z, pair_input, node_mask, pair_mask

    def encode(self, x, edge_index, edge_attr, z, pos, batch):
        node_features, dense_z, pair_input, node_mask, pair_mask = self._dense_inputs(
            x, edge_index, edge_attr, z, pos, batch
        )
        node = self.node_embedding(dense_z.clamp(min=0, max=118)) + self.node_features(node_features)
        pair = self.pair_input(pair_input) * pair_mask.unsqueeze(-1)
        for layer in self.layers:
            node, pair = layer(node, pair, node_mask, pair_mask)
        masked_node = node * node_mask.unsqueeze(-1)
        pooled = masked_node.sum(dim=1) / node_mask.sum(dim=1, keepdim=True).clamp_min(1)
        return pooled

    def forward(self, x, edge_index, edge_attr, z, pos, batch):
        return self.head(self.encode(x, edge_index, edge_attr, z, pos, batch))
