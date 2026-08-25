"""Portable EGT/TGT-style 2D+ETKDG hybrid for the QM9 route-C screen.

The high-ranking EGT/TGT family keeps a persistent pair state, lets pair
states communicate through triplets, and feeds pair values back to nodes.  The
original MolGap transfer screen had the pair bias but not the pair-value path
and used a late topology branch that was weaker than GPS.  This module keeps
the useful operations while using only dense PyTorch/index_add operations, so
it does not depend on the optional torch-cluster CUDA extension.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


def _counts_to_batch(counts: torch.Tensor) -> torch.Tensor:
    counts = counts.view(-1).long()
    return torch.repeat_interleave(
        torch.arange(counts.numel(), device=counts.device), counts
    )


def _offset_edges(
    edges: torch.Tensor,
    node_counts: torch.Tensor,
    edge_counts: torch.Tensor,
) -> torch.Tensor:
    """Convert per-molecule ``[E, 2]`` edges to one global ``[2, E]`` tensor."""
    if edges.numel() == 0:
        return edges.new_zeros((2, 0), dtype=torch.long)
    node_offsets = torch.cat(
        (
            node_counts.new_zeros(1),
            node_counts.cumsum(dim=0)[:-1],
        )
    )
    edge_offsets = torch.repeat_interleave(node_offsets, edge_counts)
    return edges.t().long() + edge_offsets.unsqueeze(0)


class _LocalGlobalBlock(nn.Module):
    """GINE-free local message passing followed by global node attention."""

    def __init__(self, hidden: int, edge_dim: int, heads: int, dropout: float):
        super().__init__()
        if hidden % heads:
            raise ValueError("hidden must be divisible by heads")
        self.hidden = hidden
        self.heads = heads
        self.head_dim = hidden // heads
        self.edge_message = nn.Sequential(
            nn.Linear(hidden + edge_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.local_norm = nn.LayerNorm(hidden)
        self.local_update = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
        )
        self.attn_norm = nn.LayerNorm(hidden)
        self.qkv = nn.Linear(hidden, hidden * 3)
        self.attn_out = nn.Linear(hidden, hidden)
        self.dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, hidden * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 4, hidden),
        )

    def _local(self, node, edges, edge_attr):
        if edges.numel() == 0:
            return node
        row, col = edges
        messages = self.edge_message(
            torch.cat((node[row], edge_attr.to(node.dtype)), dim=-1)
        )
        aggregated = node.new_zeros(node.shape)
        aggregated.index_add_(0, col, messages)
        degree = node.new_zeros((node.shape[0], 1))
        degree.index_add_(
            0,
            col,
            node.new_ones((col.numel(), 1)),
        )
        aggregated = aggregated / degree.clamp_min(1.0)
        return node + self.local_update(
            torch.cat((self.local_norm(node), aggregated), dim=-1)
        )

    def forward(self, node, batch, edges, edge_attr):
        node = self._local(node, edges, edge_attr)
        from torch_geometric.utils import to_dense_batch

        dense, mask = to_dense_batch(node, batch)
        residual = dense
        normalized = self.attn_norm(dense)
        qkv = self.qkv(normalized).view(
            dense.shape[0], dense.shape[1], 3, self.heads, self.head_dim
        )
        q, k, value = qkv.unbind(dim=2)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(
            ~mask[:, None, None, :], torch.finfo(scores.dtype).min
        )
        attention = torch.softmax(scores, dim=-1)
        attended = torch.matmul(attention, value).transpose(1, 2)
        attended = attended.reshape_as(dense)
        dense = residual + self.dropout(self.attn_out(attended))
        dense = dense + self.dropout(self.ffn(self.ffn_norm(dense)))
        dense = dense * mask.unsqueeze(-1)
        return dense[mask]


class _PairGeometryBlock(nn.Module):
    """Pair-bias attention with triplet mixing and pair-to-node values."""

    def __init__(self, hidden: int, pair_channels: int, heads: int, dropout: float):
        super().__init__()
        if hidden % heads:
            raise ValueError("hidden must be divisible by heads")
        self.hidden = hidden
        self.heads = heads
        self.head_dim = hidden // heads
        self.node_norm = nn.LayerNorm(hidden)
        self.qkv = nn.Linear(hidden, hidden * 3)
        self.pair_bias = nn.Linear(pair_channels, heads)
        self.pair_value = nn.Linear(pair_channels, hidden)
        self.attn_out = nn.Linear(hidden, hidden)
        self.dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, hidden * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 4, hidden),
        )
        self.node_to_pair = nn.Linear(hidden, pair_channels)
        self.pair_update = nn.Sequential(
            nn.Linear(pair_channels * 3, pair_channels * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(pair_channels * 2, pair_channels),
        )
        self.pair_norm = nn.LayerNorm(pair_channels)

    def forward(self, node, pair, node_mask, pair_mask):
        residual = node
        normalized = self.node_norm(node)
        qkv = self.qkv(normalized).view(
            node.shape[0], node.shape[1], 3, self.heads, self.head_dim
        )
        q, k, value = qkv.unbind(dim=2)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        scores = scores + self.pair_bias(pair).permute(0, 3, 1, 2)
        scores = scores.masked_fill(
            ~node_mask[:, None, None, :], torch.finfo(scores.dtype).min
        )
        attended = torch.softmax(scores, dim=-1)
        attended = torch.matmul(attended, value).transpose(1, 2)
        attended = attended.reshape_as(node)

        # This value path is the key EGT/TGT transfer: pair state contributes
        # directly to each receiving node, not only to the attention logits.
        pair_values = self.pair_value(pair) * pair_mask.unsqueeze(-1)
        pair_values = pair_values.sum(dim=2) / pair_mask.sum(
            dim=2, keepdim=True
        ).clamp_min(1)
        node = residual + self.dropout(self.attn_out(attended))
        node = node + self.dropout(pair_values)
        node = node + self.dropout(self.ffn(self.ffn_norm(node)))
        node = node * node_mask.unsqueeze(-1)

        triplet = torch.einsum("bikd,bkjd->bijd", pair, pair)
        triplet = triplet / math.sqrt(max(pair.shape[-1], 1))
        node_pair = self.node_to_pair(node)
        node_pair = node_pair.unsqueeze(2) + node_pair.unsqueeze(1)
        pair_input = torch.cat((pair, triplet, node_pair), dim=-1)
        pair = self.pair_norm(pair + self.pair_update(pair_input))
        pair = pair * pair_mask.unsqueeze(-1)
        return node, pair


class _GeometryPairEncoder(nn.Module):
    def __init__(
        self,
        hidden: int,
        pair_channels: int,
        heads: int,
        layers: int,
        rbf: int,
        cutoff: float,
        edge_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.hidden = hidden
        self.pair_channels = pair_channels
        self.edge_dim = edge_dim
        self.cutoff = float(cutoff)
        self.node_embedding = nn.Embedding(119, hidden)
        self.node_scalar = nn.Linear(1, hidden)
        self.register_buffer("rbf_centers", torch.linspace(0.0, cutoff, rbf))
        self.rbf_width = nn.Parameter(torch.tensor(0.35))
        self.pair_input = nn.Sequential(
            nn.Linear(rbf + edge_dim + 3, pair_channels),
            nn.SiLU(),
            nn.Linear(pair_channels, pair_channels),
        )
        self.layers = nn.ModuleList(
            [
                _PairGeometryBlock(hidden, pair_channels, heads, dropout)
                for _ in range(layers)
            ]
        )

    def _dense_bonds(
        self,
        topology_edges,
        topology_edge_attr,
        topology_node_counts,
        topology_edge_counts,
        geometry_counts,
        max_nodes,
        batch_size,
        dtype,
    ):
        dense = torch.zeros(
            (batch_size, max_nodes, max_nodes, self.edge_dim),
            device=geometry_counts.device,
            dtype=dtype,
        )
        if topology_edges is None or topology_edges.numel() == 0:
            return dense
        edges = _offset_edges(
            topology_edges,
            topology_node_counts,
            topology_edge_counts,
        )
        node_offsets = torch.cat(
            (
                topology_node_counts.new_zeros(1),
                topology_node_counts.cumsum(dim=0)[:-1],
            )
        )
        edge_batch = torch.repeat_interleave(
            torch.arange(batch_size, device=edges.device), topology_edge_counts
        )
        local_row = edges[0] - node_offsets[edge_batch]
        local_col = edges[1] - node_offsets[edge_batch]
        values = topology_edge_attr.to(dtype)
        # Custom topology attributes are stored with per-molecule local edge
        # indices, but malformed/legacy batches must not turn one bad edge
        # into a device-side CUDA assert.  Ignore only out-of-range entries;
        # accepted in-range bonds remain unchanged.
        valid = (
            (edge_batch >= 0)
            & (local_row >= 0)
            & (local_col >= 0)
            & (local_row < max_nodes)
            & (local_col < max_nodes)
        )
        if valid.any():
            dense[edge_batch[valid], local_row[valid], local_col[valid]] = values[valid]
        return dense

    def encode(
        self,
        z,
        pos,
        geometry_batch,
        *,
        topology_edges,
        topology_edge_attr,
        topology_node_counts,
        topology_edge_counts,
    ):
        from torch_geometric.utils import to_dense_batch

        dense_z, node_mask = to_dense_batch(z, geometry_batch, fill_value=0)
        dense_pos, _ = to_dense_batch(pos, geometry_batch, fill_value=0.0)
        batch_size, max_nodes = dense_z.shape
        dense_z = dense_z.long().clamp(min=0, max=118)
        node = self.node_embedding(dense_z)
        node = node + self.node_scalar(dense_z.float().unsqueeze(-1) / 118.0)

        distance = torch.cdist(dense_pos.float(), dense_pos.float()).clamp(
            max=self.cutoff
        )
        width = self.rbf_width.abs().clamp_min(0.05)
        rbf = torch.exp(
            -width * (distance.unsqueeze(-1) - self.rbf_centers) ** 2
        )
        bonds = self._dense_bonds(
            topology_edges,
            topology_edge_attr,
            topology_node_counts,
            topology_edge_counts,
            node_mask.sum(dim=1),
            max_nodes,
            batch_size,
            rbf.dtype,
        )
        pair_mask = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
        same_element = (dense_z.unsqueeze(1) == dense_z.unsqueeze(2)).to(rbf.dtype)
        scaled_distance = (distance / self.cutoff).unsqueeze(-1)
        bond_mask = bonds.abs().sum(dim=-1, keepdim=True).clamp(max=1.0)
        pair_input = torch.cat(
            (rbf, bonds, bond_mask, same_element.unsqueeze(-1), scaled_distance),
            dim=-1,
        )
        pair = self.pair_input(pair_input) * pair_mask.unsqueeze(-1)
        for layer in self.layers:
            node, pair = layer(node, pair, node_mask, pair_mask)
        masked = node * node_mask.unsqueeze(-1)
        mean = masked.sum(dim=1) / node_mask.sum(dim=1, keepdim=True).clamp_min(1)
        maximum = masked.masked_fill(~node_mask.unsqueeze(-1), torch.finfo(masked.dtype).min).max(dim=1).values
        return (mean + maximum) * 0.5


class TGTEGTHybridWrapper(nn.Module):
    """GINE-free GPS-like topology + EGT/TGT-like ETKDG geometry hybrid."""

    def __init__(
        self,
        in_channels: int = 11,
        edge_dim: int = 4,
        hidden_channels: int = 192,
        pair_channels: int = 64,
        num_layers: int = 8,
        num_heads: int = 4,
        num_rbf: int = 64,
        cutoff: float = 12.0,
        dropout: float = 0.05,
        topology_layers: int = 9,
        n_targets: int = 3,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.edge_dim = edge_dim
        self.topology_node = nn.Linear(in_channels, hidden_channels)
        self.topology_edge = nn.Linear(edge_dim, edge_dim)
        self.topology_layers = nn.ModuleList(
            [
                _LocalGlobalBlock(hidden_channels, edge_dim, num_heads, dropout)
                for _ in range(topology_layers)
            ]
        )
        self.geometry = _GeometryPairEncoder(
            hidden_channels,
            pair_channels,
            num_heads,
            num_layers,
            num_rbf,
            cutoff,
            edge_dim,
            dropout,
        )
        self.fusion_gate = nn.Sequential(
            nn.LayerNorm(hidden_channels * 2),
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.Sigmoid(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.SiLU(),
            nn.Linear(hidden_channels // 2, n_targets),
        )

    def _topology_encode(
        self,
        topology_x,
        topology_edges,
        topology_edge_attr,
        topology_node_counts,
        topology_edge_counts,
    ):
        topology_batch = _counts_to_batch(topology_node_counts)
        edges = _offset_edges(
            topology_edges,
            topology_node_counts,
            topology_edge_counts,
        )
        edge_attr = self.topology_edge(topology_edge_attr.float())
        node = self.topology_node(topology_x.float())
        for layer in self.topology_layers:
            node = layer(node, topology_batch, edges, edge_attr)
        from torch_geometric.nn import global_mean_pool, global_max_pool

        return 0.5 * (
            global_mean_pool(node, topology_batch)
            + global_max_pool(node, topology_batch)
        )

    def encode(
        self,
        x,
        edge_index,
        edge_attr,
        z,
        pos,
        batch,
        *,
        topology_x=None,
        topology_edges=None,
        topology_edge_attr=None,
        topology_node_counts=None,
        topology_edge_counts=None,
        geometry_node_counts=None,
    ):
        topology_x = x if topology_x is None else topology_x
        topology_edges = (
            edge_index.t().contiguous() if topology_edges is None else topology_edges
        )
        topology_edge_attr = edge_attr if topology_edge_attr is None else topology_edge_attr
        if topology_node_counts is None:
            topology_node_counts = torch.bincount(batch)
        if topology_edge_counts is None:
            topology_edge_counts = torch.bincount(
                torch.zeros(topology_edges.shape[0], dtype=torch.long, device=z.device),
                minlength=topology_node_counts.numel(),
            )
        if geometry_node_counts is None:
            geometry_node_counts = torch.bincount(batch)
        geometry_batch = _counts_to_batch(geometry_node_counts)
        topology = self._topology_encode(
            topology_x,
            topology_edges,
            topology_edge_attr,
            topology_node_counts,
            topology_edge_counts,
        )
        geometry = self.geometry.encode(
            z,
            pos,
            geometry_batch,
            topology_edges=topology_edges,
            topology_edge_attr=topology_edge_attr,
            topology_node_counts=topology_node_counts,
            topology_edge_counts=topology_edge_counts,
        )
        gate = self.fusion_gate(torch.cat((topology, geometry), dim=-1))
        return gate * topology + (1.0 - gate) * geometry

    def forward(self, x, edge_index, edge_attr, z, pos, batch, **kwargs):
        return self.head(self.encode(x, edge_index, edge_attr, z, pos, batch, **kwargs))
