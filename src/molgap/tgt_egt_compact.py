"""Stable compact EGT/TGT geometry encoder for the QM9 Route-C screen.

This candidate starts from the already measured ``tgt_lite`` geometry path.
It adds explicit heavy-atom bond channels to the ETKDG pair input and a
zero-initialized pair-to-node residual.  The zero initialization keeps the
first optimization step close to the known baseline while allowing the EGT
signal to learn if it is useful.  The implementation uses only dense PyTorch
operations and supports the cache's separate heavy-atom and explicit-H views.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .tgt_lite import _PairTripletBlock, _aligned_atomic_features


class _CompactPairBlock(_PairTripletBlock):
    """TGT-lite block with a zero-initialized pair-value residual."""

    def __init__(
        self,
        hidden_channels: int,
        pair_channels: int,
        num_heads: int,
        dropout: float,
    ):
        super().__init__(hidden_channels, pair_channels, num_heads, dropout)
        self.pair_value = nn.Linear(pair_channels, hidden_channels)
        self.pair_value_scale = nn.Parameter(torch.zeros(()))

    def forward(self, node, pair, node_mask, pair_mask):
        node, pair = super().forward(node, pair, node_mask, pair_mask)
        pair_values = self.pair_value(pair) * pair_mask.unsqueeze(-1)
        pair_values = pair_values.sum(dim=2) / pair_mask.sum(
            dim=2, keepdim=True
        ).clamp_min(1)
        node = node + torch.tanh(self.pair_value_scale) * pair_values
        return node * node_mask.unsqueeze(-1), pair


class TGTCompactEGTWrapper(nn.Module):
    """ETKDG TGT-lite with EGT bond channels and a safe value residual."""

    def __init__(
        self,
        in_channels: int = 11,
        edge_dim: int = 4,
        hidden_channels: int = 192,
        pair_channels: int = 48,
        num_layers: int = 8,
        num_heads: int = 4,
        num_rbf: int = 32,
        cutoff: float = 12.0,
        dropout: float = 0.05,
        zero_init_bond_channels: bool = False,
        n_targets: int = 3,
    ):
        super().__init__()
        self.cutoff = float(cutoff)
        self.edge_dim = int(edge_dim)
        self.node_embedding = nn.Embedding(119, hidden_channels)
        self.node_features = nn.Linear(in_channels, hidden_channels)
        self.register_buffer("rbf_centers", torch.linspace(0.0, cutoff, num_rbf))
        self.rbf_width = nn.Parameter(torch.tensor(0.35))
        self.pair_input = nn.Sequential(
            nn.Linear(num_rbf + edge_dim + 1, pair_channels),
            nn.SiLU(),
            nn.Linear(pair_channels, pair_channels),
        )
        if zero_init_bond_channels:
            # Keep the step-0 function equal to the measured tgt_lite path;
            # the new EGT columns must earn their contribution during training.
            with torch.no_grad():
                self.pair_input[0].weight[:, -(edge_dim + 1) :].zero_()
        self.layers = nn.ModuleList(
            [
                _CompactPairBlock(
                    hidden_channels, pair_channels, num_heads, dropout
                )
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

    @staticmethod
    def _dense_bonds(
        topology_edges,
        topology_edge_attr,
        topology_node_counts,
        topology_edge_counts,
        batch_size,
        max_nodes,
        edge_dim,
        dtype,
        device,
    ):
        dense = torch.zeros(
            (batch_size, max_nodes, max_nodes, edge_dim),
            device=device,
            dtype=dtype,
        )
        if (
            topology_edges is None
            or topology_edge_attr is None
            or topology_edges.numel() == 0
        ):
            return dense
        node_counts = topology_node_counts.view(-1).long()
        edge_counts = topology_edge_counts.view(-1).long()
        if int(node_counts.numel()) != batch_size:
            raise ValueError("topology counts do not match geometry batch")
        if int(edge_counts.sum()) != int(topology_edges.shape[0]):
            raise ValueError("topology edge counts do not match topology edges")
        node_offsets = torch.cat(
            (node_counts.new_zeros(1), node_counts.cumsum(dim=0)[:-1])
        )
        edge_batch = torch.repeat_interleave(
            torch.arange(batch_size, device=device), edge_counts
        )
        local_row = topology_edges[:, 0].long() - node_offsets[edge_batch]
        local_col = topology_edges[:, 1].long() - node_offsets[edge_batch]
        valid = (
            (local_row >= 0)
            & (local_col >= 0)
            & (local_row < max_nodes)
            & (local_col < max_nodes)
        )
        if valid.any():
            dense[edge_batch[valid], local_row[valid], local_col[valid]] = (
                topology_edge_attr[valid].to(dtype)
            )
        return dense

    def _dense_inputs(
        self,
        x,
        edge_index,
        edge_attr,
        z,
        pos,
        batch,
        *,
        topology_edges=None,
        topology_edge_attr=None,
        topology_node_counts=None,
        topology_edge_counts=None,
        geometry_node_counts=None,
    ):
        del x, edge_index, edge_attr
        from torch_geometric.utils import to_dense_batch

        if geometry_node_counts is not None:
            counts = geometry_node_counts.view(-1).long()
            if int(counts.sum()) != int(z.shape[0]):
                raise ValueError("geometry node counts do not match z")
            geometry_batch = torch.repeat_interleave(
                torch.arange(counts.numel(), device=z.device), counts
            )
        else:
            geometry_batch = batch

        aligned = _aligned_atomic_features(z, self.node_features.in_features)
        node_features, node_mask = to_dense_batch(aligned, geometry_batch)
        dense_z, _ = to_dense_batch(z, geometry_batch, fill_value=0)
        dense_pos, _ = to_dense_batch(pos, geometry_batch, fill_value=0.0)
        batch_size, max_nodes = dense_z.shape
        dense_z = dense_z.long().clamp(min=0, max=118)
        pair_mask = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
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
            batch_size,
            max_nodes,
            self.edge_dim,
            rbf.dtype,
            z.device,
        )
        bond_mask = bonds.abs().sum(dim=-1, keepdim=True).clamp(max=1.0)
        pair_input = torch.cat((rbf, bonds, bond_mask), dim=-1)
        return node_features, dense_z, pair_input, node_mask, pair_mask

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
        del topology_x
        node_features, dense_z, pair_input, node_mask, pair_mask = (
            self._dense_inputs(
                x,
                edge_index,
                edge_attr,
                z,
                pos,
                batch,
                topology_edges=topology_edges,
                topology_edge_attr=topology_edge_attr,
                topology_node_counts=topology_node_counts,
                topology_edge_counts=topology_edge_counts,
                geometry_node_counts=geometry_node_counts,
            )
        )
        node = self.node_embedding(dense_z) + self.node_features(node_features)
        node = node * node_mask.unsqueeze(-1)
        pair = self.pair_input(pair_input) * pair_mask.unsqueeze(-1)
        for layer in self.layers:
            node, pair = layer(node, pair, node_mask, pair_mask)
        masked = node * node_mask.unsqueeze(-1)
        return masked.sum(dim=1) / node_mask.sum(dim=1, keepdim=True).clamp_min(1)

    def forward(self, x, edge_index, edge_attr, z, pos, batch, **kwargs):
        return self.head(self.encode(x, edge_index, edge_attr, z, pos, batch, **kwargs))
