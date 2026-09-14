"""GPTrans-T core adapted to MolGap's accepted PyG graph batches.

This module follows the published GPTrans-T node/pair propagation shape:
12 blocks, 256-channel nodes, 32-channel all-pairs state, eight attention
heads, a virtual graph node, shortest-path pair encoding, and a unit-ratio
node FFN.  It intentionally excludes PairGPS additions such as GINE, RWSE,
triplet updates, and handcrafted pair descriptors.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_batch


class DropPath(nn.Module):
    def __init__(self, probability: float = 0.0) -> None:
        super().__init__()
        self.probability = float(probability)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if self.probability == 0.0 or not self.training:
            return value
        keep = 1.0 - self.probability
        shape = (value.shape[0],) + (1,) * (value.ndim - 1)
        mask = value.new_empty(shape).bernoulli_(keep)
        return value * mask / keep


class GraphPropagationAttention(nn.Module):
    """Published node-to-node, node-to-pair, and pair-to-node propagation."""

    def __init__(
        self,
        node_channels: int,
        pair_channels: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if node_channels % num_heads:
            raise ValueError("node_channels must be divisible by num_heads")
        self.num_heads = int(num_heads)
        self.head_channels = node_channels // num_heads
        self.scale = self.head_channels**-0.5
        self.qkv = nn.Linear(node_channels, node_channels * 3)
        self.pair_to_attention = nn.Conv2d(pair_channels, num_heads, 1)
        self.attention_to_pair = nn.Conv2d(num_heads, pair_channels, 1)
        self.pair_to_node = nn.Linear(pair_channels, node_channels)
        self.output = nn.Linear(node_channels, node_channels)
        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)

    def forward(
        self,
        node: torch.Tensor,
        pair: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, node_count, channels = node.shape
        qkv = self.qkv(node).reshape(
            batch_size,
            node_count,
            3,
            self.num_heads,
            self.head_channels,
        ).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        logits = (query @ key.transpose(-2, -1)) * self.scale
        logits = logits + self.pair_to_attention(pair)
        residual_logits = logits
        logits = logits.masked_fill(key_padding_mask, float("-inf"))
        attention = self.attention_dropout(torch.softmax(logits, dim=-1))
        node_update = (attention @ value).transpose(1, 2).reshape(
            batch_size, node_count, channels
        )

        pair_update = self.attention_to_pair(attention + residual_logits)
        pair_weights = pair_update.masked_fill(key_padding_mask, float("-inf"))
        pair_weights = torch.softmax(pair_weights, dim=-1)
        pair_message = (pair_weights * pair_update).sum(-1).transpose(1, 2)
        node_update = node_update + self.pair_to_node(pair_message)
        return self.output_dropout(self.output(node_update)), pair_update


class GPTransBlock(nn.Module):
    def __init__(
        self,
        node_channels: int,
        pair_channels: int,
        num_heads: int,
        dropout: float,
        drop_path: float,
        layer_scale: float,
    ) -> None:
        super().__init__()
        self.node_norm1 = nn.LayerNorm(node_channels)
        self.attention = GraphPropagationAttention(
            node_channels, pair_channels, num_heads, dropout
        )
        self.node_norm2 = nn.LayerNorm(node_channels)
        self.ffn = nn.Sequential(
            nn.Linear(node_channels, node_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(node_channels, node_channels),
            nn.Dropout(dropout),
        )
        self.drop_path = DropPath(drop_path)
        self.attention_scale = nn.Parameter(
            torch.full((node_channels,), float(layer_scale))
        )
        self.ffn_scale = nn.Parameter(
            torch.full((node_channels,), float(layer_scale))
        )

    def forward(
        self,
        node: torch.Tensor,
        pair: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        node_update, pair_update = self.attention(
            self.node_norm1(node), pair, key_padding_mask
        )
        pair = pair + pair_update
        node = node + self.drop_path(self.attention_scale * node_update)
        node = node + self.drop_path(self.ffn_scale * self.ffn(self.node_norm2(node)))
        return node, pair


class OGBGPTransTiny(nn.Module):
    """Direct-Gap GPTrans-T with OGB categorical atom and bond encoders."""

    def __init__(
        self,
        node_channels: int = 256,
        pair_channels: int = 32,
        num_layers: int = 12,
        num_heads: int = 8,
        shortest_path_cap: int = 20,
        dropout: float = 0.1,
        drop_path: float = 0.1,
        layer_scale: float = 1.0,
        n_targets: int = 1,
    ) -> None:
        super().__init__()
        from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder

        if node_channels % num_heads:
            raise ValueError("node_channels must be divisible by num_heads")
        if shortest_path_cap < 1:
            raise ValueError("shortest_path_cap must be positive")
        self.node_channels = int(node_channels)
        self.pair_channels = int(pair_channels)
        self.shortest_path_cap = int(shortest_path_cap)
        self.atom_encoder = AtomEncoder(node_channels)
        self.bond_encoder = BondEncoder(pair_channels)
        self.in_degree_encoder = nn.Embedding(512, node_channels)
        self.out_degree_encoder = nn.Embedding(512, node_channels)
        self.spatial_encoder = nn.Embedding(shortest_path_cap + 2, pair_channels)
        self.graph_token = nn.Parameter(torch.empty(1, 1, node_channels))
        self.virtual_pair = nn.Parameter(torch.empty(1, pair_channels, 1, 1))
        self.node_norm = nn.LayerNorm(node_channels)
        self.input_dropout = nn.Dropout(dropout)
        path_rates = torch.linspace(0.0, drop_path, num_layers).tolist()
        self.blocks = nn.ModuleList(
            GPTransBlock(
                node_channels,
                pair_channels,
                num_heads,
                dropout,
                path_rates[index],
                layer_scale,
            )
            for index in range(num_layers)
        )
        self.readout = nn.Sequential(
            nn.Linear(node_channels + pair_channels, node_channels),
            nn.LayerNorm(node_channels),
            nn.GELU(),
            nn.Linear(node_channels, n_targets),
        )
        nn.init.normal_(self.graph_token, std=0.02)
        nn.init.normal_(self.virtual_pair, std=0.02)

    @staticmethod
    def _local_edges(
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        node_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = int(batch.max().item()) + 1 if batch.numel() else 0
        counts = torch.bincount(batch, minlength=batch_size)
        offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0)[:-1]))
        local = torch.arange(node_count, device=batch.device) - offsets[batch]
        return batch[edge_index[0]], local[edge_index[0]], local[edge_index[1]]

    def _shortest_path(
        self,
        adjacency: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, max_nodes = adjacency.shape[:2]
        unreachable = self.shortest_path_cap + 1
        distance = torch.full(
            (batch_size, max_nodes, max_nodes),
            unreachable,
            dtype=torch.long,
            device=adjacency.device,
        )
        diagonal = torch.eye(max_nodes, dtype=torch.bool, device=adjacency.device)
        distance.masked_fill_(diagonal.unsqueeze(0) & pair_mask, 0)
        frontier = adjacency
        adjacency_float = adjacency.float()
        for step in range(1, self.shortest_path_cap + 1):
            distance.masked_fill_(frontier & (distance == unreachable) & pair_mask, step)
            frontier = torch.bmm(frontier.float(), adjacency_float) > 0
        return distance

    def _dense_inputs(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dense_x, node_mask = to_dense_batch(x.long(), batch)
        batch_size, max_nodes = dense_x.shape[:2]
        embedded_nodes, embedded_mask = to_dense_batch(self.atom_encoder(x.long()), batch)
        if not torch.equal(node_mask, embedded_mask):
            raise RuntimeError("categorical and embedded node masks differ")
        edge_batch, edge_src, edge_dst = self._local_edges(
            edge_index, batch, int(x.shape[0])
        )
        adjacency = torch.zeros(
            (batch_size, max_nodes, max_nodes),
            dtype=torch.bool,
            device=x.device,
        )
        adjacency[edge_batch, edge_src, edge_dst] = True
        degree = adjacency.sum(-1).clamp_max(511)
        embedded_nodes = (
            embedded_nodes
            + self.in_degree_encoder(degree)
            + self.out_degree_encoder(degree)
        )
        token = self.graph_token.expand(batch_size, -1, -1)
        node = self.input_dropout(self.node_norm(torch.cat((token, embedded_nodes), dim=1)))

        pair_mask = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
        spatial = self._shortest_path(adjacency, pair_mask)
        pair = self.spatial_encoder(spatial).permute(0, 3, 1, 2)
        bond = self.bond_encoder(edge_attr.long())
        pair[edge_batch, :, edge_src, edge_dst] += bond
        full_pair = pair.new_zeros(
            batch_size, self.pair_channels, max_nodes + 1, max_nodes + 1
        )
        full_pair[:, :, 1:, 1:] = pair
        full_pair[:, :, 0:1, :] += self.virtual_pair
        full_pair[:, :, 1:, 0:1] += self.virtual_pair
        full_node_mask = torch.cat(
            (
                torch.ones(batch_size, 1, dtype=torch.bool, device=x.device),
                node_mask,
            ),
            dim=1,
        )
        key_padding_mask = ~full_node_mask[:, None, None, :]
        return node, full_pair, key_padding_mask

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
        random_walk_pe: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del random_walk_pe
        node, pair, key_padding_mask = self._dense_inputs(
            x, edge_index, edge_attr, batch
        )
        for block in self.blocks:
            node, pair = block(node, pair, key_padding_mask)
        graph_state = torch.cat((node[:, 0], pair[:, :, 0, 0]), dim=-1)
        return self.readout(graph_state)
