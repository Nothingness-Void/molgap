"""Pure-2D Pair-GPS encoder for the MolGap architecture screen and target run.

The encoder combines the portable architectural signals from the top-ranked
graph models without importing a conformer or another model's prediction:

* a persistent all-pairs state carries long-range 2D topology;
* a low-rank triplet update lets pair states sharing an atom communicate;
* pair-to-node and bond-local pair messages provide an explicit edge-to-node
  path;
* the same pair state biases the node global-attention path.

The head is a direct HOMO/LUMO/Gap regressor.  There is no target residual,
checkpoint warm start, prediction fusion, or 3D input in this module.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch_geometric.nn import GINEConv


class _PairGPSBlock(nn.Module):
    """One node/pair block with explicit bidirectional state exchange."""

    def __init__(
        self,
        hidden_channels: int,
        pair_channels: int,
        num_heads: int,
        dropout: float,
        triplet_rank: int,
    ) -> None:
        super().__init__()
        if hidden_channels % num_heads:
            raise ValueError("hidden_channels must be divisible by num_heads")

        self.num_heads = int(num_heads)
        self.head_channels = hidden_channels // num_heads
        self.dropout = nn.Dropout(dropout)

        self.node_attn_norm = nn.LayerNorm(hidden_channels)
        self.qkv = nn.Linear(hidden_channels, hidden_channels * 3)
        self.attn_out = nn.Linear(hidden_channels, hidden_channels)
        self.pair_bias = nn.Linear(pair_channels, num_heads)

        self.pair_to_node = nn.Linear(pair_channels, hidden_channels)
        self.bond_pair_to_node = nn.Linear(pair_channels, hidden_channels)
        self.edge_state_to_hidden = nn.Linear(pair_channels, hidden_channels)
        self.local_gine = GINEConv(
            nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.SiLU(),
                nn.Linear(hidden_channels, hidden_channels),
            ),
            edge_dim=hidden_channels,
        )
        self.local_out = nn.Linear(hidden_channels, hidden_channels)
        self.node_ffn_norm = nn.LayerNorm(hidden_channels)
        self.node_ffn = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels * 4, hidden_channels),
        )

        self.node_to_pair = nn.Linear(hidden_channels * 3, pair_channels)
        self.triplet_left = nn.Linear(pair_channels, triplet_rank, bias=False)
        self.triplet_right = nn.Linear(pair_channels, triplet_rank, bias=False)
        self.triplet_out = nn.Linear(triplet_rank, pair_channels)
        self.pair_norm = nn.LayerNorm(pair_channels)
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
        edge_index: torch.Tensor,
        edge_batch: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = self.node_attn_norm(node)
        qkv = self.qkv(normalized).view(
            normalized.shape[0],
            normalized.shape[1],
            3,
            self.num_heads,
            self.head_channels,
        )
        query, key, value = qkv.unbind(dim=2)
        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)
        scores = torch.matmul(query, key.transpose(-1, -2))
        scores = scores / math.sqrt(self.head_channels)
        scores = scores + self.pair_bias(pair).permute(0, 3, 1, 2)
        scores = scores.masked_fill(
            ~node_mask[:, None, None, :], torch.finfo(scores.dtype).min
        )
        attention = torch.softmax(scores, dim=-1)
        attended = torch.matmul(attention, value)
        attended = attended.transpose(1, 2).reshape_as(node)

        # Every target node receives a message from its all-pairs row.  The
        # bond-only path is kept separate so long-range pair context cannot
        # erase the chemically exact local bond channel.
        pair_message = self.pair_to_node(pair)
        pair_message = pair_message * pair_mask.unsqueeze(-1)
        pair_degree = pair_mask.sum(dim=2, keepdim=True).clamp_min(1.0)
        pair_message = pair_message.sum(dim=2) / pair_degree

        local_message = self.bond_pair_to_node(pair)
        local_message = local_message * bond_mask.unsqueeze(-1)
        bond_degree = bond_mask.sum(dim=2, keepdim=True).clamp_min(1.0)
        local_message = local_message.sum(dim=2) / bond_degree
        local_message = self.local_out(local_message)

        # Recover the current pair state on each actual bond and run the
        # standard GPS/GINE local branch over the original sparse graph.  This
        # preserves exact bond-local propagation while the dense pair path
        # supplies long-range and triplet context.
        flat_node = node[node_mask]
        bond_pair_state = pair[edge_batch, edge_src, edge_dst]
        bond_edge_state = self.edge_state_to_hidden(bond_pair_state)
        flat_local = self.local_gine(flat_node, edge_index, bond_edge_state)
        local_gine = node.new_zeros(node.shape)
        local_gine[node_mask] = flat_local.to(dtype=node.dtype)

        node = node + self.dropout(self.attn_out(attended))
        node = node + self.dropout(pair_message)
        node = node + self.dropout(local_message)
        node = node + self.dropout(local_gine)
        node = node * node_mask.unsqueeze(-1)
        node = node + self.dropout(self.node_ffn(self.node_ffn_norm(node)))
        node = node * node_mask.unsqueeze(-1)

        node_i = node.unsqueeze(2)
        node_j = node.unsqueeze(1)
        node_pair = self.node_to_pair(
            torch.cat(
                (node_i + node_j, (node_i - node_j).abs(), node_i * node_j),
                dim=-1,
            )
        )
        left = self.triplet_left(pair)
        right = self.triplet_right(pair)
        triplet = torch.einsum("bikd,bkjd->bijd", left, right)
        triplet = self.triplet_out(triplet / math.sqrt(left.shape[-1]))
        pair_input = torch.cat((pair, triplet, node_pair), dim=-1)
        pair = self.pair_norm(pair + self.pair_update(pair_input))
        pair = pair * pair_mask.unsqueeze(-1)
        return node, pair


class PairGPS2DWrapper(nn.Module):
    """Single pure-2D pair/triplet/GPS encoder with a direct prediction head."""

    def __init__(
        self,
        in_channels: int = 11,
        edge_dim: int = 4,
        hidden_channels: int = 192,
        pair_channels: int = 64,
        num_layers: int = 8,
        num_heads: int = 4,
        dropout: float = 0.05,
        n_targets: int = 3,
        pooling: str = "mean",
        path_steps: int = 5,
        triplet_rank: int = 16,
    ) -> None:
        super().__init__()
        if pooling not in {"mean", "mean_max"}:
            raise ValueError(f"Unsupported pooling: {pooling}")
        if path_steps < 1:
            raise ValueError("path_steps must be positive")
        self.pooling = pooling
        self.edge_dim = int(edge_dim)
        self.in_channels = int(in_channels)
        self.path_steps = int(path_steps)

        self.node_input = nn.Linear(in_channels, hidden_channels)
        pair_input_dim = edge_dim + 2 * path_steps + 3 * in_channels + 3
        self.pair_input = nn.Sequential(
            nn.Linear(pair_input_dim, pair_channels),
            nn.SiLU(),
            nn.Linear(pair_channels, pair_channels),
        )
        self.layers = nn.ModuleList(
            [
                _PairGPSBlock(
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
    def _local_edge_index(
        edge_index: torch.Tensor, batch: torch.Tensor, node_count: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = int(batch.max().item()) + 1 if batch.numel() else 0
        counts = torch.bincount(batch, minlength=batch_size)
        offsets = torch.cat((counts.new_zeros(1), counts.cumsum(dim=0)[:-1]))
        local = torch.arange(node_count, device=batch.device) - offsets[batch]
        return batch[edge_index[0]], local[edge_index[0]], local[edge_index[1]]

    def _dense_inputs(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        from torch_geometric.utils import to_dense_batch

        node, node_mask = to_dense_batch(x.float(), batch)
        batch_size, max_nodes = node.shape[:2]
        dense_edges = node.new_zeros((batch_size, max_nodes, max_nodes, self.edge_dim))
        edge_batch, edge_src, edge_dst = self._local_edge_index(
            edge_index, batch, int(x.shape[0])
        )
        dense_edges.index_put_(
            (edge_batch, edge_src, edge_dst),
            edge_attr.float().to(dtype=dense_edges.dtype),
            accumulate=True,
        )
        bond = node.new_zeros((batch_size, max_nodes, max_nodes))
        bond.index_put_(
            (edge_batch, edge_src, edge_dst),
            torch.ones(edge_batch.shape[0], device=x.device, dtype=bond.dtype),
            accumulate=True,
        )
        bond = (bond > 0).to(dtype=node.dtype)

        reach = bond
        path_reach = [reach]
        path_count = [torch.log1p(reach)]
        for _ in range(1, self.path_steps):
            reach = (torch.bmm(reach, bond) > 0).to(dtype=bond.dtype)
            path_reach.append(reach)
            count = torch.bmm(torch.expm1(path_count[-1]), bond).clamp(max=32.0)
            path_count.append(torch.log1p(count))
        path_features = torch.cat(
            (
                torch.stack(path_reach, dim=-1),
                torch.stack(path_count, dim=-1),
            ),
            dim=-1,
        )

        node_i = node.unsqueeze(2)
        node_j = node.unsqueeze(1)
        node_pair = torch.cat(
            (node_i + node_j, (node_i - node_j).abs(), node_i * node_j),
            dim=-1,
        )
        degree = bond.sum(dim=-1, keepdim=True)
        degree_i = degree.unsqueeze(2)
        degree_j = degree.unsqueeze(1)
        degree_pair = torch.cat(
            (degree_i + degree_j, (degree_i - degree_j).abs(), degree_i * degree_j),
            dim=-1,
        )
        pair_features = torch.cat(
            (dense_edges, path_features, node_pair, degree_pair), dim=-1
        )
        pair_mask = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
        pair = self.pair_input(pair_features) * pair_mask.unsqueeze(-1)
        return node, pair, bond, node_mask, pair_mask, edge_batch, edge_src, edge_dst

    def encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        (
            node,
            pair,
            bond,
            node_mask,
            pair_mask,
            edge_batch,
            edge_src,
            edge_dst,
        ) = self._dense_inputs(
            x, edge_index, edge_attr, batch
        )
        node = self.node_input(node) * node_mask.unsqueeze(-1)
        for layer in self.layers:
            node, pair = layer(
                node,
                pair,
                bond,
                node_mask,
                pair_mask,
                edge_index,
                edge_batch,
                edge_src,
                edge_dst,
            )

        masked = node * node_mask.unsqueeze(-1)
        mean = masked.sum(dim=1) / node_mask.sum(dim=1, keepdim=True).clamp_min(1)
        if self.pooling == "mean":
            return mean
        maximum = masked.masked_fill(
            ~node_mask.unsqueeze(-1), torch.finfo(masked.dtype).min
        ).amax(dim=1)
        return self.pool_proj(torch.cat((mean, maximum), dim=-1))

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        return self.head(self.encode(x, edge_index, edge_attr, batch))
