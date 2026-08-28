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
from torch_geometric.utils import to_dense_batch

from .gps import CategoricalFeatureEncoder, FrontierCenterGapHead


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


class _PairGPSR2Block(nn.Module):
    """A balanced GPS block with one local path and a gated pair update."""

    def __init__(
        self,
        hidden_channels: int,
        pair_channels: int,
        num_heads: int,
        dropout: float,
        triplet_rank: int,
        *,
        use_triplet: bool,
        gate_init: float,
        triplet_attention: bool = False,
        pre_norm_pair_residual: bool = False,
    ) -> None:
        super().__init__()
        if hidden_channels % num_heads:
            raise ValueError("hidden_channels must be divisible by num_heads")
        if not 0.0 < gate_init < 1.0:
            raise ValueError("gate_init must lie strictly between zero and one")

        self.num_heads = int(num_heads)
        self.head_channels = hidden_channels // num_heads
        self.use_triplet = bool(use_triplet)
        self.triplet_attention = bool(triplet_attention)
        self.pre_norm_pair_residual = bool(pre_norm_pair_residual)
        self.dropout = nn.Dropout(dropout)
        gate_logit = math.log(gate_init / (1.0 - gate_init))

        self.attention_input_norm = nn.LayerNorm(hidden_channels)
        self.qkv = nn.Linear(hidden_channels, hidden_channels * 3)
        self.pair_bias = nn.Linear(pair_channels, num_heads)
        self.attention_out = nn.Linear(hidden_channels, hidden_channels)
        self.global_update_norm = nn.LayerNorm(hidden_channels)
        self.global_gate_logit = nn.Parameter(torch.tensor(gate_logit))

        # There is exactly one bond-local node path.  GINEConv performs its
        # own pair_channels -> hidden_channels edge projection, avoiding the
        # duplicate bond-average and edge-projection branches in R1.
        self.local_input_norm = nn.LayerNorm(hidden_channels)
        self.local_gine = GINEConv(
            nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.SiLU(),
                nn.Linear(hidden_channels, hidden_channels),
            ),
            edge_dim=pair_channels,
        )
        self.local_update_norm = nn.LayerNorm(hidden_channels)
        self.local_gate_logit = nn.Parameter(torch.tensor(gate_logit))

        self.node_ffn_input_norm = nn.LayerNorm(hidden_channels)
        self.node_ffn = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels * 2, hidden_channels),
        )
        self.node_output_norm = nn.LayerNorm(hidden_channels)

        self.pair_input_norm = nn.LayerNorm(pair_channels)
        self.node_to_pair = nn.Sequential(
            nn.Linear(hidden_channels * 3, pair_channels * 2),
            nn.SiLU(),
            nn.Linear(pair_channels * 2, pair_channels),
        )
        if self.use_triplet:
            self.triplet_left = nn.Linear(pair_channels, triplet_rank)
            self.triplet_right = nn.Linear(pair_channels, triplet_rank)
            self.triplet_left_gate = nn.Linear(pair_channels, triplet_rank)
            self.triplet_right_gate = nn.Linear(pair_channels, triplet_rank)
            self.triplet_norm = nn.LayerNorm(triplet_rank)
            self.triplet_out = nn.Linear(triplet_rank, pair_channels)
            self.triplet_out_gate = nn.Linear(pair_channels, pair_channels)
            if self.triplet_attention:
                self.triplet_left_score = nn.Linear(pair_channels, 1)
                self.triplet_right_score = nn.Linear(pair_channels, 1)
            pair_update_channels = pair_channels * 3
        else:
            pair_update_channels = pair_channels * 2
        self.pair_update = nn.Sequential(
            nn.Linear(pair_update_channels, pair_channels * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(pair_channels * 2, pair_channels),
        )
        self.pair_update_norm = nn.LayerNorm(pair_channels)
        self.pair_gate_logit = nn.Parameter(torch.tensor(gate_logit))
        self.pair_output_norm = (
            nn.Identity()
            if self.pre_norm_pair_residual
            else nn.LayerNorm(pair_channels)
        )

    @staticmethod
    def _gate(logit: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(logit)

    def forward(
        self,
        node: torch.Tensor,
        pair: torch.Tensor,
        node_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        edge_index: torch.Tensor,
        edge_batch: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = self.attention_input_norm(node)
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
        global_update = self.global_update_norm(self.attention_out(attended))

        flat_node = self.local_input_norm(node)[node_mask]
        bond_pair_state = pair[edge_batch, edge_src, edge_dst]
        flat_local = self.local_gine(flat_node, edge_index, bond_pair_state)
        local_update = node.new_zeros(node.shape)
        local_update[node_mask] = flat_local.to(dtype=node.dtype)
        local_update = self.local_update_norm(local_update)

        node = node + self.dropout(
            self._gate(self.global_gate_logit) * global_update
        )
        node = node + self.dropout(
            self._gate(self.local_gate_logit) * local_update
        )
        node = node * node_mask.unsqueeze(-1)
        node = node + self.dropout(self.node_ffn(self.node_ffn_input_norm(node)))
        node = self.node_output_norm(node) * node_mask.unsqueeze(-1)

        pair_normalized = self.pair_input_norm(pair)
        node_i = node.unsqueeze(2)
        node_j = node.unsqueeze(1)
        node_pair = self.node_to_pair(
            torch.cat(
                (node_i + node_j, (node_i - node_j).abs(), node_i * node_j),
                dim=-1,
            )
        )
        pair_inputs = [pair_normalized, node_pair]
        if self.use_triplet:
            mask = pair_mask.unsqueeze(-1)
            left = self.triplet_left(pair_normalized)
            right = self.triplet_right(pair_normalized)
            left = left * torch.sigmoid(self.triplet_left_gate(pair_normalized))
            right = right * torch.sigmoid(self.triplet_right_gate(pair_normalized))
            left = left * mask
            right = right * mask
            if self.triplet_attention:
                left_score = self.triplet_left_score(pair_normalized).squeeze(-1)
                right_score = self.triplet_right_score(pair_normalized).squeeze(-1)
                logits = left_score.unsqueeze(3) + right_score.unsqueeze(1)
                valid = pair_mask.unsqueeze(3) & pair_mask.unsqueeze(1)
                logits = logits.masked_fill(
                    ~valid, torch.finfo(logits.dtype).min
                )
                weights = torch.softmax(logits, dim=2) * valid
                triplet = torch.einsum(
                    "bikj,bikd,bkjd->bijd", weights, left, right
                )
            else:
                triplet = torch.einsum("bikd,bkjd->bijd", left, right)
                valid_intermediates = torch.einsum(
                    "bik,bkj->bij", pair_mask.float(), pair_mask.float()
                ).clamp_min(1.0)
                triplet = triplet / valid_intermediates.unsqueeze(-1)
            triplet = self.triplet_out(self.triplet_norm(triplet))
            triplet = triplet * torch.sigmoid(
                self.triplet_out_gate(pair_normalized)
            )
            pair_inputs.append(triplet)
        pair_update = self.pair_update(torch.cat(pair_inputs, dim=-1))
        pair_update = self.pair_update_norm(pair_update)
        pair = pair + self.dropout(self._gate(self.pair_gate_logit) * pair_update)
        pair = self.pair_output_norm(pair) * pair_mask.unsqueeze(-1)
        return node, pair


class PairGPS2DR2Wrapper(nn.Module):
    """RWSE-guided, compute-bounded repair of :class:`PairGPS2DWrapper`.

    R2 keeps the useful persistent all-pairs state and attention bias, but it
    replaces adjacency-power features with first-hit shortest-path buckets,
    removes direct all-pair/bond-average node branches, and evaluates the
    size-normalized triplet update only every ``triplet_interval`` layers.
    """

    def __init__(
        self,
        in_channels: int = 11,
        edge_dim: int = 4,
        hidden_channels: int = 192,
        pair_channels: int = 64,
        num_layers: int = 9,
        num_heads: int = 4,
        dropout: float = 0.05,
        n_targets: int = 3,
        pooling: str = "mean",
        distance_cap: int = 5,
        triplet_rank: int = 8,
        triplet_interval: int = 3,
        rwse_dim: int = 16,
        gate_init: float = 0.1,
        triplet_attention: bool = False,
        pre_norm_pair_residual: bool = False,
    ) -> None:
        super().__init__()
        if pooling not in {"mean", "mean_max"}:
            raise ValueError(f"Unsupported pooling: {pooling}")
        if distance_cap < 1:
            raise ValueError("distance_cap must be positive")
        if triplet_interval < 1:
            raise ValueError("triplet_interval must be positive")
        self.pooling = pooling
        self.edge_dim = int(edge_dim)
        self.in_channels = int(in_channels)
        self.distance_cap = int(distance_cap)
        self.rwse_dim = int(rwse_dim)

        self.node_input = nn.Linear(in_channels, hidden_channels)
        self.rwse_encoder = nn.Sequential(
            nn.LayerNorm(rwse_dim),
            nn.Linear(rwse_dim, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.node_input_norm = nn.LayerNorm(hidden_channels)
        pair_input_dim = edge_dim + 3 * in_channels + 3
        self.pair_input = nn.Sequential(
            nn.Linear(pair_input_dim, pair_channels),
            nn.SiLU(),
            nn.Linear(pair_channels, pair_channels),
        )
        # distance_cap + 1 is the overflow/unreachable bucket.
        self.distance_embedding = nn.Embedding(distance_cap + 2, pair_channels)
        self.pair_input_norm = nn.LayerNorm(pair_channels)
        self.layers = nn.ModuleList(
            [
                _PairGPSR2Block(
                    hidden_channels,
                    pair_channels,
                    num_heads,
                    dropout,
                    triplet_rank,
                    use_triplet=(layer_index + 1) % triplet_interval == 0,
                    gate_init=gate_init,
                    triplet_attention=triplet_attention,
                    pre_norm_pair_residual=pre_norm_pair_residual,
                )
                for layer_index in range(num_layers)
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
        return PairGPS2DWrapper._local_edge_index(edge_index, batch, node_count)

    def _shortest_path_buckets(
        self, bond: torch.Tensor, pair_mask: torch.Tensor
    ) -> torch.Tensor:
        batch_size, max_nodes = bond.shape[:2]
        overflow = self.distance_cap + 1
        distances = torch.full(
            (batch_size, max_nodes, max_nodes),
            overflow,
            dtype=torch.long,
            device=bond.device,
        )
        diagonal = torch.eye(max_nodes, dtype=torch.bool, device=bond.device)
        diagonal = diagonal.unsqueeze(0) & pair_mask
        distances.masked_fill_(diagonal, 0)
        frontier = bond.bool()
        adjacency = bond.float()
        for distance in range(1, self.distance_cap + 1):
            unseen = distances == overflow
            distances.masked_fill_(frontier & unseen & pair_mask, distance)
            frontier = torch.bmm(frontier.float(), adjacency) > 0
        return distances

    def _dense_inputs(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
    ):
        raw_node, node_mask = to_dense_batch(x.float(), batch)
        batch_size, max_nodes = raw_node.shape[:2]
        dense_edges = raw_node.new_zeros(
            (batch_size, max_nodes, max_nodes, self.edge_dim)
        )
        edge_batch, edge_src, edge_dst = self._local_edge_index(
            edge_index, batch, int(x.shape[0])
        )
        dense_edges.index_put_(
            (edge_batch, edge_src, edge_dst),
            edge_attr.float().to(dtype=dense_edges.dtype),
            accumulate=True,
        )
        bond = raw_node.new_zeros((batch_size, max_nodes, max_nodes))
        bond.index_put_(
            (edge_batch, edge_src, edge_dst),
            torch.ones(edge_batch.shape[0], device=x.device, dtype=bond.dtype),
            accumulate=True,
        )
        bond = (bond > 0).to(dtype=raw_node.dtype)
        pair_mask = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
        distances = self._shortest_path_buckets(bond, pair_mask)

        node_i = raw_node.unsqueeze(2)
        node_j = raw_node.unsqueeze(1)
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
        pair_features = torch.cat((dense_edges, node_pair, degree_pair), dim=-1)
        pair = self.pair_input(pair_features) + self.distance_embedding(distances)
        pair = self.pair_input_norm(pair) * pair_mask.unsqueeze(-1)
        return (
            raw_node,
            pair,
            node_mask,
            pair_mask,
            edge_batch,
            edge_src,
            edge_dst,
        )

    def encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
        random_walk_pe: torch.Tensor,
    ) -> torch.Tensor:
        if random_walk_pe is None:
            raise ValueError("PairGPS2D R2 requires random_walk_pe")
        expected = (x.shape[0], self.rwse_dim)
        if random_walk_pe.ndim != 2 or tuple(random_walk_pe.shape) != expected:
            raise ValueError(
                f"random_walk_pe must have shape {expected}, "
                f"got {tuple(random_walk_pe.shape)}"
            )
        if not torch.isfinite(random_walk_pe).all():
            raise ValueError("random_walk_pe contains non-finite values")
        (
            raw_node,
            pair,
            node_mask,
            pair_mask,
            edge_batch,
            edge_src,
            edge_dst,
        ) = self._dense_inputs(x, edge_index, edge_attr, batch)
        dense_rwse, rwse_mask = to_dense_batch(random_walk_pe.float(), batch)
        if not torch.equal(node_mask, rwse_mask):
            raise ValueError("RWSE and atom batches have different node masks")
        node = self.node_input(raw_node) + self.rwse_encoder(dense_rwse)
        node = self.node_input_norm(node) * node_mask.unsqueeze(-1)
        for layer in self.layers:
            node, pair = layer(
                node,
                pair,
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
        random_walk_pe: torch.Tensor,
    ) -> torch.Tensor:
        return self.head(
            self.encode(x, edge_index, edge_attr, batch, random_walk_pe)
        )


class PairGPS2DR3Wrapper(PairGPS2DR2Wrapper):
    """R2 backbone with bounded relational and/or frontier-head repairs."""

    def __init__(
        self,
        *args,
        attentive_triplet: bool = False,
        consistent_head: bool = False,
        **kwargs,
    ) -> None:
        n_targets = int(kwargs.get("n_targets", 3))
        if n_targets != 3:
            raise ValueError("PairGPS2D R3 requires three frontier targets")
        super().__init__(
            *args,
            triplet_attention=attentive_triplet,
            pre_norm_pair_residual=attentive_triplet,
            **kwargs,
        )
        self.attentive_triplet = bool(attentive_triplet)
        self.consistent_head = bool(consistent_head)
        if self.consistent_head:
            embedding_dim = self.node_input.out_features
            dropout = next(
                module.p for module in self.head if isinstance(module, nn.Dropout)
            )
            self.head = FrontierCenterGapHead(
                embedding_dim,
                hidden_channels=embedding_dim,
                dropout=dropout,
            )


class CategoricalPairGPS2DWrapper(PairGPS2DWrapper):
    """PairGPS adapter for OGB categorical atom/bond fields plus RWSE."""

    def __init__(
        self,
        *,
        atom_feature_dims,
        bond_feature_dims,
        atom_input_channels: int = 64,
        bond_input_channels: int = 32,
        rwse_dim: int = 16,
        **kwargs,
    ) -> None:
        if atom_input_channels <= 0 or bond_input_channels <= 0 or rwse_dim <= 0:
            raise ValueError("categorical and RWSE dimensions must be positive")
        super().__init__(
            in_channels=atom_input_channels,
            edge_dim=bond_input_channels,
            **kwargs,
        )
        self.rwse_dim = int(rwse_dim)
        self.atom_encoder = CategoricalFeatureEncoder(
            atom_feature_dims, atom_input_channels
        )
        self.bond_encoder = CategoricalFeatureEncoder(
            bond_feature_dims, bond_input_channels
        )
        self.rwse_encoder = nn.Sequential(
            nn.Linear(self.rwse_dim, atom_input_channels),
            nn.SiLU(),
            nn.Linear(atom_input_channels, atom_input_channels),
        )

    def encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
        random_walk_pe: torch.Tensor,
    ) -> torch.Tensor:
        if random_walk_pe.ndim != 2 or random_walk_pe.shape != (
            x.shape[0],
            self.rwse_dim,
        ):
            raise ValueError(
                "random_walk_pe must have shape "
                f"[{x.shape[0]}, {self.rwse_dim}]"
            )
        if not torch.isfinite(random_walk_pe).all():
            raise ValueError("random_walk_pe contains non-finite values")
        atom = self.atom_encoder(x) + self.rwse_encoder(random_walk_pe.float())
        bond = self.bond_encoder(edge_attr)
        return super().encode(atom, edge_index, bond, batch)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
        random_walk_pe: torch.Tensor,
    ) -> torch.Tensor:
        return self.head(
            self.encode(x, edge_index, edge_attr, batch, random_walk_pe)
        )
