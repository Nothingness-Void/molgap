"""Warm-started two-expert blend for the QM9 route-C screen.

This candidate keeps the measured ETKDG hybrid and the already-trained rich
pure-2D expert as separate identity paths.  A target-specific blend is
initialized from the held-out payload analysis, while a zero-initialized
residual head can learn corrections from both embeddings and both predictions.
The initialization avoids asking a new residual expert to rediscover a useful
representation from scratch.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .pair_triplet_2d_rich import PairTriplet2DRichWrapper
from .tgt_egt_hybrid import TGTEGTHybridWrapper, _counts_to_batch, _offset_edges


class _WarmBlendHead(nn.Module):
    def __init__(
        self,
        base_hidden_channels: int,
        expert_hidden_channels: int,
        n_targets: int,
        dropout: float,
        initial_base_weight: tuple[float, ...] = (0.567, 0.567, 0.567),
    ):
        super().__init__()
        if len(initial_base_weight) != n_targets:
            raise ValueError("initial_base_weight must match n_targets")
        self.base_hidden_channels = int(base_hidden_channels)
        self.expert_hidden_channels = int(expert_hidden_channels)
        weights = torch.tensor(initial_base_weight, dtype=torch.float32).clamp(
            1e-4, 1.0 - 1e-4
        )
        self.blend_logits = nn.Parameter(torch.logit(weights))
        input_channels = (
            base_hidden_channels + expert_hidden_channels + n_targets * 2
        )
        self.correction = nn.Sequential(
            nn.LayerNorm(input_channels),
            nn.Linear(input_channels, base_hidden_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(base_hidden_channels, base_hidden_channels // 2),
            nn.SiLU(),
            nn.Linear(base_hidden_channels // 2, n_targets),
        )
        # The first forward pass is the measured validation-selected blend.
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        base_hidden = self.base_hidden_channels
        expert_hidden = self.expert_hidden_channels
        base = embedding[..., :base_hidden]
        expert = embedding[..., base_hidden : base_hidden + expert_hidden]
        base_prediction = embedding[
            ..., base_hidden + expert_hidden : base_hidden + expert_hidden + 3
        ]
        expert_prediction = embedding[..., base_hidden + expert_hidden + 3 :]
        base_weight = torch.sigmoid(self.blend_logits).view(1, -1)
        blended = (
            base_weight * base_prediction
            + (1.0 - base_weight) * expert_prediction
        )
        correction_input = torch.cat(
            (base, expert, base_prediction, expert_prediction), dim=-1
        )
        return blended + self.correction(correction_input)


class TGTEGTHybridWarmBlendWrapper(nn.Module):
    """ETKDG hybrid plus a warm-started rich pure-2D prediction expert."""

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
        expert_hidden_channels: int = 256,
        expert_pair_channels: int = 96,
        expert_layers: int = 10,
        expert_heads: int = 8,
        expert_path_steps: int = 5,
        expert_triplet_rank: int = 16,
        n_targets: int = 3,
        learning_rate: float | None = None,
        amp: bool | None = None,
        freeze_base: bool = False,
        freeze_expert: bool = False,
    ):
        super().__init__()
        del learning_rate, amp
        self.hidden_channels = int(hidden_channels)
        self.base = TGTEGTHybridWrapper(
            in_channels=in_channels,
            edge_dim=edge_dim,
            hidden_channels=hidden_channels,
            pair_channels=pair_channels,
            num_layers=num_layers,
            num_heads=num_heads,
            num_rbf=num_rbf,
            cutoff=cutoff,
            dropout=dropout,
            topology_layers=topology_layers,
            n_targets=n_targets,
        )
        if freeze_base:
            for parameter in self.base.parameters():
                parameter.requires_grad_(False)
        self.expert = PairTriplet2DRichWrapper(
            in_channels=in_channels,
            edge_dim=edge_dim,
            hidden_channels=expert_hidden_channels,
            pair_channels=expert_pair_channels,
            num_layers=expert_layers,
            num_heads=expert_heads,
            dropout=dropout,
            n_targets=n_targets,
            pooling="mean_max",
            path_steps=expert_path_steps,
            triplet_rank=expert_triplet_rank,
        )
        if freeze_expert:
            for parameter in self.expert.parameters():
                parameter.requires_grad_(False)
        self.head = _WarmBlendHead(
            hidden_channels,
            expert_hidden_channels,
            n_targets,
            dropout,
        )

    def _expert_encode(
        self,
        topology_x,
        topology_edges,
        topology_edge_attr,
        topology_node_counts,
        topology_edge_counts,
    ):
        topology_batch = _counts_to_batch(topology_node_counts)
        edge_index = _offset_edges(
            topology_edges,
            topology_node_counts,
            topology_edge_counts,
        ).contiguous()
        return self.expert.encode(
            topology_x.float(),
            edge_index,
            topology_edge_attr.float(),
            topology_batch,
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
        if topology_x is None:
            topology_x = x
        if topology_edges is None:
            topology_edges = edge_index.t().contiguous()
        if topology_edge_attr is None:
            topology_edge_attr = edge_attr
        if topology_node_counts is None:
            topology_node_counts = torch.bincount(batch)
        if topology_edge_counts is None:
            topology_edge_counts = torch.bincount(
                torch.zeros(
                    topology_edges.shape[0],
                    dtype=torch.long,
                    device=topology_edges.device,
                ),
                minlength=topology_node_counts.numel(),
            )
        if geometry_node_counts is None:
            geometry_node_counts = torch.bincount(batch)

        base = self.base.encode(
            x,
            edge_index,
            edge_attr,
            z,
            pos,
            batch,
            topology_x=topology_x,
            topology_edges=topology_edges,
            topology_edge_attr=topology_edge_attr,
            topology_node_counts=topology_node_counts,
            topology_edge_counts=topology_edge_counts,
            geometry_node_counts=geometry_node_counts,
        )
        base_prediction = self.base.head(base)
        expert = self._expert_encode(
            topology_x,
            topology_edges,
            topology_edge_attr,
            topology_node_counts,
            topology_edge_counts,
        )
        expert_prediction = self.expert.head(expert)
        return torch.cat(
            (base, expert, base_prediction, expert_prediction), dim=-1
        )

    def forward(self, x, edge_index, edge_attr, z, pos, batch, **kwargs):
        return self.head(self.encode(x, edge_index, edge_attr, z, pos, batch, **kwargs))
