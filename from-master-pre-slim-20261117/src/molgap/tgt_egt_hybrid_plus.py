"""Residual 2D expert on top of the measured ETKDG TGT/EGT hybrid.

The preceding rich hybrid fused two molecule embeddings with a convex gate.
Held-out payload analysis showed that a prediction-level blend was better, so
this wrapper keeps the measured hybrid prediction as the identity path and
lets a conformer-free pair/triplet expert learn only a bounded residual.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .pair_triplet_2d_rich import PairTriplet2DRichWrapper
from .tgt_egt_hybrid import TGTEGTHybridWrapper, _counts_to_batch, _offset_edges


class _ResidualHead(nn.Module):
    def __init__(self, hidden_channels: int, dropout: float, n_targets: int):
        super().__init__()
        self.hidden_channels = int(hidden_channels)
        self.base_targets = n_targets
        self.correction = nn.Sequential(
            nn.LayerNorm(hidden_channels * 2),
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.SiLU(),
            nn.Linear(hidden_channels // 2, n_targets),
        )
        # At initialization the candidate is exactly the warm-started hybrid;
        # the 2D branch earns its influence through supervised residuals.
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        base = embedding[..., : self.hidden_channels]
        expert = embedding[..., self.hidden_channels : self.hidden_channels * 2]
        base_prediction = embedding[..., self.hidden_channels * 2 :]
        correction = self.correction(torch.cat((base, expert), dim=-1))
        return base_prediction + correction


class TGTEGTHybridPlusWrapper(nn.Module):
    """Measured ETKDG hybrid plus a conformer-free rich pair/triplet residual."""

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
        expert_hidden_channels: int = 192,
        expert_pair_channels: int = 64,
        expert_layers: int = 6,
        expert_heads: int = 4,
        expert_path_steps: int = 5,
        expert_triplet_rank: int = 8,
        n_targets: int = 3,
        learning_rate: float | None = None,
        amp: bool | None = None,
        freeze_base: bool = False,
    ):
        super().__init__()
        del learning_rate, amp
        if expert_hidden_channels != hidden_channels:
            raise ValueError("expert_hidden_channels must equal hidden_channels")
        if expert_heads != num_heads:
            raise ValueError("expert_heads must equal num_heads")

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
        self.expert.head = nn.Identity()
        self.head = _ResidualHead(hidden_channels, dropout, n_targets)

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
        return torch.cat((base, expert, base_prediction), dim=-1)

    def forward(self, x, edge_index, edge_attr, z, pos, batch, **kwargs):
        return self.head(self.encode(x, edge_index, edge_attr, z, pos, batch, **kwargs))
