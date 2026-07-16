"""From-scratch GINE/GPS/SchNet heterogeneous soft MoE."""

from __future__ import annotations

import torch
import torch.nn as nn

from molgap.gps import GPSWrapper
from molgap.schnet import SchNetWrapper

from molgap.dual2d_static_candidate.local_gine import LocalGINEExpert
from .target_router import TargetSpecificRouter


class HeterogeneousMoE(nn.Module):
    """Run all experts and combine each target with molecule-specific weights."""

    def __init__(
        self,
        *,
        hidden: int = 192,
        n_descriptors: int = 0,
        use_residual: bool = False,
        expert_dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.use_residual = use_residual
        self.local = LocalGINEExpert(hidden_channels=hidden)
        self.global_expert = GPSWrapper(
            hidden_channels=hidden,
            num_layers=9,
            num_heads=4,
            dropout=0.05,
            pooling="mean_max",
        )
        self.geometry = SchNetWrapper(
            hidden_channels=hidden,
            num_filters=hidden,
            num_interactions=6,
            num_gaussians=50,
            cutoff=6.0,
            dropout=0.05,
        )
        self.router = TargetSpecificRouter(
            embedding_dim=hidden,
            n_descriptors=n_descriptors,
            expert_dropout=expert_dropout,
        )
        self.residual = nn.Sequential(
            nn.Linear(128 * 3, 192),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(192, 64),
            nn.SiLU(),
            nn.Linear(64, 3),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, batch_2d, batch_3d, descriptors=None, temperature=1.0):
        local_z = self.local.encode(
            batch_2d.x, batch_2d.edge_index, batch_2d.edge_attr, batch_2d.batch
        )
        global_z = self.global_expert.encode(
            batch_2d.x, batch_2d.edge_index, batch_2d.edge_attr, batch_2d.batch
        )
        charges = getattr(batch_3d, "charges", None)
        geometry_z = self.geometry.encode(
            batch_3d.z, batch_3d.pos, batch_3d.batch, charges=charges
        )
        expert_predictions = torch.stack([
            self.local.head(local_z),
            self.global_expert.head(global_z),
            self.geometry.head(geometry_z),
        ], dim=-1)
        weights, projected = self.router(
            [local_z, global_z, geometry_z],
            expert_predictions,
            descriptors,
            temperature,
        )
        mixture = torch.sum(weights * expert_predictions, dim=-1)
        residual = torch.zeros_like(mixture)
        if self.use_residual:
            residual = 0.1 * torch.tanh(self.residual(torch.cat(projected, dim=-1)))
        return {
            "prediction": mixture + residual,
            "mixture": mixture,
            "residual": residual,
            "expert_predictions": expert_predictions,
            "router_weights": weights,
            "embeddings": (local_z, global_z, geometry_z),
        }


class FrozenExpertMixer(nn.Module):
    """Train Router/residual cheaply on cached frozen-expert outputs."""

    def __init__(
        self,
        *,
        n_descriptors: int,
        use_residual: bool,
        shared_targets: bool = False,
    ) -> None:
        super().__init__()
        self.use_residual = use_residual
        self.router = TargetSpecificRouter(
            n_descriptors=n_descriptors, shared_targets=shared_targets
        )
        self.residual = nn.Sequential(
            nn.Linear(128 * 3, 192),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(192, 64),
            nn.SiLU(),
            nn.Linear(64, 3),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(
        self,
        embeddings,
        expert_predictions,
        descriptors,
        *,
        temperature: float = 1.0,
        residual_enabled: bool = True,
    ):
        weights, projected = self.router(
            embeddings, expert_predictions, descriptors, temperature
        )
        mixture = torch.sum(weights * expert_predictions, dim=-1)
        residual = torch.zeros_like(mixture)
        if self.use_residual and residual_enabled:
            residual = 0.1 * torch.tanh(self.residual(torch.cat(projected, dim=-1)))
        return {
            "prediction": mixture + residual,
            "mixture": mixture,
            "residual": residual,
            "expert_predictions": expert_predictions,
            "router_weights": weights,
        }
