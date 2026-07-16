"""Target-specific soft routing for heterogeneous molecular experts."""

from __future__ import annotations

import torch
import torch.nn as nn


class TargetSpecificRouter(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 192,
        projection_dim: int = 128,
        n_descriptors: int = 0,
        n_targets: int = 3,
        n_experts: int = 3,
        dropout: float = 0.10,
        expert_dropout: float = 0.10,
        shared_targets: bool = False,
    ) -> None:
        super().__init__()
        self.n_targets = n_targets
        self.n_experts = n_experts
        self.n_descriptors = n_descriptors
        self.expert_dropout = expert_dropout
        self.shared_targets = shared_targets
        self.projections = nn.ModuleList([
            nn.Linear(embedding_dim, projection_dim) for _ in range(n_experts)
        ])
        router_in = projection_dim * n_experts + n_targets * n_experts + n_descriptors
        self.network = nn.Sequential(
            nn.Linear(router_in, 256),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, (1 if shared_targets else n_targets) * n_experts),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, embeddings, predictions, descriptors=None, temperature=1.0):
        projected = [
            layer(value.detach()) for layer, value in zip(self.projections, embeddings)
        ]
        parts = [*projected, predictions.detach().flatten(start_dim=1)]
        if self.n_descriptors:
            if descriptors is None:
                raise ValueError("descriptors are required by this Router")
            parts.append(descriptors.float())
        output_targets = 1 if self.shared_targets else self.n_targets
        logits = self.network(torch.cat(parts, dim=-1)).view(
            -1, output_targets, self.n_experts
        )
        if self.shared_targets:
            logits = logits.expand(-1, self.n_targets, -1).clone()
        if self.training and self.expert_dropout > 0:
            apply = torch.rand(logits.size(0), device=logits.device) < self.expert_dropout
            dropped = torch.randint(self.n_experts, (logits.size(0),), device=logits.device)
            rows = torch.arange(logits.size(0), device=logits.device)[apply]
            logits[rows, :, dropped[apply]] = torch.finfo(logits.dtype).min
        weights = torch.softmax(logits / max(float(temperature), 1e-6), dim=-1)
        return weights, projected
