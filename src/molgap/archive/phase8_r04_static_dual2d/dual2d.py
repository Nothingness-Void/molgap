"""Dual-2D controls for Local GINE and Global GPS experts."""

from __future__ import annotations

import torch
import torch.nn as nn


class Dual2DConcatFusion(nn.Module):
    """Dense readout over concatenated frozen Local/GPS embeddings."""

    def __init__(self, embedding_dim: int = 192, dropout: float = 0.05) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(embedding_dim * 2, 192),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(192, 96),
            nn.SiLU(),
            nn.Linear(96, 3),
        )

    def forward(self, local_embedding, global_embedding):
        return self.network(torch.cat([local_embedding, global_embedding], dim=-1))


class Dual2DTargetGate(nn.Module):
    """Per-target convex weights over frozen Local and Global predictions."""

    def __init__(
        self,
        embedding_dim: int = 192,
        dropout: float = 0.10,
        prior_weights=None,
    ) -> None:
        super().__init__()
        self.local_projection = nn.Linear(embedding_dim, 128)
        self.global_projection = nn.Linear(embedding_dim, 128)
        self.router = nn.Sequential(
            nn.Linear(128 * 2 + 3 * 2, 192),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(192, 96),
            nn.SiLU(),
            nn.Linear(96, 3 * 2),
        )
        nn.init.zeros_(self.router[-1].weight)
        nn.init.zeros_(self.router[-1].bias)
        if prior_weights is None:
            prior_logits = torch.zeros(3, 2)
        else:
            prior = torch.as_tensor(prior_weights, dtype=torch.float32).clamp_min(1e-6)
            prior_logits = prior.log()
        self.register_buffer("prior_logits", prior_logits)

    def forward(self, local_embedding, global_embedding, expert_predictions, temperature=1.0):
        local = self.local_projection(local_embedding)
        global_value = self.global_projection(global_embedding)
        features = torch.cat([
            local,
            global_value,
            expert_predictions.flatten(start_dim=1),
        ], dim=-1)
        logits = self.router(features).view(-1, 3, 2) + self.prior_logits
        weights = torch.softmax(logits / max(float(temperature), 1e-6), dim=-1)
        prediction = torch.sum(weights * expert_predictions, dim=-1)
        return prediction, weights
