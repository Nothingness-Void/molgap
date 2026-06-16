"""Embedding-level fusion head for the hybrid model (GPS 2D + 3D encoder).

The hybrid is *late* fusion: the two encoders are frozen, their pooled
embeddings are pre-computed, and only this head is trained. Gate fusion
forms a per-molecule, per-dimension convex combination of the two projected
embeddings (g·h2d + (1-g)·h3d), which beat plain concatenation in the Optuna
search — see `docs/phase7.md`.

Supports asymmetric input dims (e.g. GPS 2D = 192, TensorNet 3D = 128).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class FusionHead(nn.Module):
    """Combine pre-computed 2D and 3D molecular embeddings into HOMO/LUMO/Gap.

    Args:
        fusion_type: ``"gate"`` (convex combination, default) or ``"concat"``.
        hidden: projection / head width.
        dropout: dropout in the regression head.
        dim_2d, dim_3d: input embedding dims (both 192 in Phase 7).
    """

    def __init__(self, fusion_type="gate", hidden=128, dropout=0.0,
                 dim_2d=192, dim_3d=192, n_targets=3):
        super().__init__()
        self.fusion_type = fusion_type
        self.proj_2d = nn.Linear(dim_2d, hidden)
        self.proj_3d = nn.Linear(dim_3d, hidden)
        if fusion_type == "gate":
            self.gate = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Sigmoid())
            head_in = hidden
        else:
            head_in = hidden * 2
        self.head = nn.Sequential(
            nn.Linear(head_in, hidden), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.SiLU(),
            nn.Linear(hidden // 2, n_targets),
        )

    def forward(self, h_2d, h_3d):
        h_2d = self.proj_2d(h_2d)
        h_3d = self.proj_3d(h_3d)
        if self.fusion_type == "gate":
            g = self.gate(torch.cat([h_2d, h_3d], dim=-1))
            h = g * h_2d + (1 - g) * h_3d
        else:
            h = torch.cat([h_2d, h_3d], dim=-1)
        return self.head(h)
