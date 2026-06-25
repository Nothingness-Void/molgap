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


class MoEFusionHead(nn.Module):
    """Fusion head variant with learned soft routing over expert MLP heads.

    This keeps the Phase 7 frozen-encoder setup intact: GPS 2D and 3D encoder
    embeddings are projected, gate-fused into a shared representation, then a
    router assigns per-molecule soft weights over regression experts.
    """

    def __init__(
        self,
        hidden=192,
        dropout=0.0,
        dim_2d=192,
        dim_3d=192,
        n_targets=3,
        n_experts=4,
    ):
        super().__init__()
        if n_experts < 1:
            raise ValueError("n_experts must be >= 1")
        self.n_experts = n_experts
        self.n_targets = n_targets
        self.proj_2d = nn.Linear(dim_2d, hidden)
        self.proj_3d = nn.Linear(dim_3d, hidden)
        self.gate_fuse = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Sigmoid())
        self.router = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, n_experts),
        )
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, hidden // 2),
                nn.SiLU(),
                nn.Linear(hidden // 2, n_targets),
            )
            for _ in range(n_experts)
        ])

    def forward(self, h_2d, h_3d, return_gate=False):
        h_2d = self.proj_2d(h_2d)
        h_3d = self.proj_3d(h_3d)
        g = self.gate_fuse(torch.cat([h_2d, h_3d], dim=-1))
        h = g * h_2d + (1 - g) * h_3d
        w = torch.softmax(self.router(h), dim=-1)
        outs = torch.stack([expert(h) for expert in self.experts], dim=1)
        y = torch.sum(w.unsqueeze(-1) * outs, dim=1)
        if return_gate:
            return y, w
        return y


class DescriptorAwareFusionHead(nn.Module):
    """Fusion head whose 2D/3D gate also sees lightweight molecule descriptors.

    The descriptors are standardized scalar context features such as fragment
    count, rotatable bonds, element flags, and 2D topology proxies. They are not
    a replacement for the frozen GPS/SchNet encoders; they tell the fusion gate
    when a molecule resembles known failure modes such as salts or flexible
    structures.
    """

    def __init__(
        self,
        n_desc,
        hidden=192,
        desc_hidden=64,
        dropout=0.0,
        dim_2d=192,
        dim_3d=192,
        n_targets=3,
    ):
        super().__init__()
        if n_desc < 1:
            raise ValueError("n_desc must be >= 1")
        self.n_desc = n_desc
        self.proj_2d = nn.Linear(dim_2d, hidden)
        self.proj_3d = nn.Linear(dim_3d, hidden)
        self.proj_desc = nn.Sequential(
            nn.Linear(n_desc, desc_hidden),
            nn.SiLU(),
            nn.Linear(desc_hidden, hidden),
            nn.SiLU(),
        )
        self.gate = nn.Sequential(nn.Linear(hidden * 3, hidden), nn.Sigmoid())
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.SiLU(),
            nn.Linear(hidden // 2, n_targets),
        )

    def forward(self, h_2d, h_3d, desc, return_gate=False):
        h_2d = self.proj_2d(h_2d)
        h_3d = self.proj_3d(h_3d)
        h_desc = self.proj_desc(desc)
        g = self.gate(torch.cat([h_2d, h_3d, h_desc], dim=-1))
        h = g * h_2d + (1 - g) * h_3d
        y = self.head(torch.cat([h, h_desc], dim=-1))
        if return_gate:
            return y, g
        return y
