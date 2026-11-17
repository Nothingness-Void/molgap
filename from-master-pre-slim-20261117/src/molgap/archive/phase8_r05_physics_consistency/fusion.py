"""Physics-consistent heads retained only for archive-r05 reproduction."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from molgap.fusion import FusionHead


def homo_lumo_gap_consistency_loss(
    prediction: torch.Tensor,
    *,
    delta: float = 0.1,
    reduction: str = "mean",
) -> torch.Tensor:
    """Penalize violation of ``Gap = LUMO - HOMO`` in a three-target output."""
    if prediction.ndim < 1 or prediction.shape[-1] != 3:
        raise ValueError("prediction must have final dimension 3: HOMO, LUMO, Gap")
    residual = prediction[..., 2] - (prediction[..., 1] - prediction[..., 0])
    return F.huber_loss(residual, torch.zeros_like(residual), delta=delta, reduction=reduction)


class StructuredPhysicsFusionHead(FusionHead):
    """Derive LUMO exactly from learned HOMO and a non-negative Gap."""

    def __init__(self, fusion_type="gate", hidden=128, dropout=0.0,
                 dim_2d=192, dim_3d=192):
        super().__init__(
            fusion_type=fusion_type,
            hidden=hidden,
            dropout=dropout,
            dim_2d=dim_2d,
            dim_3d=dim_3d,
            n_targets=2,
        )

    def forward(self, h_2d, h_3d):
        raw = super().forward(h_2d, h_3d)
        homo = raw[..., 0]
        gap = F.softplus(raw[..., 1])
        lumo = homo + gap
        return torch.stack((homo, lumo, gap), dim=-1)
