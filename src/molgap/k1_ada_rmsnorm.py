"""Adaptive Root-Mean-Square Normalization (AdaRMSN) for Neural-Atom K1.

From "Plain Transformers Can be Powerful Graph Learners" (Ma et al., 2025/2026).
Replaces the 18 BatchNorm layers in K1's LocalGPSBlocks with per-token /
per-node adaptive RMSNorm. Removes all running batch buffers, eliminating
Clean-BN calibration requirements and EMA running-stat degradation.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class AdaRMSNorm(nn.Module):
    """Adaptive Root-Mean-Square Normalization layer.

    Formula:
        AdaRMSN(x) = (x / RMS(x)) * gamma'(x)
        RMS(x) = sqrt(mean(x^2) + eps)
        gamma'(x) = sqrt(mean((alpha * x + beta)^2) + eps)

    Parameters:
        alpha: learnable vector in R^D, initialized to 0.
        beta: learnable vector in R^D, initialized to 1.
    At initialization, gamma'(x) == 1, recovering standard RMSNorm.
    """

    def __init__(self, dim: int = 192, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.zeros(dim))
        self.beta = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x_norm = x * rms
        gamma = torch.sqrt(
            (self.alpha * x + self.beta).pow(2).mean(dim=-1, keepdim=True) + self.eps
        )
        return x_norm * gamma


def make_k1_ada_rmsnorm():
    """Build NeuralAtomEdgeStateGPS with AdaRMSNorm in place of BatchNorm."""
    from .qm9_neural_atom import make_encoder

    model = make_encoder("neural_atom_k1")
    for block in model.local_blocks:
        block.norm1 = AdaRMSNorm(192)
        block.norm3 = AdaRMSNorm(192)
        block.norm_with_batch = False
    return model
