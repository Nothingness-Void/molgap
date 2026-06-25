"""End-to-end hybrid models that combine 2D and 3D encoders."""
from __future__ import annotations

import torch.nn as nn

from .fusion import FusionHead, MoEFusionHead
from .gps import GPSWrapper
from .schnet import SchNetWrapper


class EndToEndHybrid(nn.Module):
    """Jointly train GPS 2D, SchNet 3D, and a fusion head.

    The frozen Phase 7 hybrid trains only the fusion head on precomputed
    embeddings. This wrapper keeps the same late-fusion architecture but lets
    gradients flow through both encoders.
    """

    def __init__(
        self,
        gps: GPSWrapper,
        schnet: SchNetWrapper,
        *,
        head: str = "moe",
        hidden: int = 192,
        dropout: float = 0.0,
        n_experts: int = 4,
    ):
        super().__init__()
        if head == "single":
            fusion = FusionHead("gate", hidden=hidden, dropout=dropout)
        elif head == "moe":
            fusion = MoEFusionHead(hidden=hidden, dropout=dropout, n_experts=n_experts)
        else:
            raise ValueError(f"Unknown hybrid head: {head}")
        self.gps = gps
        self.schnet = schnet
        self.fusion = fusion
        self.head = head

    def forward(self, batch_2d, batch_3d):
        h_2d = self.gps.encode(
            batch_2d.x,
            batch_2d.edge_index,
            batch_2d.edge_attr,
            batch_2d.batch,
        )
        charges = batch_3d.charges if hasattr(batch_3d, "charges") else None
        h_3d = self.schnet.encode(
            batch_3d.z,
            batch_3d.pos,
            batch_3d.batch,
            charges=charges,
        )
        return self.fusion(h_2d, h_3d)
