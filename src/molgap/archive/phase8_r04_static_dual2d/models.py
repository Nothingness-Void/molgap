"""Model construction for the archive-r04 Local-GINE/GPS blend."""

from __future__ import annotations

from molgap.gps import GPSWrapper

from .local_gine import LocalGINEExpert


def make_expert(kind: str):
    """Create one of the two frozen archive-r04 expert architectures."""
    if kind == "local":
        return LocalGINEExpert()
    if kind == "global":
        return GPSWrapper(
            hidden_channels=192,
            num_layers=9,
            num_heads=4,
            dropout=0.05,
            pooling="mean_max",
        )
    raise ValueError(f"Unsupported dual-2D expert: {kind}")
