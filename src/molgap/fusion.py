"""Compatibility exports for :mod:`molgap.models.fusion`."""

from .models.fusion import (
    FusionHead,
    BoundedResidualFusionHead,
    DualGPSFusionHead,
    MoEFusionHead,
    DescriptorAwareFusionHead,
)

__all__ = ['FusionHead', 'BoundedResidualFusionHead', 'DualGPSFusionHead', 'MoEFusionHead', 'DescriptorAwareFusionHead']
