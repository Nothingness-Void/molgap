"""Archived three-expert archive-r03 implementation; not an active model path."""

from .hetero_moe import FrozenExpertMixer, HeterogeneousMoE
from .losses import HeteroMoELoss

__all__ = ["FrozenExpertMixer", "HeterogeneousMoE", "HeteroMoELoss"]
