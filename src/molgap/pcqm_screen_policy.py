"""Compatibility wrapper for the global MolGap screening policy."""
from __future__ import annotations

from .screen_policy import PHYSICAL_BATCH_PER_DEVICE, SCREEN_POLICY, validate_screen_arm


SCREEN_BATCH_POLICY = SCREEN_POLICY
MIN_PHYSICAL_BATCH_PER_MODEL = PHYSICAL_BATCH_PER_DEVICE
MIN_EFFECTIVE_BATCH_PER_MODEL = PHYSICAL_BATCH_PER_DEVICE


def validate_screen_batch(
    *,
    physical_batch_per_device: int,
    device_count: int = 1,
    gradient_accumulation_steps: int = 1,
) -> dict:
    """Validate one independent arm under the global exact-batch contract."""
    return validate_screen_arm(
        physical_batch_per_device=physical_batch_per_device,
        device_count=device_count,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )
