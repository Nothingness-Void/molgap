"""Shared batch contract for newly frozen PCQM 50K/100K screens."""
from __future__ import annotations


SCREEN_BATCH_POLICY = "pcqm-screen-batch-policy-v2"
MIN_PHYSICAL_BATCH_PER_MODEL = 128
MIN_EFFECTIVE_BATCH_PER_MODEL = 128


def validate_screen_batch(
    *,
    physical_batch_per_device: int,
    device_count: int = 1,
    gradient_accumulation_steps: int = 1,
) -> dict:
    """Validate one model process before a new 50K/100K screen is released."""
    values = {
        "physical_batch_per_device": physical_batch_per_device,
        "device_count": device_count,
        "gradient_accumulation_steps": gradient_accumulation_steps,
    }
    if any(not isinstance(value, int) or value < 1 for value in values.values()):
        raise ValueError(f"Batch contract values must be positive integers: {values}")
    if physical_batch_per_device < MIN_PHYSICAL_BATCH_PER_MODEL:
        raise ValueError(
            "New PCQM 50K/100K screens require physical batch >= "
            f"{MIN_PHYSICAL_BATCH_PER_MODEL} per model and device"
        )
    effective_batch = (
        physical_batch_per_device * device_count * gradient_accumulation_steps
    )
    if effective_batch < MIN_EFFECTIVE_BATCH_PER_MODEL:
        raise ValueError("Effective batch is below the screen minimum")
    return {
        "policy": SCREEN_BATCH_POLICY,
        **values,
        "effective_batch_per_optimizer_step": effective_batch,
    }
