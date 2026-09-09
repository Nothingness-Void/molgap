"""Executable comparability guard for newly frozen model screens."""
from __future__ import annotations

from collections.abc import Mapping, Sequence


SCREEN_POLICY = "molgap-screen-comparability-v3"
PHYSICAL_BATCH_PER_DEVICE = 128
REQUIRED_PAIRED_FIELDS = (
    "task_id",
    "platform_id",
    "accelerator",
    "data_role_fingerprint",
    "seed",
    "precision",
    "optimizer_fingerprint",
    "schedule_fingerprint",
    "sample_exposure",
)


def validate_screen_arm(
    *,
    physical_batch_per_device: int,
    device_count: int = 1,
    gradient_accumulation_steps: int = 1,
) -> dict:
    """Validate the immutable resource contract of one independent arm."""
    values = {
        "physical_batch_per_device": physical_batch_per_device,
        "device_count": device_count,
        "gradient_accumulation_steps": gradient_accumulation_steps,
    }
    if any(not isinstance(value, int) or value < 1 for value in values.values()):
        raise ValueError(f"Screen resource values must be positive integers: {values}")
    if physical_batch_per_device != PHYSICAL_BATCH_PER_DEVICE:
        raise ValueError(
            "New screens require physical batch exactly "
            f"{PHYSICAL_BATCH_PER_DEVICE} per model and device"
        )
    if device_count != 1:
        raise ValueError("Each independently optimized screen arm uses one visible device")
    if gradient_accumulation_steps != 1:
        raise ValueError("Screen arms do not use gradient accumulation")
    return {
        "policy": SCREEN_POLICY,
        **values,
        "effective_batch_per_optimizer_step": PHYSICAL_BATCH_PER_DEVICE,
    }


def validate_paired_screen_contract(arms: Sequence[Mapping]) -> dict:
    """Reject comparisons confounded by task, platform, or training exposure."""
    if len(arms) < 2:
        raise ValueError("A paired screen requires at least two arms")
    normalized = []
    for index, arm in enumerate(arms):
        missing = [field for field in REQUIRED_PAIRED_FIELDS if field not in arm]
        if missing:
            raise ValueError(f"Arm {index} is missing paired fields: {missing}")
        resource = validate_screen_arm(
            physical_batch_per_device=arm.get("physical_batch_per_device"),
            device_count=arm.get("device_count", 1),
            gradient_accumulation_steps=arm.get("gradient_accumulation_steps", 1),
        )
        normalized.append({**dict(arm), **resource})
    reference = normalized[0]
    mismatches = {
        field: [arm[field] for arm in normalized]
        for field in REQUIRED_PAIRED_FIELDS
        if any(arm[field] != reference[field] for arm in normalized[1:])
    }
    if mismatches:
        raise ValueError(f"Paired screen contract mismatch: {mismatches}")
    return {
        "policy": SCREEN_POLICY,
        "arm_count": len(normalized),
        "shared": {field: reference[field] for field in REQUIRED_PAIRED_FIELDS},
        "resource": {
            key: reference[key]
            for key in (
                "physical_batch_per_device",
                "device_count",
                "gradient_accumulation_steps",
                "effective_batch_per_optimizer_step",
            )
        },
    }

