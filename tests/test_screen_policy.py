import pytest

from molgap.screen_policy import (
    PHYSICAL_BATCH_PER_DEVICE,
    REQUIRED_PAIRED_FIELDS,
    SCREEN_POLICY,
    validate_paired_screen_contract,
    validate_screen_arm,
)


def make_arm(name="baseline", **changes):
    arm = {
        "arm": name,
        "task_id": "qm9-gape-s42-v1",
        "platform_id": "kaggle2",
        "accelerator": "NvidiaTeslaT4",
        "data_role_fingerprint": "split-sha",
        "seed": 42,
        "precision": "fp32",
        "optimizer_fingerprint": "adamw-lr4e-4-wd1e-5",
        "schedule_fingerprint": "cosine40-eta1e-6",
        "sample_exposure": "30000x40",
        "physical_batch_per_device": 128,
        "device_count": 1,
        "gradient_accumulation_steps": 1,
    }
    arm.update(changes)
    return arm


def test_exact_batch128_is_the_only_screen_batch():
    assert validate_screen_arm(physical_batch_per_device=128) == {
        "policy": SCREEN_POLICY,
        "physical_batch_per_device": 128,
        "device_count": 1,
        "gradient_accumulation_steps": 1,
        "effective_batch_per_optimizer_step": 128,
    }
    assert PHYSICAL_BATCH_PER_DEVICE == 128


@pytest.mark.parametrize("batch", [48, 96, 127, 129, 256])
def test_non_128_physical_batch_is_rejected(batch):
    with pytest.raises(ValueError, match="exactly 128"):
        validate_screen_arm(physical_batch_per_device=batch)


@pytest.mark.parametrize("field", REQUIRED_PAIRED_FIELDS)
def test_paired_screen_rejects_every_shared_contract_mismatch(field):
    candidate = make_arm("candidate")
    candidate[field] = f"different-{field}"
    with pytest.raises(ValueError, match="contract mismatch"):
        validate_paired_screen_contract([make_arm(), candidate])


def test_paired_screen_accepts_only_matched_arms():
    result = validate_paired_screen_contract(
        [make_arm(), make_arm("random-control"), make_arm("candidate")]
    )
    assert result["arm_count"] == 3
    assert result["shared"]["task_id"] == "qm9-gape-s42-v1"
    assert result["resource"]["physical_batch_per_device"] == 128

