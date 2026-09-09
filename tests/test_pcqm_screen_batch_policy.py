import pytest

from molgap.pcqm_screen_policy import (
    MIN_EFFECTIVE_BATCH_PER_MODEL,
    MIN_PHYSICAL_BATCH_PER_MODEL,
    SCREEN_BATCH_POLICY,
    validate_screen_batch,
)


def test_new_screen_batch128_is_minimum():
    contract = validate_screen_batch(physical_batch_per_device=128)
    assert contract == {
        "policy": SCREEN_BATCH_POLICY,
        "physical_batch_per_device": 128,
        "device_count": 1,
        "gradient_accumulation_steps": 1,
        "effective_batch_per_optimizer_step": 128,
    }
    assert MIN_PHYSICAL_BATCH_PER_MODEL == 128
    assert MIN_EFFECTIVE_BATCH_PER_MODEL == 128


@pytest.mark.parametrize("batch_size", [1, 47, 48, 96, 127])
def test_new_screen_rejects_low_physical_batch(batch_size):
    with pytest.raises(ValueError, match="physical batch"):
        validate_screen_batch(physical_batch_per_device=batch_size)


def test_two_independent_t4_arms_do_not_form_one_effective_batch():
    left = validate_screen_batch(physical_batch_per_device=128, device_count=1)
    right = validate_screen_batch(physical_batch_per_device=128, device_count=1)
    assert left["effective_batch_per_optimizer_step"] == 128
    assert right["effective_batch_per_optimizer_step"] == 128


def test_distributed_batch_is_reported_per_model_optimizer_step():
    contract = validate_screen_batch(
        physical_batch_per_device=128,
        device_count=2,
        gradient_accumulation_steps=2,
    )
    assert contract["effective_batch_per_optimizer_step"] == 512
