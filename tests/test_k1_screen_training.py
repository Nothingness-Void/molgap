"""Pure frozen-contract checks; no molecular role or model execution."""
import copy

import pytest

from molgap.k1_screen_training import (
    EPOCHS, INITIAL_STATE_SHA256, ROW_ORDER_FINGERPRINT, ROWS_PER_EPOCH,
    SAMPLE_EXPOSURE, STEPS_PER_EPOCH, compute_row_order_fingerprint,
    epoch_order, validate_recipe,
)


def recipe(mode="ssma"):
    return {"mode": mode, "row_order_fingerprint": ROW_ORDER_FINGERPRINT,
            "initialization_sha256": INITIAL_STATE_SHA256,
            "training_recipe": {"seed": 42, "batch_size": 128, "drop_last": True,
                "optimizer": "AdamW", "learning_rate": 4e-4, "weight_decay": 1e-5,
                "clip_grad_norm": 1.0, "scheduler": "CosineAnnealingLR",
                "scheduler_t_max": 40, "scheduler_eta_min": 1e-6,
                "ema": False, "selection": "best-development-live",
                "auxiliary_weight": 0.1 if mode == "clean_fingerprint" else 0.0},
            "acceptance_requirements": {"epochs": 40, "optimizer_steps": 31240,
                "sample_presentations": 3998720, "development_rows": 50000,
                "precision": "fp32"}}


def test_historical_sampler_and_exposure():
    assert compute_row_order_fingerprint() == ROW_ORDER_FINGERPRINT
    assert len(epoch_order(0)) == ROWS_PER_EPOCH == 99968
    assert len(set(epoch_order(0))) == ROWS_PER_EPOCH
    assert epoch_order(0) != epoch_order(1)
    assert EPOCHS * STEPS_PER_EPOCH == 31240
    assert SAMPLE_EXPOSURE == 3998720


@pytest.mark.parametrize("mode", ["ssma", "clean_fingerprint"])
def test_historical_recipe(mode):
    validate_recipe(recipe(mode), mode=mode)


@pytest.mark.parametrize("field,value", [
    ("epochs", 41), ("optimizer_steps", 31280),
    ("sample_presentations", 4000000), ("precision", "fp16")])
def test_reject_recipe_drift(field, value):
    altered = copy.deepcopy(recipe())
    altered["acceptance_requirements"][field] = value
    with pytest.raises(ValueError, match="exposure"):
        validate_recipe(altered, mode="ssma")


def test_reject_continuous_sampler_substitution():
    altered = recipe()
    altered["row_order_fingerprint"] = "0" * 64
    with pytest.raises(ValueError, match="sampler"):
        validate_recipe(altered, mode="ssma")
