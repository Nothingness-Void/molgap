"""Pure frozen-contract checks; no molecular role or model execution."""
import copy
import json

import pytest

from molgap.k1_screen_training import (
    EPOCHS, INITIAL_STATE_SHA256, ROW_ORDER_FINGERPRINT, ROWS_PER_EPOCH,
    SAMPLE_EXPOSURE, STEPS_PER_EPOCH, compute_row_order_fingerprint,
    epoch_order, validate_recipe,
    validate_runtime_preflight, _allocation_costs,
)
from molgap.screen_policy import canonical_fingerprint


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


@pytest.mark.parametrize("mode", ["reference", "ssma", "clean_fingerprint"])
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


def preflight_files(root, *, accepted=True):
    provenance = {"runtime_fingerprint": "1" * 64, "context": {"arm_id": "reference"}}
    architecture = {"accepted": accepted, "repeatability": {"accepted": True},
        "resume_roundtrip": {"accepted": True}, "zero_initialization_delta": 0.0,
        "maximum_overhead_fraction": 0.25, "synchronized_step_overhead_fraction": 0.20}
    certificate = {"status": "accepted", "calibration_checks_passed": True,
        "runtime_fingerprint": provenance["runtime_fingerprint"],
        "provenance_sha256": canonical_fingerprint(provenance),
        "architecture_sha256": canonical_fingerprint(architecture)}
    for name, value in [("runtime_provenance", provenance), ("runtime_certificate", certificate),
                         ("architecture_preflight", architecture),
                         ("runtime_manifest", {"runtime_fingerprint": provenance["runtime_fingerprint"]})]:
        (root / (name + ".json")).write_text(json.dumps(value), encoding="utf-8")
    return provenance


def test_training_requires_matching_arm_preflight(tmp_path):
    provenance = preflight_files(tmp_path)
    assert validate_runtime_preflight(tmp_path, provenance)["status"] == "accepted"
    changed = copy.deepcopy(provenance)
    changed["context"]["arm_id"] = "ssma"
    with pytest.raises(ValueError, match="identity"):
        validate_runtime_preflight(tmp_path, changed)


def test_preflight_requires_accepted_architecture(tmp_path):
    provenance = preflight_files(tmp_path, accepted=False)
    with pytest.raises(ValueError, match="qualified"):
        validate_runtime_preflight(tmp_path, provenance)


def test_preflight_does_not_trust_mutated_timing(tmp_path):
    provenance = preflight_files(tmp_path)
    path = tmp_path / "architecture_preflight.json"
    altered = json.loads(path.read_text())
    altered["synchronized_step_overhead_fraction"] = 0.40
    path.write_text(json.dumps(altered), encoding="utf-8")
    with pytest.raises(ValueError, match="qualified"):
        validate_runtime_preflight(tmp_path, provenance)


def test_allocation_cost_window_keeps_wall_and_allocated_device_separate():
    costs = _allocation_costs(12.5, "Tesla T4", scope="invocation_excludes_queue_bootstrap")
    assert costs[0]["value"] == costs[1]["value"] == 12.5
    assert costs[0]["semantics"] == "process_wall"
    assert costs[1]["semantics"] == "allocated_device"
    assert len(costs) == 2
