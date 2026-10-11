"""Local prefix boundaries; synthetic identities only, no accelerator/data roles."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from molgap import k1_screen_training as owner


def recipe(mode="reference"):
    value = owner.build_screen_recipe(mode, source_idx_sha256="a" * 64, target_sha256="b" * 64)
    value["local_execution_prefix"] = {"epochs": 10, "accelerator": "NVIDIA GeForce RTX 5060",
        "original_execution": mode == "reference", "purpose": "execution-quality-prefix"}
    value["acceptance_requirements"].update(epochs=10, optimizer_steps=7810, sample_presentations=999680)
    return value


@pytest.mark.parametrize("mode", ["reference", "fused_layout"])
def test_valid_local_recipe_preserves_full_schedule(mode):
    value = recipe(mode)
    owner.validate_recipe(value, mode=mode)
    assert value["training_recipe"]["scheduler_t_max"] == 40
    spec = SimpleNamespace(to_dict=lambda: {"platform": {"name": "local", "device_count": 1,
        "accelerator": "NVIDIA GeForce RTX 5060"}})
    assert owner._execution_scope(spec, value) == (10, "NVIDIA GeForce RTX 5060",
        {"original_execution": mode == "reference"})


@pytest.mark.parametrize("change", [{"epochs": 10.0}, {"epochs": 20}, {"original_execution": 1},
                                  {"original_execution": False}, {"accelerator": "Tesla T4"}, {"extra": True}])
def test_mutated_prefix_rejected(change):
    value = recipe()
    value["local_execution_prefix"].update(change)
    with pytest.raises(ValueError, match="local execution prefix"):
        owner.validate_recipe(value, mode="reference")


@pytest.mark.parametrize("platform", [{"name": "kaggle", "device_count": 1, "accelerator": "NVIDIA GeForce RTX 5060"},
                                     {"name": "local", "device_count": 2, "accelerator": "NVIDIA GeForce RTX 5060"},
                                     {"name": "local", "device_count": 1, "accelerator": "Tesla T4"}])
def test_local_prefix_cannot_override_remote_or_other_device_contract(platform):
    with pytest.raises(ValueError, match="single-device local Spec"):
        owner._execution_scope(SimpleNamespace(to_dict=lambda: {"platform": platform}), recipe())


def test_default_recipe_still_requires_full_t4_run():
    value = owner.build_screen_recipe("reference", source_idx_sha256="a" * 64, target_sha256="b" * 64)
    owner.validate_recipe(value, mode="reference")
    assert owner._execution_scope(SimpleNamespace(to_dict=lambda: {}), value) == (40, "T4", {})
    value = deepcopy(value)
    value["acceptance_requirements"]["epochs"] = 10
    with pytest.raises(ValueError, match="frozen acceptance exposure"):
        owner.validate_recipe(value, mode="reference")
