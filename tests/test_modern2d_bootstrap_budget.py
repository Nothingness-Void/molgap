"""Bounded startup metadata tests: no environment install/model/network."""
from copy import deepcopy
import hashlib
import json

import pytest

from test_modern2d_formal_release import _load_module


def setup(tmp_path):
    from pathlib import Path
    entry = _load_module(Path(__file__).parents[1] / "platforms/kaggle/run_experiment.py", "modern2d_budget_fixture")
    declaration = dict(arms=[dict(family=dict(name="ppgt_small", version="3"))],
        prospective=dict(arms=[dict(plan_spec_ref="experiments/profile/ppgt/plan_input.json")]))
    authority = dict(user_authorized=True, purpose="CONFIGURED_EXECUTION_PROFILE",
        scientific_training_authorized=False, scratch_optimizer_updates_per_model=6,
        allocated_t4_hours_ceiling=1,
        gpu=dict(wall_ceiling_seconds=1800, allocated_devices=2, accelerator="NvidiaTeslaT4"))
    folder = tmp_path / "experiments/profile"
    folder.mkdir(parents=True)
    return entry, declaration, authority, folder


def publish(folder, authority):
    raw = json.dumps(authority).encode()
    (folder / "authority.json").write_bytes(raw)
    (folder / "acceptance_plan.json").write_text(json.dumps(dict(authority=dict(
        path="experiments/profile/authority.json", sha256=hashlib.sha256(raw).hexdigest()))))


def test_profile_budget_is_1800_not_formal_16200(tmp_path):
    entry, declaration, authority, folder = setup(tmp_path)
    publish(folder, authority)
    assert entry._modern2d_wall_ceiling(declaration, tmp_path) == 1800
    (folder / "authority.json").write_bytes(b"{}")
    with pytest.raises(ValueError, match="authority hash"):
        entry._modern2d_wall_ceiling(declaration, tmp_path)


@pytest.mark.parametrize("delta", [dict(user_authorized=False),
    dict(scientific_training_authorized=True), dict(scratch_optimizer_updates_per_model=7),
    dict(allocated_t4_hours_ceiling=2), dict(purpose="UNKNOWN")])
def test_profile_scope_expansion_fails_before_environment(tmp_path, delta):
    entry, declaration, authority, folder = setup(tmp_path)
    authority.update(deepcopy(delta))
    publish(folder, authority)
    with pytest.raises(ValueError):
        entry._modern2d_wall_ceiling(declaration, tmp_path)


def test_repair_bootstrap_accepts_exact_two_family_probe(tmp_path):
    entry, declaration, authority, folder = setup(tmp_path)
    declaration["arms"].append(dict(family=dict(name="esa_small", version="4")))
    authority.update(purpose="CONFIGURED_EXECUTION_REPAIR",
        scratch_optimizer_updates_per_model=8, execution_reference_required=True)
    publish(folder, authority)
    assert entry._modern2d_wall_ceiling(declaration, tmp_path) == 1800


@pytest.mark.parametrize("delta", [dict(scientific_training_authorized=True),
    dict(scratch_optimizer_updates_per_model=6), dict(execution_reference_required=False),
    dict(allocated_t4_hours_ceiling=2)])
def test_repair_bootstrap_rejects_expanded_or_missing_reference_probe(tmp_path, delta):
    entry, declaration, authority, folder = setup(tmp_path)
    declaration["arms"].append(dict(family=dict(name="esa_small", version="4")))
    authority.update(purpose="CONFIGURED_EXECUTION_REPAIR",
        scratch_optimizer_updates_per_model=8, execution_reference_required=True)
    authority.update(delta)
    publish(folder, authority)
    with pytest.raises(ValueError):
        entry._modern2d_wall_ceiling(declaration, tmp_path)


def test_repair_bootstrap_rejects_single_or_reordered_families(tmp_path):
    entry, declaration, authority, folder = setup(tmp_path)
    authority.update(purpose="CONFIGURED_EXECUTION_REPAIR",
        scratch_optimizer_updates_per_model=8, execution_reference_required=True)
    publish(folder, authority)
    with pytest.raises(ValueError):
        entry._modern2d_wall_ceiling(declaration, tmp_path)
    declaration["arms"].insert(0, dict(family=dict(name="esa_small", version="4")))
    with pytest.raises(ValueError):
        entry._modern2d_wall_ceiling(declaration, tmp_path)
