"""Fail closed on actual frozen diagnostic metadata; no model/data execution."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from molgap.constants import REPO_ROOT
from molgap.experiment_spec import ExperimentSpec
from molgap.k1_execution_backend import K1ExecutionConfig
from molgap.k1_infrastructure_comparison import check_plan, comparison_base, BASE
from molgap.training_reproducibility import atomic_json, sha256_file


@pytest.fixture
def declaration(tmp_path, monkeypatch):
    source = REPO_ROOT / BASE
    spec = ExperimentSpec.from_json((source / "experiment_spec.json").read_text())
    plan = json.loads((source / "acceptance_plan.json").read_text())
    pointers = [plan["authority"], *[e["contract"] for e in plan["arms"]]]
    for pointer in pointers:
        dest = tmp_path / pointer["path"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((REPO_ROOT / pointer["path"]).read_bytes())
    for entry in plan["arms"]:
        initial = tmp_path / entry["initial_state"]["path"]
        initial.parent.mkdir(parents=True, exist_ok=True)
        initial.write_bytes(b"synthetic frozen-state metadata fixture")
        entry["initial_state"]["sha256"] = sha256_file(initial)
    import molgap.v4_runtime as runtime
    monkeypatch.setattr(runtime, "inspect_frozen_state_artifact", lambda *args, **kwargs: {})
    return tmp_path, spec, plan


def _execution_provider_round(declaration):
    """Build a metadata-only v2 declaration without running a model."""
    root, spec, plan = declaration
    base = BASE + "/execution_providers"
    payload = deepcopy(spec.to_dict())
    payload["logical_run_id"] = payload["logical_run_id"].replace("-v1", "-v2")
    for item in payload["prospective"]["arms"]:
        arm_id = item["arm_id"]
        item["plan_spec_ref"] = f"{base}/{arm_id}/plan_input.json"
        item["output"] = f"{base}/{arm_id}/rml"
        item["trajectory_id"] += "-v2"
    spec = ExperimentSpec(payload)
    plan = deepcopy(plan)
    plan["spec_identity"] = spec.identity
    plan["authority"]["path"] = base + "/authority.json"
    config = K1ExecutionConfig()
    authority = json.loads((root / BASE / "authority.json").read_text())
    authority["optimized_execution"] = config.to_dict()
    atomic_json(root / plan["authority"]["path"], authority)
    plan["authority"]["sha256"] = sha256_file(root / plan["authority"]["path"])
    for entry in plan["arms"]:
        arm_id = entry["arm_id"]
        old_path = root / entry["contract"]["path"]
        record = json.loads(old_path.read_text())
        if arm_id == "optimized":
            record["k1_execution"] = config.to_dict()
            record["execution_policy"]["k1_execution"] = config.to_dict()
        new_path = f"{base}/{arm_id}/contract.json"
        atomic_json(root / new_path, record)
        entry["contract"]["path"] = new_path
        entry["contract"]["sha256"] = sha256_file(root / new_path)
    return root, spec, plan


def test_complete_bounded_execution_declaration(declaration):
    root, spec, plan = declaration
    result = check_plan(spec, root, plan)
    assert result["status"] == "ACCEPTANCE_INPUTS_AVAILABLE"
    assert result["scientific_acceptance"] == "NOT_EVALUATED"
    assert result["replay_readiness"] == "NOT_APPLICABLE_EXECUTION_DIAGNOSTIC"


def test_round_base_is_derived_from_spec_and_binds_execution_config(declaration):
    root, spec, plan = _execution_provider_round(declaration)
    assert comparison_base(spec) == BASE + "/execution_providers"
    result = check_plan(spec, root, plan)
    assert result["status"] == "ACCEPTANCE_INPUTS_AVAILABLE"


def test_round_rejects_authority_config_drift(declaration):
    root, spec, plan = _execution_provider_round(declaration)
    authority_path = root / plan["authority"]["path"]
    authority = json.loads(authority_path.read_text())
    authority["optimized_execution"]["pooling"] = "segmented"
    atomic_json(authority_path, authority)
    plan["authority"]["sha256"] = sha256_file(authority_path)
    with pytest.raises(ValueError, match="configuration"):
        check_plan(spec, root, plan)


def test_legacy_recipe_cannot_gain_unbound_execution_authority(declaration):
    root, spec, plan = declaration
    authority_path = root / plan["authority"]["path"]
    authority = json.loads(authority_path.read_text())
    authority["optimized_execution"] = K1ExecutionConfig().to_dict()
    atomic_json(authority_path, authority)
    plan["authority"]["sha256"] = sha256_file(authority_path)
    with pytest.raises(ValueError, match="configuration"):
        check_plan(spec, root, plan)


def test_round_rejects_cross_round_authority_alias(declaration):
    root, spec, plan = _execution_provider_round(declaration)
    old_authority = root / BASE / "authority.json"
    plan["authority"] = {"path": BASE + "/authority.json", "sha256": sha256_file(old_authority)}
    with pytest.raises(ValueError, match="authority path"):
        check_plan(spec, root, plan)


def test_round_rejects_plan_suffix_alias(declaration):
    root, spec, _ = _execution_provider_round(declaration)
    payload = deepcopy(spec.to_dict())
    payload["prospective"]["arms"][0]["plan_spec_ref"] = (
        BASE + "/execution_providers/original/other.json")
    malformed = ExperimentSpec(payload)
    with pytest.raises(ValueError, match="suffix"):
        comparison_base(malformed)


@pytest.mark.parametrize("field,value", [("user_authorized", False), ("seeds", [42, 43]),
    ("official_validation_allowed", True), ("wall_ceiling_seconds", 28800)])
def test_rehashed_authority_cannot_expand_scope(declaration, field, value):
    root, spec, plan = declaration
    pointer = plan["authority"]
    path = root / pointer["path"]
    authority = json.loads(path.read_text())
    authority[field] = value
    atomic_json(path, authority)
    pointer["sha256"] = sha256_file(path)
    with pytest.raises(ValueError, match="scope"):
        check_plan(spec, root, plan)


def test_rehashed_profile_budget_cannot_expand(declaration):
    root, spec, plan = declaration
    for entry in plan["arms"]:
        path = root / entry["contract"]["path"]
        record = json.loads(path.read_text())
        record["profiling"]["active_steps"] = 100000
        atomic_json(path, record)
        entry["contract"]["sha256"] = sha256_file(path)
    with pytest.raises(ValueError, match="window"):
        check_plan(spec, root, plan)
