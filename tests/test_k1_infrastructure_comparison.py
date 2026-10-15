"""Fail closed on actual frozen diagnostic metadata; no model/data execution."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from molgap.constants import REPO_ROOT
from molgap.experiment_spec import ExperimentSpec
from molgap.k1_infrastructure_comparison import check_plan, BASE
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


def test_complete_bounded_execution_declaration(declaration):
    root, spec, plan = declaration
    result = check_plan(spec, root, plan)
    assert result["status"] == "ACCEPTANCE_INPUTS_AVAILABLE"
    assert result["scientific_acceptance"] == "NOT_EVALUATED"
    assert result["replay_readiness"] == "NOT_APPLICABLE_EXECUTION_DIAGNOSTIC"


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
