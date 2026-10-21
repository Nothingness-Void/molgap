"""Shared human declarations feed existing same-run and attempt owners."""
from copy import deepcopy
import json

import pytest

from molgap.experiment_input import resolve_experiment_input
from molgap.experiment_family_workflow import check_acceptance_plan
from molgap.common.identity import validate_execution_identity
from test_scientific_input import scientific
from test_experiment_workflow import _write_json, _git


def _pair(case):
    value = deepcopy(case.value)
    value.update(logical_run_id="compact-pair", authority_ref="experiments/compact/authority.json")
    shared = deepcopy(value["arms"][0])
    shared.pop("arm_id")
    shared.pop("reference_ref")
    shared["training"] = {"learning_rate": .0006}
    value["shared"] = shared
    value["arms"] = [{"arm_id": "baseline", "training": {}}, {"arm_id": "candidate"}]
    target = "experiments/compact/target.json"
    recipe = json.loads((case.repo / shared["recipe_ref"]).read_bytes())
    _write_json(case.repo / target, {
        "development_target_sha256": recipe["acceptance_requirements"]["target_sha256"],
        "target_encoding": "little-endian-float32-contiguous-raw-bytes"})
    value["same_run"] = {"reference_arm_id": "baseline", "target_manifest_ref": target}
    authority = {"user_authorized": True, "allocated_t4_hours_ceiling": 1,
                 "arms": ["baseline", "candidate"], "seeds": [42],
                 "same_run_reference_arm_id": "baseline"}
    path = case.repo / "experiments/compact/input.json"
    return value, authority, path


def _commit(case, value, authority, path):
    _write_json(path, value)
    _write_json(case.repo / value["authority_ref"], authority)
    _git(case.repo, "add", ".")
    _git(case.repo, "-c", "commit.gpgsign=false", "commit", "--allow-empty", "-m", "Synthetic pair input")


def test_shared_same_job_pair_uses_existing_acceptance(scientific):
    case = scientific
    value, authority, path = _pair(case)
    _commit(case, value, authority, path)
    _, spec, plan, _ = resolve_experiment_input(path, case.repo)
    assert [a["scientific_role"] for a in spec.to_dict()["arms"]] == ["reference", "candidate"]
    acceptance = json.loads((case.repo / plan["acceptance_plan"]).read_bytes())
    assert check_acceptance_plan(spec, case.repo, acceptance)["status"] == "ACCEPTANCE_INPUTS_AVAILABLE"
    for binding in spec.to_dict()["prospective"]["arms"]:
        proposal = json.loads((case.repo / binding["plan_spec_ref"]).read_bytes())
        commit = proposal["trajectory"]["state_at_start"]["source_commit"]
        assert validate_execution_identity(spec, binding["arm_id"], proposal["trajectory"], commit) == "compact-pair-v1"
    assert not any("comparison_prelaunch" in name for name in plan["generated_records"])


def test_reference_cannot_infer_authority(scientific):
    case = scientific
    value, authority, path = _pair(case)
    value["logical_run_id"] = "unauthorized-pair"
    authority.pop("same_run_reference_arm_id")
    _commit(case, value, authority, path)
    with pytest.raises(ValueError, match="explicitly release"):
        resolve_experiment_input(path, case.repo)


@pytest.mark.parametrize("arm", [None, [], "candidate", {"training": {}}])
def test_arm_override_shape_is_checked_before_arm_id_access(scientific, arm, request):
    case = scientific
    value, authority, path = _pair(case)
    value["logical_run_id"] = f"arm-shape-{request.node.callspec.indices['arm']}"
    value["arms"][0] = arm
    _commit(case, value, authority, path)
    with pytest.raises(ValueError, match="arm entries"):
        resolve_experiment_input(path, case.repo)
