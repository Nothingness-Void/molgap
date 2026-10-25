"""Shared human declarations feed existing same-run and attempt owners."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import struct

import pytest

from molgap.experiment_input import resolve_experiment_input, freeze_experiment_input
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
        "training_float32_target_sha256": hashlib.sha256(struct.pack("<3f", -1., 0., 1.)).hexdigest(),
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


def test_shared_same_job_pair_uses_existing_acceptance_and_exact_attempt(scientific):
    case = scientific
    value, authority, path = _pair(case)
    _commit(case, value, authority, path)
    _, spec, plan, _ = resolve_experiment_input(path, case.repo)
    assert [a["scientific_role"] for a in spec.to_dict()["arms"]] == ["reference", "candidate"]
    acceptance = json.loads((case.repo / plan["acceptance_plan"]).read_bytes())
    target_path = case.repo / value["same_run"]["target_manifest_ref"]
    retained = target_path.read_bytes()
    targets = json.loads(retained)
    targets.pop("training_float32_target_sha256")
    _write_json(target_path, targets)
    missing = deepcopy(acceptance)
    for entry in missing["arms"]:
        entry["target_manifest"]["sha256"] = hashlib.sha256(target_path.read_bytes()).hexdigest()
    assert check_acceptance_plan(spec, case.repo, missing)["status"] == "ACCEPTANCE_INPUTS_AVAILABLE"
    blocked = check_acceptance_plan(spec, case.repo, missing, require_training_targets=True)
    assert blocked["status"] == "BLOCKED"
    assert all(any("training_float32_target_sha256" in reason for reason in arm["blockers"])
               for arm in blocked["arms"])
    target_path.write_bytes(retained)
    assert check_acceptance_plan(spec, case.repo, acceptance, require_training_targets=True)["status"] == "ACCEPTANCE_INPUTS_AVAILABLE"
    for binding in spec.to_dict()["prospective"]["arms"]:
        proposal = json.loads((case.repo / binding["plan_spec_ref"]).read_bytes())
        commit = proposal["trajectory"]["state_at_start"]["source_commit"]
        assert validate_execution_identity(spec, binding["arm_id"], proposal["trajectory"], commit) == "compact-pair-v1"
    assert not any("comparison_prelaunch" in name for name in plan["generated_records"])
    from molgap.experiment_family_workflow import same_run_comparison_scope
    scope = same_run_comparison_scope(value, "candidate", "baseline")
    assert scope["experiment_purpose"] == "optimizer_comparison"
    assert scope["declared_intervention_fields"] == ["optimizer_identity"]
    for purpose in ("architecture_comparison", "unknown"):
        invalid = deepcopy(value)
        invalid["shared"]["purpose"] = purpose
        with pytest.raises(ValueError):
            same_run_comparison_scope(invalid, "candidate", "baseline", declared_fields=["optimizer_identity"])
    freeze_experiment_input(path, case.repo)
    previous = spec.identity
    value["attempt_id"] = "v2"
    value["retry_of"] = {"logical_run_id": "compact-pair", "attempt_id": "v1",
                         "spec_identity": previous, "decision_ref": "experiments/compact/retry.json"}
    _write_json(case.repo / value["retry_of"]["decision_ref"], {
        "retry_authorized": True, "retry_of": {k: value["retry_of"][k] for k in
            ("logical_run_id", "attempt_id", "spec_identity")},
        "remote_state": "NOT_SUBMITTED", "evidence_ref": value["authority_ref"]})
    authority.update(attempt_ids=["v2"], retry_of=value["retry_of"])
    _commit(case, value, authority, path)
    _, retry_spec, retry_plan, _ = resolve_experiment_input(path, case.repo)
    assert retry_spec.identity != previous
    assert retry_spec.to_dict()["arms"] == spec.to_dict()["arms"]
    assert all("/compact-pair-v2/prospective/" in b["output"]
               for b in retry_spec.to_dict()["prospective"]["arms"])
    assert {a["recipe"] for a in retry_plan["arms"]}.isdisjoint(a["recipe"] for a in plan["arms"])
    for binding in retry_spec.to_dict()["prospective"]["arms"]:
        proposal = json.loads((case.repo / binding["plan_spec_ref"]).read_bytes())
        commit = proposal["trajectory"]["state_at_start"]["source_commit"]
        assert validate_execution_identity(retry_spec, binding["arm_id"], proposal["trajectory"], commit) == "compact-pair-v2"


def test_reference_and_retry_cannot_infer_authority(scientific):
    case = scientific
    value, authority, path = _pair(case)
    value["logical_run_id"] = "unauthorized-pair"
    authority.pop("same_run_reference_arm_id")
    _commit(case, value, authority, path)
    with pytest.raises(ValueError, match="explicitly release"):
        resolve_experiment_input(path, case.repo)
    authority["same_run_reference_arm_id"] = "baseline"
    value["attempt_id"] = "v2"
    _commit(case, value, authority, path)
    with pytest.raises(ValueError):
        resolve_experiment_input(path, case.repo)
    value["attempt_id"] = "v1"
    value["shared"]["purpose"] = "architecture_comparison"
    _commit(case, value, authority, path)
    with pytest.raises(ValueError, match="outside one logical group"):
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


@pytest.mark.parametrize("report", [
    {"format": "molgap-workflow-preparation-v1", "status": "ERROR_REQUIRES_RECONCILIATION",
     "submitted": False, "prospective_published": "inspect_plan_result", "publication_intent": True},
    {"status": "PREPARED_FOR_PLATFORM", "submitted": False},
])
def test_not_submitted_retry_rejects_published_or_unknown_workflow_report(scientific, report, request):
    case = scientific
    value, authority, path = _pair(case)
    value["logical_run_id"] = f"not-submitted-{request.node.callspec.indices['report']}"
    _commit(case, value, authority, path)
    _, spec, _, _ = resolve_experiment_input(path, case.repo)
    frozen = freeze_experiment_input(path, case.repo)
    run = Path(frozen["run_directory"])
    report = {**report, "spec_identity": spec.identity}
    _write_json(run / "prepared/workflow_report.json", report)
    value["attempt_id"] = "v2"
    value["retry_of"] = {"logical_run_id": value["logical_run_id"], "attempt_id": "v1",
                         "spec_identity": spec.identity, "decision_ref": "experiments/compact/retry.json"}
    _write_json(case.repo / value["retry_of"]["decision_ref"], {
        "retry_authorized": True,
        "retry_of": {k: value["retry_of"][k] for k in ("logical_run_id", "attempt_id", "spec_identity")},
        "remote_state": "NOT_SUBMITTED", "evidence_ref": value["authority_ref"]})
    authority.update(attempt_ids=["v2"], retry_of=value["retry_of"])
    _commit(case, value, authority, path)
    with pytest.raises(ValueError, match="publication state"):
        resolve_experiment_input(path, case.repo)
