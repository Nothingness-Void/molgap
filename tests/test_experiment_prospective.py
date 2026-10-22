"""Synthetic per-arm planning boundaries; no repository RML records are written."""
from __future__ import annotations

import copy
import hashlib
import json
from unittest.mock import Mock

import pytest

from molgap import experiment_cli as cli
from molgap import experiment_prospective as prospective
from molgap.experiment_spec import ExperimentSpec, SCHEMA_VERSION_V2
from molgap.experiment_terminal import TerminalDescriptor, translate_terminal_descriptor
from molgap.research_memory.plan import PlanBatchError
from molgap.screen_policy import canonical_fingerprint
from test_experiment_spec import payload
from test_experiment_terminal import case


@pytest.fixture
def v2_payload(payload):
    value = copy.deepcopy(payload)
    value["schema_version"] = SCHEMA_VERSION_V2
    value["prospective"] = {"arms": [
        {
            "arm_id": arm["arm_id"], "trajectory_id": f"T-{arm['arm_id']}",
            "plan_spec_ref": f"inputs/{arm['arm_id']}.json",
            "plan_spec_sha256": "0" * 64,
            "output": f"experiments/planned-{arm['arm_id']}",
        }
        for arm in value["arms"]
    ]}
    return value


def test_v2_roundtrip_and_v1_compatibility(v2_payload, payload):
    v2 = ExperimentSpec(v2_payload)
    assert ExperimentSpec.from_json(v2.to_json()) == v2
    assert ExperimentSpec(payload).to_dict()["prospective"] == payload["prospective"]


def test_same_run_replay_requires_explicit_role_bound_declaration(v2_payload):
    reference_id = v2_payload["arms"][0]["arm_id"]
    candidate_id = v2_payload["arms"][1]["arm_id"]
    v2_payload["arms"][1]["scientific_role"] = "candidate"
    v2_payload["prospective"]["same_run_replay"] = {
        "reference_arm_id": reference_id, "candidate_arm_ids": [candidate_id],
    }
    assert ExperimentSpec(v2_payload).to_dict()["prospective"]["same_run_replay"]["reference_arm_id"] == reference_id
    bad = copy.deepcopy(v2_payload)
    bad["prospective"]["same_run_replay"]["candidate_arm_ids"] = []
    with pytest.raises(ValueError, match="expected nonempty array"):
        ExperimentSpec(bad)
    bad = copy.deepcopy(v2_payload)
    bad["prospective"]["same_run_replay"]["candidate_arm_ids"] = [reference_id]
    with pytest.raises(ValueError, match="reference and candidate roles"):
        ExperimentSpec(bad)
    bad = copy.deepcopy(v2_payload)
    bad["prospective"]["same_run_replay"]["candidate_arm_ids"] = [candidate_id, candidate_id]
    with pytest.raises(ValueError, match="Duplicate same-run candidate arm"):
        ExperimentSpec(bad)


@pytest.mark.parametrize("change", [
    "missing", "extra", "duplicate_arm", "duplicate_trajectory", "duplicate_output",
    "unknown_field", "missing_field", "bad_digest", "absolute_ref", "drive_ref",
    "traversal_ref", "linked_spelling", "outside_output", "bare_experiments",
])
def test_v2_strict_mapping_and_paths(v2_payload, change):
    arms = v2_payload["prospective"]["arms"]
    if change == "missing":
        arms.pop()
    elif change == "extra":
        arms.append(dict(arms[0], arm_id="extra", trajectory_id="T-extra", output="experiments/extra"))
    elif change == "duplicate_arm":
        arms[1]["arm_id"] = arms[0]["arm_id"]
    elif change == "duplicate_trajectory":
        arms[1]["trajectory_id"] = arms[0]["trajectory_id"]
    elif change == "duplicate_output":
        arms[1]["output"] = arms[0]["output"].upper().replace("EXPERIMENTS", "experiments")
    elif change == "unknown_field":
        arms[0]["hypothesis"] = "not a plan binding"
    elif change == "missing_field":
        arms[0].pop("plan_spec_sha256")
    elif change == "bad_digest":
        arms[0]["plan_spec_sha256"] = "A" * 64
    elif change == "absolute_ref":
        arms[0]["plan_spec_ref"] = "/tmp/plan.json"
    elif change == "drive_ref":
        arms[0]["plan_spec_ref"] = "C:/plan.json"
    elif change == "traversal_ref":
        arms[0]["plan_spec_ref"] = "inputs/../plan.json"
    elif change == "linked_spelling":
        arms[0]["plan_spec_ref"] = "inputs\\plan.json"
    elif change == "outside_output":
        arms[0]["output"] = "research_memory/derived/arm"
    else:
        arms[0]["output"] = "experiments"
    with pytest.raises(ValueError):
        ExperimentSpec(v2_payload)


@pytest.fixture
def planning_case(tmp_path, v2_payload):
    root = tmp_path / "repo"
    root.mkdir()
    for arm, binding in zip(v2_payload["arms"], v2_payload["prospective"]["arms"]):
        plan_input = {
            "trajectory": {
                "trajectory_id": binding["trajectory_id"],
                "state_at_start": {"source_config_identity": canonical_fingerprint(arm)},
            },
            "decision_state": {},
        }
        path = root / binding["plan_spec_ref"]
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = json.dumps(plan_input, sort_keys=True, separators=(",", ":")).encode()
        path.write_bytes(raw)
        binding["plan_spec_sha256"] = hashlib.sha256(raw).hexdigest()
    spec = ExperimentSpec(v2_payload)
    spec_path = root / "spec.json"
    spec_path.write_bytes(spec.to_json().encode())
    return root, spec_path, spec


def _invoke(capsys, root, spec_path, expected_code):
    code = cli.main(["plan-prospective", "--spec", str(spec_path), "--repo-root", str(root)])
    assert code == expected_code
    return json.loads(capsys.readouterr().out)


def test_execution_commit_is_bound_only_at_publication(planning_case, monkeypatch):
    import subprocess
    root, _, spec = planning_case
    originals = {b["plan_spec_ref"]: (root / b["plan_spec_ref"]).read_bytes()
                 for b in spec.to_dict()["prospective"]["arms"]}
    commit = "a" * 40
    monkeypatch.setattr(subprocess, "check_output", lambda *args, **kw: commit + "\n")
    captured = []
    def publish(root, plans):
        captured.extend(plans)
        return {"status": "PLANNED", "batch": {"size": len(plans)}, "results": [
            {"trajectory_id": p["spec"]["trajectory"]["trajectory_id"], "path": p["output"], "status": "PLANNED"}
            for p in plans]}
    monkeypatch.setattr(prospective, "plan_many", publish)
    assert prospective.plan_prospective(spec, root, execution_source_commit=commit)[1] == 0
    assert all(p["spec"]["trajectory"]["state_at_start"]["source_commit"] == commit
               and p["spec"]["decision_state"]["source_commit"] == commit for p in captured)
    assert all((root / path).read_bytes() == raw for path, raw in originals.items())
    with pytest.raises(ValueError, match="Packaged execution commit"):
        prospective.plan_prospective(spec, root, execution_source_commit="b" * 40)


@pytest.mark.parametrize("change", [
    "second_hash", "second_trajectory", "second_arm_identity", "existing_output",
])
def test_cli_prevalidates_all_arms_before_planning(
        capsys, monkeypatch, planning_case, change):
    root, spec_path, spec = planning_case
    bindings = spec.to_dict()["prospective"]["arms"]
    second = bindings[1]
    input_path = root / second["plan_spec_ref"]
    if change == "second_hash":
        input_path.write_bytes(input_path.read_bytes() + b"\n")
    elif change in {"second_trajectory", "second_arm_identity"}:
        value = json.loads(input_path.read_text())
        if change == "second_trajectory":
            value["trajectory"]["trajectory_id"] = "T-wrong"
        else:
            value["trajectory"]["state_at_start"]["source_config_identity"] = "wrong"
        raw = json.dumps(value).encode()
        input_path.write_bytes(raw)
        declaration = spec.to_dict()
        declaration["prospective"]["arms"][1]["plan_spec_sha256"] = hashlib.sha256(raw).hexdigest()
        spec_path.write_bytes(ExperimentSpec(declaration).to_json().encode())
    else:
        (root / second["output"]).mkdir(parents=True)
    planner = Mock()
    monkeypatch.setattr(prospective, "plan_many", planner)
    assert _invoke(capsys, root, spec_path, 2)["status"] == "ERROR"
    planner.assert_not_called()


def test_cli_rejects_ref_symlink_even_inside_repo(capsys, monkeypatch, planning_case):
    root, spec_path, spec = planning_case
    binding = spec.to_dict()["prospective"]["arms"][0]
    target = root / binding["plan_spec_ref"]
    alias = target.with_name("linked.json")
    try:
        alias.symlink_to(target)
    except OSError:
        pytest.skip("symlinks unavailable on this host")
    declaration = spec.to_dict()
    declaration["prospective"]["arms"][0]["plan_spec_ref"] = alias.relative_to(root).as_posix()
    spec_path.write_bytes(ExperimentSpec(declaration).to_json().encode())
    planner = Mock()
    monkeypatch.setattr(prospective, "plan_many", planner)
    assert _invoke(capsys, root, spec_path, 2)["status"] == "ERROR"
    planner.assert_not_called()


def _planned(spec):
    bindings = spec.to_dict()["prospective"]["arms"]
    return {
        "status": "PLANNED",
        "results": [
            {"trajectory_id": item["trajectory_id"], "status": "PLANNED", "path": item["output"]}
            for item in bindings
        ],
        "batch": {"size": len(bindings)},
    }


def test_cli_calls_one_batch_and_leaves_derived_pending(capsys, monkeypatch, planning_case):
    root, spec_path, spec = planning_case
    planner = Mock(return_value=_planned(spec))
    monkeypatch.setattr(prospective, "plan_many", planner)
    result = _invoke(capsys, root, spec_path, 0)
    assert result["status"] == "PROSPECTIVE_PLANNED_RML_PENDING"
    assert result["scope"] == "prospective_planning_only"
    assert result["submission_authorized"] is False
    assert result["training_authorized"] is False
    assert result["ready_for_desktop"] is False
    planner.assert_called_once()
    args = planner.call_args.args
    assert args[0] == root
    assert [item["output"] for item in args[1]] == [
        item["output"] for item in spec.to_dict()["prospective"]["arms"]
    ]
    assert result["rml_rebuilt"] is False
    assert result["rml_rebuild_pending"] is True


def test_reference_candidate_spec_freezes_same_run_pair(monkeypatch, planning_case):
    root, _, original = planning_case
    declaration = original.to_dict()
    candidate = declaration["arms"][1]
    candidate["scientific_role"] = "candidate"
    declaration["prospective"]["same_run_replay"] = {
        "reference_arm_id": declaration["arms"][0]["arm_id"],
        "candidate_arm_ids": [candidate["arm_id"]],
    }
    binding = declaration["prospective"]["arms"][1]
    path = root / binding["plan_spec_ref"]
    plan_input = json.loads(path.read_text())
    plan_input["trajectory"]["state_at_start"]["source_config_identity"] = canonical_fingerprint(candidate)
    raw = json.dumps(plan_input, sort_keys=True, separators=(",", ":")).encode()
    path.write_bytes(raw)
    binding["plan_spec_sha256"] = hashlib.sha256(raw).hexdigest()
    spec = ExperimentSpec(declaration)
    planner = Mock(return_value=_planned(spec))
    monkeypatch.setattr(prospective, "plan_many", planner)
    result, code = prospective.plan_prospective(spec, root)
    assert code == 0 and result["rml_rebuild_pending"] and not result["rml_rebuilt"]
    plans = planner.call_args.args[1]
    reference_pair = plans[0]["spec"]["trajectory"]["state_at_start"]["same_run_replay"]
    candidate_pair = plans[1]["spec"]["trajectory"]["state_at_start"]["same_run_replay"]
    assert reference_pair["comparison_role"] == "reference"
    assert candidate_pair["comparison_role"] == "candidate"
    assert candidate_pair["spec_identity"] == spec.identity
    assert candidate_pair["reference_trajectory_id"] == reference_pair["reference_trajectory_id"]
    assert candidate_pair["reference_trajectory_ref"] == "experiments/planned-gptrans_t/trajectory.json"


def test_cli_partial_batch_does_not_rebuild(capsys, monkeypatch, planning_case):
    root, spec_path, spec = planning_case
    first = _planned(spec)["results"][0]
    planner = Mock(side_effect=PlanBatchError(1, 2, [first], ValueError("second failed")))
    monkeypatch.setattr(prospective, "plan_many", planner)
    result = _invoke(capsys, root, spec_path, 1)
    assert result["status"] == "PARTIAL_PROSPECTIVE_PLAN_REQUIRES_RECONCILIATION"
    assert result["completed_records"] == [first]
    assert result["failed_index"] == 1
    assert result["submission_authorized"] is False


@pytest.mark.parametrize("fault", [None, "authority", "unknown", "duplicate", "readiness"])
def test_cli_checks_only_referenced_evidence_before_publication(capsys, monkeypatch, planning_case, fault):
    from test_comparison_readiness import _assess
    from test_v5_common import _envelope
    root, spec_path, spec = planning_case
    authority = root / "authority.json"
    authority.write_bytes(b"{}")
    referenced = _envelope(authority.name)
    readiness_ref = "experiments/reference/comparison_readiness.json"
    referenced["authority"]["pointers"].append(readiness_ref)
    readiness = _assess()
    for side in ("candidate_artifact_bindings", "reference_artifact_bindings"):
        for name, binding in readiness[side].items():
            artifact = root / binding["ref"]
            artifact.parent.mkdir(parents=True, exist_ok=True)
            artifact.write_bytes((side + ":" + name).encode())
            binding["sha256"] = hashlib.sha256(artifact.read_bytes()).hexdigest()
    checkpoint = root / readiness["reference_artifact_bindings"]["checkpoint"]["ref"]
    path = root / "experiments/reference/v5_evidence.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(referenced), encoding="utf-8")
    (root / readiness_ref).write_text(json.dumps(readiness), encoding="utf-8")
    # Unrelated historical authority is unavailable; it is only knowledge metadata.
    history = _envelope("missing-history-authority.json")
    history["evidence_id"] = "evidence-unrelated-history"
    historical_path = root / "experiments/history/v5_evidence.json"
    historical_path.parent.mkdir(parents=True)
    historical_path.write_text(json.dumps(history), encoding="utf-8")
    declaration = spec.to_dict()
    for binding in declaration["prospective"]["arms"]:
        plan_path = root / binding["plan_spec_ref"]
        plan = json.loads(plan_path.read_bytes())
        plan["trajectory"]["state_at_start"]["reference_ids"] = [referenced["evidence_id"]]
        if fault == "unknown":
            plan["trajectory"]["state_at_start"]["reference_ids"] = ["evidence-unknown"]
        raw = json.dumps(plan).encode()
        plan_path.write_bytes(raw)
        binding["plan_spec_sha256"] = hashlib.sha256(raw).hexdigest()
    spec = ExperimentSpec(declaration)
    spec_path.write_text(spec.to_json(), encoding="utf-8")
    if fault == "authority":
        authority.unlink()
    elif fault == "duplicate":
        historical_path.write_text(json.dumps(referenced), encoding="utf-8")
    elif fault == "readiness":
        checkpoint.write_bytes(checkpoint.read_bytes() + b"tampered")
    planner = Mock(return_value=_planned(spec))
    monkeypatch.setattr(prospective, "plan_many", planner)
    result = _invoke(capsys, root, spec_path, 2 if fault else 0)
    if fault:
        assert result["status"] == "ERROR"
        if fault == "readiness":
            assert "bound artifact SHA mismatch" in result["error"]["message"]
        planner.assert_not_called()
    else:
        assert result["status"] == "PROSPECTIVE_PLANNED_RML_PENDING"
        assert result["completed_records"] == _planned(spec)["results"]
        planner.assert_called_once()


def test_cli_batch_return_mismatch_never_rebuilds(capsys, monkeypatch, planning_case):
    root, spec_path, spec = planning_case
    wrong = _planned(spec)
    wrong["results"].reverse()
    monkeypatch.setattr(prospective, "plan_many", Mock(return_value=wrong))
    result = _invoke(capsys, root, spec_path, 1)
    assert result["status"] == "PROSPECTIVE_PLAN_RESULT_MISMATCH_REQUIRES_RECONCILIATION"


def test_plan_cli_rejects_v1(capsys, monkeypatch, tmp_path, payload):
    spec_path = tmp_path / "spec.json"
    spec_path.write_bytes(ExperimentSpec(payload).to_json().encode())
    planner = Mock()
    monkeypatch.setattr(prospective, "plan_many", planner)
    assert _invoke(capsys, tmp_path, spec_path, 2)["status"] == "ERROR"
    planner.assert_not_called()


def _v2_terminal_case(tmp_path, case):
    _, descriptor = case
    descriptor = copy.deepcopy(descriptor)
    declaration = case[0].to_dict()
    declaration["schema_version"] = SCHEMA_VERSION_V2
    declaration["prospective"] = {"arms": [
        {
            "arm_id": arm["arm_id"], "trajectory_id": arm["trajectory_id"],
            "plan_spec_ref": f"inputs/{arm['arm_id']}.json",
            "plan_spec_sha256": "0" * 64,
            "output": f"experiments/planned-{arm['arm_id']}",
        }
        for arm in descriptor["arms"]
    ]}
    spec = ExperimentSpec(declaration)
    descriptor["spec_identity"] = spec.identity
    for arm in descriptor["arms"]:
        arm["observed"]["identity"]["spec_identity"] = spec.identity
        path = tmp_path / arm["trajectory"]
        record = json.loads(path.read_text(encoding="utf-8"))
        expected_arm = next(item for item in declaration["arms"] if item["arm_id"] == arm["arm_id"])
        record["state_at_start"]["source_config_identity"] = canonical_fingerprint(expected_arm)
        path.write_text(json.dumps(record), encoding="utf-8")
    return spec, descriptor


def test_terminal_v2_binds_trajectory_and_arm_identity(tmp_path, case):
    spec, payload = _v2_terminal_case(tmp_path, case)
    descriptor = TerminalDescriptor(spec, payload)
    assert len(translate_terminal_descriptor(tmp_path, spec, descriptor)) == 2
    wrong_id = copy.deepcopy(payload)
    wrong_id["arms"][0]["trajectory_id"] = "T-wrong"
    with pytest.raises(ValueError, match="prospective trajectory identity"):
        TerminalDescriptor(spec, wrong_id)
    path = tmp_path / payload["arms"][0]["trajectory"]
    record = json.loads(path.read_text(encoding="utf-8"))
    record["state_at_start"]["source_config_identity"] = "wrong"
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="prospective trajectory arm identity"):
        translate_terminal_descriptor(tmp_path, spec, descriptor)


def test_terminal_v1_still_uses_existing_contract(tmp_path, case):
    spec, payload = case
    assert len(translate_terminal_descriptor(tmp_path, spec, TerminalDescriptor(spec, payload))) == 2
