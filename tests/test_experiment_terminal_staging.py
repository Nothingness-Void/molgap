"""Compact terminal metadata over original retained bytes; no remote work."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from molgap.experiment_family_workflow import build_verified_terminal_descriptor, close_verified_outputs, inspect_terminal_output
from molgap.experiment_terminal_staging import _cost_events, stage_verified_terminal_packages
from molgap.research_memory.paths import artifact_locator, bound_artifact_path, repo_local_path, verify_bound_artifact
from molgap.research_memory.trace import file_digest
from test_experiment_family_workflow import _closure_descriptor, launch_contexts, payload, repo


@pytest.fixture
def staging_case(tmp_path, repo, launch_contexts):
    spec, descriptor, outputs = _closure_descriptor(tmp_path, repo, launch_contexts)
    locations = {arm["arm_id"]: {key: arm[key] for key in
        ("trajectory_id", "run_id", "trajectory", "terminal", "trace")}
        for arm in descriptor.to_dict()["arms"]}
    for arm_id, item in outputs.items():
        external = tmp_path / "retrieved" / arm_id
        external.parent.mkdir(exist_ok=True)
        shutil.move(str(item["output_dir"]), external)
        item["output_dir"] = external
    resolved = {"spec": spec, "outputs": outputs, "locations": locations,
                "run_directory": tmp_path / "original-run"}
    return repo, resolved


def stage(case, **kwargs):
    root, resolved = case
    return stage_verified_terminal_packages(None, root, "synthetic-run", "v1", resolved=resolved, **kwargs)


def test_external_bound_bytes_keep_metadata_boundary(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    retained = tmp_path / "retained weights.pt"
    retained.write_bytes(b"real retained bytes")
    locator = artifact_locator(root, retained)
    assert locator.startswith("external:///")
    assert bound_artifact_path(root, locator) == retained
    verify_bound_artifact(root, locator, file_digest(retained))
    with pytest.raises(ValueError):
        repo_local_path(root, locator)
    retained.write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA mismatch"):
        verify_bound_artifact(root, locator, "a" * 64)


@pytest.mark.parametrize("locator", ["external://host/file", "external:///", "external:///tmp/../file",
    "external:///tmp/%2e%2e/file", "external:///tmp/file?other=1", "external:///tmp/file#frag"])
def test_external_custody_rejects_unsafe_locators(tmp_path, locator):
    with pytest.raises(ValueError):
        bound_artifact_path(tmp_path, locator)


def test_stage_draft_needs_no_existing_terminal_and_preserves_large_bytes(staging_case):
    root, resolved = staging_case
    originals = {str(path): path.read_bytes() for item in resolved["outputs"].values()
                 for path in item["output_dir"].iterdir() if path.is_file()}
    for where in resolved["locations"].values():
        (root / where["terminal"]).unlink()
    result = stage(staging_case)
    assert result["status"] == "STAGED_DRAFT"
    for arm_id, paths in result["paths"].items():
        package = json.loads(Path(paths["terminal"]).read_bytes())
        assert package["format"] == "molgap-rml-terminal-package-v1"
        assert package["decision"]["outcome"] == "ACTIVE"
        assert set(package["role_use"].values()) == {"unknown"}
        assert package["trace_manifest"]["backtest_eligibility"]["eligible"] is False
        assert result["locations"][arm_id]["trace"].startswith("external:///")
        assert not (root / resolved["locations"][arm_id]["trajectory"]).parent.joinpath("rml_finalized").exists()
    assert all(Path(path).read_bytes() == content for path, content in originals.items())
    assert not list(root.rglob("*.pt"))
    assert stage(staging_case) == result


def audit_files(case):
    root, resolved = case
    audits = {}
    for arm_id, item in resolved["outputs"].items():
        audit = root / "audits" / f"{arm_id}.json"
        audit.parent.mkdir(exist_ok=True)
        audit.write_text(json.dumps({"context": item["context"].to_dict(), "roles": [],
            "role_use": {role: "untouched" for role in ("official_validation", "test_dev", "test_challenge")}}), encoding="utf-8")
        audits[arm_id] = audit.relative_to(root).as_posix()
    return audits


def test_explicit_decision_and_observed_audit_enable_nonpositive_closure(staging_case):
    root, resolved = staging_case
    decision = root / "explicit_decision.md"
    decision.write_text("Explicit infrastructure-only terminal disposition.", encoding="utf-8")
    result = stage(staging_case, outcome="INCONCLUSIVE", decision_ref="explicit_decision.md", role_audit=audit_files(staging_case))
    assert result["status"] == "STAGED_CLOSURE_READY", result
    descriptor = build_verified_terminal_descriptor(root, resolved["spec"], outputs=resolved["outputs"], locations=result["locations"])
    closure = close_verified_outputs(root, resolved["spec"], descriptor, outputs=resolved["outputs"], execute=True)
    assert closure["status"] == "COMPLETE", closure
    for where in resolved["locations"].values():
        assert (root / where["trajectory"]).parent.joinpath("rml_finalized/finalization.json").is_file()


def test_decision_alone_preserves_unknown_roles(staging_case):
    root, _ = staging_case
    (root / "decision.md").write_text("Inconclusive.", encoding="utf-8")
    result = stage(staging_case, outcome="INCONCLUSIVE", decision_ref="decision.md")
    assert result["status"] == "STAGED_DRAFT"
    assert all("observed_protected_role_audit_missing" in item["blockers"] for item in result["paths"].values())


@pytest.mark.parametrize("outcome,with_audit,expected_blocker", [
    ("POSITIVE_UNDER_CONTRACT", True, "positive_or_ready_requires_owning_scientific_acceptance"),
    ("INCONCLUSIVE", False, "observed_protected_role_audit_missing"),
])
def test_staged_blockers_cannot_be_bypassed_by_terminal_execution(
    staging_case, outcome, with_audit, expected_blocker,
):
    root, resolved = staging_case
    decision = root / "explicit_decision.md"
    decision.write_text("Explicit staged disposition.", encoding="utf-8")
    kwargs = {"outcome": outcome, "decision_ref": "explicit_decision.md"}
    if with_audit:
        kwargs["role_audit"] = audit_files(staging_case)

    result = stage(staging_case, **kwargs)

    assert result["status"] == "STAGED_DRAFT", result
    for paths in result["paths"].values():
        terminal = json.loads(Path(paths["terminal"]).read_bytes())
        assert expected_blocker in terminal["closure_blockers"]
        assert terminal["requested_outcome"] == outcome
        assert terminal["decision"]["outcome"] == "ACTIVE"
        assert terminal["decision"]["final"] is False
        if not with_audit:
            assert set(terminal["role_use"].values()) == {"unknown"}

    descriptor = build_verified_terminal_descriptor(
        root, resolved["spec"], outputs=resolved["outputs"], locations=result["locations"]
    )
    closure = close_verified_outputs(
        root, resolved["spec"], descriptor, outputs=resolved["outputs"], execute=True
    )

    assert closure["status"] == "INCOMPLETE", closure
    assert all(arm["pipeline_status"] == "FINALIZATION_FAILED" for arm in closure["results"])
    assert all("unresolved closure blockers" in arm["error"] for arm in closure["results"])
    assert all(not (root / where["trajectory"]).parent.joinpath("rml_finalized").exists()
               for where in resolved["locations"].values())


def test_staging_rechecks_observed_bytes_even_when_report_reused(staging_case):
    _, resolved = staging_case
    reports = {arm_id: inspect_terminal_output(item) for arm_id, item in resolved["outputs"].items()}
    item = next(iter(resolved["outputs"].values()))
    manifest = json.loads((item["output_dir"] / "output_manifest.json").read_bytes())
    (item["output_dir"] / manifest["artifacts"]["selected_model"]["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA mismatch"):
        stage(staging_case, reports=reports)


def test_diagnostic_is_published_once_and_shared_by_pointer(staging_case):
    _, resolved = staging_case
    report = {"spec_identity": resolved["spec"].identity, "status": "DIAGNOSED", "scientific_conclusion": "NOT_EVALUATED"}
    result = stage(staging_case, diagnostic_report=report)
    bindings = []
    for paths in result["paths"].values():
        objective = json.loads(Path(paths["observed"]).read_bytes())
        bindings.append(objective["diagnostic_report"])
    assert bindings[0] == bindings[1]
    assert set(bindings[0]) == {"locator", "sha256", "spec_identity", "logical_run_id"}
    assert len(list(resolved["run_directory"].rglob("terminal_diagnostic_*.json"))) == 1


def test_explicit_mutable_diagnostic_is_bound_through_immutable_copy(staging_case):
    from molgap.screen_policy import canonical_fingerprint

    root, resolved = staging_case
    source = resolved["run_directory"] / "reports" / "working_summary.json"
    source.parent.mkdir(parents=True, exist_ok=True)
    report = {"spec_identity": resolved["spec"].identity, "status": "DIAGNOSED",
        "scientific_conclusion": "NOT_EVALUATED", "written_files": {"analysis_summary": str(source)}}
    source.write_text(json.dumps(report), encoding="utf-8")

    result = stage(staging_case, diagnostic_report=report)

    bindings = [json.loads(Path(paths["observed"]).read_bytes())["diagnostic_report"]
        for paths in result["paths"].values()]
    assert bindings[0] == bindings[1]
    binding = bindings[0]
    immutable = bound_artifact_path(root, binding["locator"])
    assert immutable.name == "terminal_diagnostic_" + canonical_fingerprint(report) + ".json"
    assert json.loads(immutable.read_bytes()) == report
    for paths in result["paths"].values():
        package = json.loads(Path(paths["terminal"]).read_bytes())
        assert package["artifact_hashes"][binding["locator"]] == binding["sha256"]

    source.write_text(json.dumps({**report, "status": "CHANGED"}), encoding="utf-8")
    verify_bound_artifact(root, binding["locator"], binding["sha256"])


def test_role_audit_must_bind_actual_context(staging_case):
    root, _ = staging_case
    audits = audit_files(staging_case)
    audit = root / next(iter(audits.values()))
    obj = json.loads(audit.read_bytes())
    obj["context"]["source_commit"] = "f" * 40
    audit.write_text(json.dumps(obj), encoding="utf-8")
    with pytest.raises(ValueError, match="role audit differs"):
        stage(staging_case, role_audit=audits)


def test_cost_conversion_requires_measured_native_hardware_and_exact_semantics():
    costs = [{"metric": "device_seconds", "unit": "seconds", "value": 3600, "status": "measured",
              "hardware": "Tesla_T4", "semantics": "allocated_device"},
             {"metric": "wall_seconds", "unit": "seconds", "value": None, "status": "missing",
              "hardware": "CPU", "semantics": "process_wall"},
             {"metric": "device_seconds", "unit": "seconds", "value": 10, "status": "estimated",
              "hardware": "Tesla_A100", "semantics": "allocated_device"}]
    events = _cost_events(costs, {"attempt_id": "v1", "platform": "kaggle"},
        {"trajectory_id": "trajectory", "run_id": "run"}, "action", "observed.json")
    assert len(events) == 1
    assert events[0]["hardware"] == "Tesla_T4"
    assert events[0]["measurement"]["device_hours"] == {"status": "measured", "value": 1}
    assert events[0]["measurement"]["wall_hours"] == {"status": "measurement_missing", "value": None}


def trace_plan_fixture():
    from molgap.research_memory.terminal_wiring import build_default_trace_manifest
    from molgap.research_memory.schemas import TRACE_MANIFEST_PLAN_FIELDS

    # Synthetic semantics only; no reference or prospective scientific evidence.
    manifest = build_default_trace_manifest(Path("."),
        {"trajectory_id": "T-plan", "state_at_start": {}}, {"run_id": "run"}, {"observations": []})
    manifest["presentation_semantics_ref"] = "synthetic-presentation-contract"
    return {key: manifest[key] for key in TRACE_MANIFEST_PLAN_FIELDS}


def test_frozen_trace_plan_requires_exact_semantics_and_cannot_claim_eligibility():
    from molgap.research_memory.schemas import validate_trace_manifest_plan

    plan = trace_plan_fixture()
    assert validate_trace_manifest_plan(plan) == plan
    for field in ("reference_id", "backtest_eligibility", "trace_artifact_ref", "costs"):
        with pytest.raises(ValueError, match="exactly"):
            validate_trace_manifest_plan({**plan, field: None})
    with pytest.raises(ValueError, match="trace_fields"):
        validate_trace_manifest_plan({**plan, "trace_fields": {}})
    with pytest.raises(ValueError, match="x_axis"):
        validate_trace_manifest_plan({**plan, "x_axis": "wall_seconds"})


def test_trace_plan_must_come_from_exact_frozen_recipe_bytes(tmp_path):
    from types import SimpleNamespace
    from molgap.experiment_terminal_staging import _frozen_trace_manifest_plan

    contract = tmp_path / "contract.json"
    plan = trace_plan_fixture()
    contract.write_text(json.dumps({"trace_manifest_plan": plan}), encoding="utf-8")
    digest = file_digest(contract)
    artifacts = {"contract": {"path": contract.name, "sha256": digest}}
    context = SimpleNamespace(training_recipe_sha256=digest)
    assert _frozen_trace_manifest_plan(tmp_path, artifacts, context) == plan
    contract.write_text(json.dumps({"trace_manifest_plan": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="frozen recipe"):
        _frozen_trace_manifest_plan(tmp_path, artifacts, context)


def test_stager_consumes_plan_without_granting_replay(staging_case, monkeypatch):
    from molgap.research_memory.schemas import validate_trace_manifest_plan

    plan = trace_plan_fixture()
    plan["model_identity"] = "synthetic-frozen-model"
    plan["trace_fields"] = {key: False for key in plan["trace_fields"]}
    validate_trace_manifest_plan(plan)
    monkeypatch.setattr("molgap.experiment_terminal_staging._frozen_trace_manifest_plan",
        lambda *args: json.loads(json.dumps(plan)))
    result = stage(staging_case)
    for paths in result["paths"].values():
        terminal = json.loads(Path(paths["terminal"]).read_bytes())
        manifest = terminal["trace_manifest"]
        assert manifest["model_identity"] == "synthetic-frozen-model"
        assert manifest["backtest_eligibility"]["eligible"] is False
        assert "strict_comparison_readiness_not_established" in manifest["backtest_eligibility"]["exclusion_reasons"]
        assert "closure_synthesized_manifest_not_prelaunch_calibrated" not in manifest["backtest_eligibility"]["exclusion_reasons"]


def test_stager_rejects_a_promised_but_missing_trace_field(staging_case, monkeypatch):
    plan = trace_plan_fixture()
    plan["trace_fields"] = {key: False for key in plan["trace_fields"]}
    plan["trace_fields"]["checkpoint_identity"] = True
    monkeypatch.setattr("molgap.experiment_terminal_staging._frozen_trace_manifest_plan",
        lambda *args: plan)
    with pytest.raises(ValueError, match="Frozen trace plan field was not observed"):
        stage(staging_case)
