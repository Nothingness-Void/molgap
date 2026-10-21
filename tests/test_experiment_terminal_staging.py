"""Compact terminal metadata over original retained bytes; no remote work."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from molgap.experiment_family_workflow import build_verified_terminal_descriptor, close_verified_outputs, inspect_terminal_output
from molgap.experiment_attribution import ATTRIBUTION_FORMAT
from molgap.experiment_terminal_staging import PROTECTED_ROLES, _cost_events, stage_verified_terminal_packages
from molgap.research_memory.paths import artifact_locator, bound_artifact_path, repo_local_path, verify_bound_artifact
from molgap.research_memory.recovery import recover_trace
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


def diagnostic_report(resolved):
    return {"format": ATTRIBUTION_FORMAT,
        "spec_identity": resolved["spec"].identity,
        "run_id": resolved["spec"].to_dict()["logical_run_id"],
        "attempt_id": "v1", "status": "DIAGNOSED",
        "scientific_conclusion": "NOT_EVALUATED"}


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
        assert all("role" in artifact for artifact in package["evidence"]["artifacts"])
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
    with pytest.raises(ValueError, match="unresolved closure blockers"):
        close_verified_outputs(
            root, resolved["spec"], descriptor, outputs=resolved["outputs"], execute=True
        )
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
    report = diagnostic_report(resolved)
    result = stage(staging_case, diagnostic_report=report)
    bindings = []
    for paths in result["paths"].values():
        objective = json.loads(Path(paths["observed"]).read_bytes())
        bindings.append(objective["diagnostic_report"])
    assert bindings[0] == bindings[1]
    assert set(bindings[0]) == {"locator", "sha256", "spec_identity", "logical_run_id"}
    assert len(list(resolved["run_directory"].rglob("terminal_diagnostic_*.json"))) == 1


def test_inline_role_audit_is_persisted_per_arm(staging_case):
    root, resolved = staging_case
    decision = root / "inline_decision.md"
    decision.write_text("Inline role audit closure.", encoding="utf-8")
    role_audit = {
        arm_id: {
            "context": item["context"].to_dict(), "roles": [],
            "role_use": {role: "untouched" for role in PROTECTED_ROLES},
        }
        for arm_id, item in resolved["outputs"].items()
    }
    result = stage(staging_case, outcome="INCONCLUSIVE", decision_ref=decision.relative_to(root).as_posix(),
                   role_audit=role_audit)
    assert result["status"] == "STAGED_CLOSURE_READY", result
    for paths in result["paths"].values():
        package = json.loads(Path(paths["terminal"]).read_bytes())
        assert set(package["role_use"].values()) == {"untouched"}
        assert any("role_audit-" in pointer for pointer in package["artifact_hashes"])


def test_single_inline_role_audit_fills_identity_and_empty_roles(staging_case):
    root, resolved = staging_case
    decision = root / "inline_identity_decision.md"
    decision.write_text("Inline role identity closure.", encoding="utf-8")
    item = next(iter(resolved["outputs"].values()))
    audit = {"role_use": {role: "untouched" for role in PROTECTED_ROLES}}
    result = stage(staging_case, outcome="INCONCLUSIVE", decision_ref=decision.relative_to(root).as_posix(),
                   role_audit=audit)
    assert result["status"] == "STAGED_CLOSURE_READY", result
    package = json.loads(Path(result["paths"][item["context"].arm_id]["terminal"]).read_bytes())
    audit_ref = next(pointer for pointer in package["artifact_hashes"] if "role_audit-" in pointer)
    persisted = json.loads((root / audit_ref).read_bytes())
    assert persisted["context"] == item["context"].to_dict()
    assert persisted["roles"] == []


def test_existing_scientific_terminal_package_is_reused_after_pin_check(staging_case):
    root, resolved = staging_case
    decision = root / "owner_package_decision.md"
    decision.write_text("Owner package closure.", encoding="utf-8")
    owner_audits = audit_files(staging_case)
    owner = stage(staging_case, outcome="INCONCLUSIVE", decision_ref=decision.relative_to(root).as_posix(),
                  role_audit=owner_audits)
    package_paths = {arm_id: paths["terminal"] for arm_id, paths in owner["paths"].items()}
    reused = stage(staging_case, outcome="INCONCLUSIVE", decision_ref=decision.relative_to(root).as_posix(),
                   scientific_acceptance=package_paths)
    assert reused["status"] == "STAGED_CLOSURE_READY", reused
    for arm_id, package_path in package_paths.items():
        assert reused["paths"][arm_id]["terminal"] == package_path
        assert reused["locations"][arm_id]["terminal"] == Path(package_path).relative_to(root).as_posix()


def test_inline_role_events_rebind_to_immutable_audit(staging_case):
    root, resolved = staging_case
    decision = root / "inline_event_decision.md"
    decision.write_text("Inline role event closure.", encoding="utf-8")
    role_audit = {}
    for arm_id, item in resolved["outputs"].items():
        where = resolved["locations"][arm_id]
        trajectory = json.loads((root / where["trajectory"]).read_bytes())
        action = next(action for action in trajectory["actions"] if where["run_id"] in action["run_ids"])
        role_audit[arm_id] = {
            "context": item["context"].to_dict(),
            "roles": [{
                "schema": "molgap-role-event-v1",
                "role_event_id": "role-inline-" + arm_id,
                "trajectory_id": where["trajectory_id"],
                "action_id": action["action_id"],
                "run_id": where["run_id"],
                "dataset_identity": "dataset-synthetic",
                "row_manifest_hash": "a" * 64,
                "role_name": "train",
                "evidence_ref": "caller-owned-audit.json",
                "access_kind": "training_membership",
                "selection_used": False,
            }],
            "role_use": {role: "untouched" for role in PROTECTED_ROLES},
        }
    result = stage(staging_case, outcome="INCONCLUSIVE", decision_ref=decision.relative_to(root).as_posix(),
                   role_audit=role_audit)
    assert result["status"] == "STAGED_CLOSURE_READY", result
    for paths in result["paths"].values():
        package = json.loads(Path(paths["terminal"]).read_bytes())
        audit_refs = [pointer for pointer in package["artifact_hashes"] if "role_audit-" in pointer]
        assert len(audit_refs) == 1
        audit = json.loads((root / audit_refs[0]).read_bytes())
        assert audit["roles"][0]["evidence_ref"] == audit_refs[0]
        assert audit["roles"][0]["role_name"] == "train"


def test_invalid_role_audit_input_fails_before_repo_path_resolution(staging_case):
    with pytest.raises(ValueError, match="role audit must be a repository path or mapping"):
        stage(staging_case, role_audit=object())


def test_external_raw_trace_recovery_keeps_output_repo_local(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    external = tmp_path / "external" / "raw.json"
    external.parent.mkdir()
    external.write_text(json.dumps({"rows": [{"step": 1, "mae": 0.5}]}), encoding="utf-8")
    output = root / "experiments" / "recovered.json"
    semantics = {
        "live_train_metric": {"metric": "MAE", "unit": "eV", "target": "Gap",
            "role_identity": "train", "weights": "live", "direction": "minimize"},
        "live_dev_metric": None, "ema_dev_metric": None,
    }
    spec = {"trajectory_id": "trajectory", "run_id": "run", "rows_key": "rows",
            "metric_semantics": semantics, "field_mapping": {
                "optimizer_step": "step", "live_train_metric": "mae"}}
    result = recover_trace([artifact_locator(root, external)], spec, output, repo_root=root)
    assert output.is_file()
    assert result["provenance"][0]["source"] == str(external.resolve())
    assert output.is_relative_to(root)


def test_explicit_mutable_diagnostic_is_bound_through_immutable_copy(staging_case):
    from molgap.screen_policy import canonical_fingerprint

    root, resolved = staging_case
    source = resolved["run_directory"] / "reports" / "working_summary.json"
    source.parent.mkdir(parents=True, exist_ok=True)
    report = {**diagnostic_report(resolved),
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


@pytest.mark.parametrize("field", ["format", "spec_identity", "run_id", "attempt_id"])
def test_foreign_diagnostic_report_cannot_be_staged(staging_case, field):
    _, resolved = staging_case
    report = diagnostic_report(resolved)
    report[field] = "foreign" if field != "format" else "foreign-attribution-v1"
    with pytest.raises(ValueError, match="Diagnostic report differs"):
        stage(staging_case, diagnostic_report=report)


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
