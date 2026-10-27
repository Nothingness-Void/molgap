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


def test_explicit_decision_and_observed_audit_enable_reviewed_closure(staging_case):
    root, resolved = staging_case
    decision = root / "explicit_decision.md"
    decision.write_text("Explicit reviewed positive disposition for the synthetic contract.", encoding="utf-8")
    result = stage(staging_case, outcome="POSITIVE_UNDER_CONTRACT", decision_ref="explicit_decision.md",
        role_audit=audit_files(staging_case), _strict_transaction=True)
    assert result["status"] == "STAGED_CLOSURE_READY", result
    for paths in result["paths"].values():
        package = json.loads(Path(paths["terminal"]).read_bytes())
        assert package["decision"]["outcome"] == "POSITIVE_UNDER_CONTRACT"
        assert package["evidence"]["outcome"]["scientific_status"] == "POSITIVE_UNDER_CONTRACT"
        assert package["trace_manifest"]["backtest_eligibility"]["eligible"] is False
        assert package["artifact_hashes"]["explicit_decision.md"] == file_digest(decision)
    ready = stage(staging_case, outcome="READY_FOR_DESKTOP", decision_ref="explicit_decision.md",
        role_audit=audit_files(staging_case), _strict_transaction=True)
    assert ready["status"] == "STAGED_DRAFT"
    assert all("positive_or_ready_requires_owning_scientific_acceptance" in paths["blockers"]
        for paths in ready["paths"].values())
    descriptor = build_verified_terminal_descriptor(root, resolved["spec"], outputs=resolved["outputs"], locations=result["locations"])
    closure = close_verified_outputs(root, resolved["spec"], descriptor, outputs=resolved["outputs"], execute=True)
    assert closure["status"] == "COMPLETE", closure
    for where in resolved["locations"].values():
        assert (root / where["trajectory"]).parent.joinpath("rml_finalized/finalization.json").is_file()


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
    if with_audit:
        # Reentry retains the original immutable draft and its rejected closure.
        reviewed = stage(staging_case, **kwargs, _strict_transaction=True)
        assert reviewed["status"] == "STAGED_CLOSURE_READY", reviewed
        assert reviewed["paths"] != result["paths"]
        assert stage(staging_case, **kwargs, _strict_transaction=True) == reviewed


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
    root, resolved = staging_case
    audits = audit_files(staging_case)
    mapping = {arm: json.loads((root / path).read_bytes()) for arm, path in audits.items()}
    audit = root / "audits" / "by_arm.json"
    audit.write_text(json.dumps(mapping), encoding="utf-8")
    result = stage(staging_case, role_audit=audit)
    for arm, paths in result["paths"].items():
        terminal = json.loads(Path(paths["terminal"]).read_bytes())
        assert set(terminal["role_use"].values()) == {"untouched"}
        assert terminal["artifact_hashes"]["audits/by_arm.json"] == file_digest(audit)
    assert set(mapping) == set(resolved["outputs"])
    mapping[next(iter(mapping))]["context"]["source_commit"] = "f" * 40
    audit.write_text(json.dumps(mapping), encoding="utf-8")
    with pytest.raises(ValueError, match="role audit differs"):
        stage(staging_case, role_audit=audit)


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
