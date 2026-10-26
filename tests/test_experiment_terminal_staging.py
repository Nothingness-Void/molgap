"""Compact terminal metadata over original retained bytes; no remote work."""
from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from molgap.experiment_family_workflow import (build_verified_terminal_descriptor, close_verified_outputs,
    inspect_terminal_output, prepare_terminal_outputs, terminal_attempt_id)
from molgap.experiment_terminal_staging import _cost_events, stage_verified_terminal_packages
from molgap.research_memory.paths import artifact_locator, bound_artifact_path, repo_local_path, verify_bound_artifact
from molgap.research_memory.trace import file_digest, json_bytes
from test_experiment_family_workflow import _closure_descriptor, _write_output, launch_contexts, payload, repo


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


def local_staging_case(case, tmp_path, *, bad_plan=None):
    from molgap import experiment_launch as launch
    from molgap.experiment_package import build_experiment_source_package
    from molgap.experiment_spec import ExperimentSpec
    from test_experiment_package import git
    from test_experiment_workflow import _spec_payload

    root, resolved = case
    original = resolved["spec"].to_dict()
    declaration = _spec_payload(original["arms"])
    declaration.update(experiment_id=original["experiment_id"], logical_run_id=original["logical_run_id"])
    plans = []
    for binding in declaration["prospective"]["arms"]:
        arm_id = binding["arm_id"]
        where = resolved["locations"][arm_id]
        path = root / where["trajectory"]
        trajectory = json.loads(path.read_bytes())
        trajectory["state_at_start"]["source_config_identity"] = resolved["outputs"][arm_id]["context"].arm_identity
        trajectory["actions"][0]["attempt_ids"] = [arm_id + "-v2"]
        path.write_bytes(json_bytes(trajectory))
        binding.update(trajectory_id=where["trajectory_id"], output=path.parent.relative_to(root).as_posix())
        plan = json.loads(json.dumps(trajectory))
        if bad_plan == "run":
            plan["actions"][0]["run_ids"] = ["another-logical-run:" + arm_id + ":downstream"]
        plan_path = root / binding["plan_spec_ref"]
        plan_path.parent.mkdir(exist_ok=True)
        plan_path.write_bytes(json_bytes({"trajectory": plan}))
        binding["plan_spec_sha256"] = "f" * 64 if bad_plan == "sha" else file_digest(plan_path)
        plans.append(binding["plan_spec_ref"])
    git(root, "add", "--", *plans)
    git(root, "-c", "commit.gpgsign=false", "commit", "-m", "Synthetic local frozen plans")
    spec = ExperimentSpec(declaration)
    package = tmp_path / "local-package"
    manifest = build_experiment_source_package(spec, root, ["src/example.py", "README.md"], package)
    known = lambda value: {"value": value, "missing_reason": None}
    base = launch.build_launch_receipt(spec, package, expected_package_identity=manifest["package_identity"])
    reference = "synthetic-host/" + declaration["logical_run_id"]
    response = {"format": launch.RESPONSE_FORMAT, "version": 1, "mode": "observed",
        "outcome": "accepted", "conflict_kind": None, "binding": base["binding"],
        "canonical_platform_reference": known(reference), "platform_version": known("1"),
        "physical_runs": known([{"run_identity": "synthetic-host/pid123/retained-start",
            "canonical_reference": reference, "platform_version": known("1"),
            "arm_ids": [arm["arm_id"] for arm in declaration["arms"]]}]),
        "timestamp": {"value": None, "missing_reason": "not_observed"},
        "monitor_paths": {"value": None, "missing_reason": "not_observed"}}
    receipt = launch.reconcile_platform_response(spec, package, launch.canonical_json(response),
        expected_package_identity=manifest["package_identity"])
    receipt_dir = tmp_path / "local-receipt"
    receipt_dir.mkdir()
    receipt_path = launch.write_launch_receipt(launch.canonical_json(receipt), receipt_dir, spec, package,
        expected_package_identity=manifest["package_identity"])
    supplied = {}
    parent = tmp_path / "local-outputs"
    parent.mkdir()
    for arm_id, item in resolved["outputs"].items():
        context = replace(item["context"], spec_identity=spec.identity, source_commit=manifest["source_commit"],
            source_archive_sha256=manifest["archive_sha256"], package_identity=manifest["package_identity"],
            account="synthetic-host", run_reference=reference, platform_version=None)
        trajectory_path = root / resolved["locations"][arm_id]["trajectory"]
        trajectory = json.loads(trajectory_path.read_bytes())
        trajectory["state_at_start"]["source_commit"] = context.source_commit
        trajectory["actions"][0]["source_commit"] = context.source_commit
        trajectory_path.write_bytes(json_bytes(trajectory))
        folder, expected, _, _ = _write_output(parent, context, "gptrans-v1" if arm_id == "gptrans_t" else "k1-v1")
        supplied[arm_id] = {"output_dir": str(folder), "expected": expected}
    outputs = prepare_terminal_outputs(spec, root, supplied, receipt_path=receipt_path, package_dir=package,
        expected_package_identity=manifest["package_identity"])
    resolved.update(spec=spec, outputs=outputs, options={"package_dir": package})
    return root, resolved, receipt_path


def test_local_staging_uses_frozen_logical_attempt_without_relabeling_launch(staging_case, tmp_path):
    from molgap.experiment_spec import ExperimentSpec

    root, resolved, receipt_path = local_staging_case(staging_case, tmp_path)
    originals = {path: path.read_bytes() for item in resolved["outputs"].values()
                 for path in item["output_dir"].iterdir() if path.is_file()}
    originals[receipt_path] = receipt_path.read_bytes()
    for binding in resolved["spec"].to_dict()["prospective"]["arms"]:
        path = root / binding["plan_spec_ref"]
        originals[path] = path.read_bytes()
    decision = root / "explicit_decision.md"
    decision.write_text("Observed infrastructure-only closure.", encoding="utf-8")
    result = stage(staging_case, outcome="INCONCLUSIVE", decision_ref="explicit_decision.md",
                   role_audit=audit_files(staging_case))
    assert result["status"] == "STAGED_CLOSURE_READY", result
    descriptor = build_verified_terminal_descriptor(root, resolved["spec"], outputs=resolved["outputs"],
                                                    locations=result["locations"])
    for arm in descriptor.to_dict()["arms"]:
        arm_id = arm["arm_id"]
        assert arm["observed"]["identity"]["attempt_id"]["value"] == arm_id + "-v2"
        assert resolved["outputs"][arm_id]["context"].platform_version == "1"
        assert result["arms"][arm_id]["observed"]["producer_platform_version"] is None
        terminal = json.loads(Path(result["paths"][arm_id]["terminal"]).read_bytes())
        binding = next(b for b in resolved["spec"].to_dict()["prospective"]["arms"] if b["arm_id"] == arm_id)
        assert terminal["artifact_hashes"][binding["plan_spec_ref"]] == binding["plan_spec_sha256"]
        nonlocal_spec = resolved["spec"].to_dict()
        nonlocal_spec["platform"]["name"] = "kaggle"
        assert terminal_attempt_id(ExperimentSpec(nonlocal_spec), arm_id, "1", package_dir=tmp_path / "absent") == arm_id + "-v1"
    changed = resolved["spec"].to_dict()
    changed["logical_run_id"] = "different-logical-run"
    with pytest.raises(ValueError, match="package Spec identity mismatch"):
        terminal_attempt_id(ExperimentSpec(changed), arm_id, "1", package_dir=resolved["options"]["package_dir"])
    closure = close_verified_outputs(root, resolved["spec"], descriptor, outputs=resolved["outputs"], execute=True)
    assert closure["status"] == "COMPLETE", closure
    assert all(path.read_bytes() == raw for path, raw in originals.items())


@pytest.mark.parametrize("bad_plan,error", [("run", "frozen downstream action/attempt"), ("sha", "plan SHA mismatch")])
def test_local_staging_rejects_wrong_logical_plan_or_hash(staging_case, tmp_path, bad_plan, error):
    local_staging_case(staging_case, tmp_path, bad_plan=bad_plan)
    with pytest.raises(ValueError, match=error):
        stage(staging_case)


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
