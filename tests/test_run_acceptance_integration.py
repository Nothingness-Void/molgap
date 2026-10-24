"""CPU-only joined run-ID acceptance path over a frozen synthetic package."""
from __future__ import annotations

import json
import hashlib
import shutil
from dataclasses import replace
from pathlib import Path
import pytest
from molgap.experiment_attribution import ATTRIBUTION_FORMAT
from molgap.experiment_allocation import AllocationLedger
from molgap.experiment_family_workflow import RunContext
from molgap.experiment_launch import canonical_json
from molgap import experiment_cli
from molgap.experiment_run import accept_run_workflow
from molgap.experiment_run_acceptance import resolve_run_acceptance_inputs
from molgap.experiment_retention import seal_execution_retention
from molgap.research_memory.paths import artifact_locator
from molgap.research_memory.validate import validate_repository_records
from molgap.family_training_outputs import SessionOutputs
from test_family_training_session import events, make_session
from run_acceptance_fixtures import frozen_shared_acceptance_run
from test_experiment_package import repo
from molgap.screen_policy import canonical_fingerprint


def _trajectory(binding, arm, context, recipe_path):
    run_id = context.logical_run_id + ":" + context.arm_id + ":downstream"
    action_id = "action-" + context.arm_id.replace("_", "-")
    return {
        "schema": "molgap-trajectory-v1",
        "trajectory_id": binding["trajectory_id"],
        "record_mode": "prospective",
        "track": "B",
        "owner": "server",
        "family_id": "family-" + context.arm_id,
        "question": "Verify frozen run-ID acceptance wiring with synthetic CPU outputs.",
        "hypothesis": {
            "hypothesis_id": "hyp-" + context.arm_id.replace("_", "-"),
            "supporting_evidence_ids": ["ev-reference"],
            "alternative_explanations": ["synthetic fixture"],
            "related_closed_family_ids": ["family-previous"],
            "observed_deficiency": "Only the joined local acceptance path is under test.",
            "changed_mechanism": "Exercise the frozen resolver and existing output owners.",
            "cheapest_falsifier": "The retained output fails mechanical inspection.",
            "expected_native_cost_ref": "cost-synthetic",
            "decision_changed_if_positive": "The local wiring is mechanically coherent.",
            "decision_changed_if_negative": "Keep the wiring blocked for repair.",
        },
        "state_at_start": {
            "source_commit": context.source_commit,
            "source_config_identity": canonical_fingerprint(arm),
            "contract_refs": [recipe_path],
            "reference_ids": ["ev-reference"],
            "parent_trajectory_ids": [],
            "prior_evidence_ids": ["ev-reference"],
            "role_snapshot_refs": ["experiments/synthetic/roles.json"],
            "budget_snapshot_ref": "experiments/synthetic/budget.json",
        },
        "actions": [{
            "action_id": action_id,
            "type": "TRAIN",
            "source_commit": context.source_commit,
            "run_ids": [run_id],
            "attempt_ids": [context.logical_run_id + "-v2"],
            "evidence_refs": ["experiments/synthetic/acceptance.json"],
            "cost_event_ids": [],
        }],
        "result": {"evidence_ids": [], "evidence_refs": []},
        "decision": {
            "outcome": "ACTIVE",
            "final": False,
            "decision_ref": "experiments/synthetic/decision.md",
            "next_allowed_actions": ["INSPECT"],
            "reopen_conditions": [],
        },
    }


def _write_shared_output_for_arm(case, arm, binding, entry):
    from molgap.experiment_package import read_packaged_text

    arm_id = arm["arm_id"]
    context = RunContext.from_launch(case.spec, case.run / "operations/launch_receipt.json",
        case.package, expected_package_identity=case.manifest["package_identity"], arm_id=arm_id)
    directory = case.paths.artifact_root / "retrieved" / arm_id
    contract_path = case.paths.scratch_root / (arm_id + "-frozen-recipe.json")
    contract_path.parent.mkdir(parents=True, exist_ok=True)
    contract_path.write_text(read_packaged_text(case.package, entry["contract"]["path"]), encoding="utf-8")
    record = json.loads(contract_path.read_bytes())
    session = make_session(arm["family"]["name"], development=True,
        schedule_kind="cosine_step", events=events())
    action_id = "action-" + arm_id.replace("_", "-")
    producer_context = replace(context, platform_version=None)
    outputs = SessionOutputs(directory, producer_context, contract=contract_path, adapter=entry["adapter"],
        trajectory_id=binding["trajectory_id"], train_role_identity=record["train_role_identity"],
        action_id=action_id)
    session.loaders.attach_role_audit(outputs.role_audit)
    session.checkpoint, session.selected, session.pass_finished = outputs.checkpoint, outputs.selected, outputs.pass_finished
    session.batch_finished = outputs.batch_finished
    progress = session.run()
    report = outputs.complete(progress, runtime={"platform": context.platform, "account": context.account,
        "precision": "fp32", "source_commit": context.source_commit,
        "source_archive_sha256": context.source_archive_sha256},
        costs=[{"metric": "wall_seconds", "unit": "seconds", "value": 1.,
            "status": "measured", "semantics": "process_wall", "hardware": "synthetic-cpu"}])
    session.close()
    assert report["status"] == "MECHANICALLY_VERIFIED", report
    trajectory = _trajectory(binding, arm, context, entry["contract"]["path"])
    trajectory_path = case.repo / binding["output"] / "trajectory.json"
    trajectory_path.parent.mkdir(parents=True, exist_ok=True)
    trajectory_path.write_bytes(canonical_json(trajectory).encode("utf-8"))
    return context, directory


def _materialize_shared_case(case):
    """Create the retained outputs and canonical locations once per test."""
    declaration = case.spec.to_dict()
    workflow = json.loads((case.run / "inputs/workflow_plan.json").read_bytes())
    plan = json.loads(case.generated[workflow["acceptance_plan"]])
    entries = {item["arm_id"]: item for item in plan["arms"]}
    arms = {item["arm_id"]: item for item in declaration["arms"]}
    locations = {}
    contexts = {}
    for arm_id, arm in arms.items():
        entry = entries[arm_id]
        binding = next(b for b in declaration["prospective"]["arms"] if b["arm_id"] == arm_id)
        contexts[arm_id], output_dir = _write_shared_output_for_arm(case, arm, binding, entry)
        manifest = json.loads((output_dir / "output_manifest.json").read_bytes())
        trace_path = output_dir / manifest["artifacts"]["trace"]["path"]
        locations[arm_id] = {
            "trajectory_id": binding["trajectory_id"],
            "run_id": declaration["logical_run_id"] + ":" + arm_id + ":downstream",
            "trajectory": binding["output"] + "/trajectory.json",
            "terminal": binding["output"] + "/terminal.json",
            "trace": artifact_locator(case.repo, trace_path),
        }
    return declaration, arms, locations, contexts


def test_original_frozen_run_reaches_diagnosis_and_terminal_staging_cpu_only(tmp_path, repo, capsys):
    case = frozen_shared_acceptance_run(tmp_path, repo)
    declaration, arms, locations, _ = _materialize_shared_case(case)

    explicit = resolve_run_acceptance_inputs(*case.original, retrieved_root=case.paths.artifact_root / "retrieved",
        locations=locations)
    assert {arm: item["trace"] for arm, item in explicit["locations"].items()} == {
        arm: item["trace"] for arm, item in locations.items()}

    # A reconciled successor changes only the physical receipt context.  The
    # original prospective locations remain canonical and cannot be reused
    # with a mismatched successor context.
    from copy import deepcopy
    from molgap.experiment_launch import build_launch_receipt
    successor_response = deepcopy(case.receipt["response"])
    successor_response["platform_version"]["value"] = "8"
    for physical in successor_response["physical_runs"]["value"]:
        physical["platform_version"]["value"] = "8"
        physical["run_identity"] = "kernel1/version8"
    successor_receipt = build_launch_receipt(case.spec, case.package,
        expected_package_identity=case.manifest["package_identity"],
        response_json=canonical_json(successor_response))
    successor_path = case.paths.artifact_root / "continuation_receipt.json"
    successor_path.write_bytes(canonical_json(successor_receipt).encode())
    first_arm = next(iter(arms))
    original_manifest_path = case.paths.artifact_root / "retrieved" / first_arm / "output_manifest.json"
    original_manifest = original_manifest_path.read_bytes()
    mismatched_manifest = json.loads(original_manifest)
    mismatched_manifest["context"]["platform_version"] = "7"
    original_manifest_path.write_bytes(canonical_json(mismatched_manifest).encode())
    try:
        with pytest.raises(ValueError, match="context/source/arm mismatch: platform_version"):
            resolve_run_acceptance_inputs(*case.original,
                retrieved_root=case.paths.artifact_root / "retrieved", continuation_receipt=successor_path)
    finally:
        original_manifest_path.write_bytes(original_manifest)
    resumed = resolve_run_acceptance_inputs(*case.original,
        retrieved_root=case.paths.artifact_root / "continued", continuation_receipt=successor_path)
    assert {context.platform_version for context in resumed["contexts"].values()} == {"8"}
    assert resumed["options"]["receipt_path"] == successor_path
    assert {arm: where["trajectory"] for arm, where in resumed["locations"].items()} == {
        arm: where["trajectory"] for arm, where in explicit["locations"].items()}
    with pytest.raises(ValueError, match="later attempt"):
        resolve_run_acceptance_inputs(*case.original,
            retrieved_root=case.paths.artifact_root / "retrieved",
            continuation_receipt=case.run / "operations/launch_receipt.json")

    result = accept_run_workflow(*case.original, retrieved_root=case.paths.artifact_root / "retrieved",
        diagnose=True, stage_terminal=True)
    assert result["status"] == "MECHANICALLY_VERIFIED", result
    assert {arm: report["status"] for arm, report in result["arms"].items()} == {
        arm: "MECHANICALLY_VERIFIED" for arm in arms}
    assert set(result["diagnosis"]["arms"]) == set(arms)
    assert result["diagnosis"]["format"] == ATTRIBUTION_FORMAT
    assert result["diagnosis"]["status"] == "DIAGNOSED", json.dumps(result["diagnosis"], indent=2)
    assert all(report["status"] == "MECHANICALLY_VERIFIED" for report in result["diagnosis"]["arms"].values())
    assert result["terminal_staging"]["status"] == "STAGED_DRAFT"
    assert set(result["terminal_staging"]["paths"]) == set(arms)
    assert all(result["terminal_staging"]["paths"][arm]["closure_ready"] is False for arm in arms)
    assert all(result["terminal_staging"]["locations"][arm]["trace"].startswith("external://") for arm in arms)

    before = sorted(path.relative_to(case.run).as_posix() for path in case.run.rglob("*") if path.is_file())
    capsys.readouterr()
    exit_code = experiment_cli.main(["accept-workflow", "--run-id", case.spec.to_dict()["logical_run_id"],
        "--attempt-id", "v1", "--repo-root", str(case.repo), "--profile", str(case.profile),
        "--retrieved-root", str(case.paths.artifact_root / "retrieved"), "--diagnose", "--stage-terminal"])
    cli_result = json.loads(capsys.readouterr().out)
    assert exit_code == 0, cli_result
    assert cli_result["diagnosis"]["status"] == "DIAGNOSED", cli_result
    assert cli_result["terminal_staging"]["status"] == "STAGED_DRAFT", cli_result
    after = sorted(path.relative_to(case.run).as_posix() for path in case.run.rglob("*") if path.is_file())
    assert after == before


def _seal_shared_retention(case, root):
    arm_ids = [arm["arm_id"] for arm in case.spec.to_dict()["arms"]]
    arm_roots = {arm_id: root / arm_id for arm_id in arm_ids}
    clock = [0.0]
    ledger = AllocationLedger(
        spec_identity=case.spec.identity,
        hardware=["Tesla T4", "Tesla T4"],
        assignments={arm_id: index for index, arm_id in enumerate(arm_ids)},
        started=0.0,
        clock=lambda: clock[0],
    )
    clock[0] = 1.0
    root.mkdir()
    ledger.write(root, "complete", arm_roots=list(arm_roots.values()))
    return seal_execution_retention(
        root, case.spec, package_identity=case.manifest["package_identity"],
        source_commit=case.manifest["source_commit"],
        source_archive_sha256=case.manifest["archive_sha256"],
        arm_roots=arm_roots,
    )


def _seed_reference_records(case):
    """Provide the small frozen pointers needed by real RML validation."""
    root = case.repo
    directory = root / "experiments" / "synthetic"
    directory.mkdir(parents=True, exist_ok=True)
    decision = directory / "decision.md"
    decision.write_text("Synthetic reference state.\n", encoding="utf-8")
    acceptance = directory / "acceptance.json"
    acceptance.write_bytes(canonical_json({"format": "synthetic-acceptance-v1", "status": "retained"}).encode())
    (directory / "roles.json").write_bytes(canonical_json({"format": "synthetic-role-snapshot-v1"}).encode())
    (directory / "budget.json").write_bytes(canonical_json({"format": "synthetic-budget-v1"}).encode())
    decision_ref = decision.relative_to(root).as_posix()
    acceptance_ref = acceptance.relative_to(root).as_posix()
    digest = hashlib.sha256(decision.read_bytes()).hexdigest()
    evidence = {
        "format": "molgap-v5-evidence-envelope-v1",
        "contract": "MOLGAP-COMMON-V5-FINAL",
        "evidence_id": "ev-reference",
        "track": "B", "scope": "synthetic", "legacy_contract": "synthetic-reference",
        "outcome": {"execution_status": "complete", "artifact_status": "retained",
            "comparison_status": "context_only", "scientific_status": "INCONCLUSIVE",
            "transfer_status": "not_evaluated", "budget_decision": "retained",
            "full_handoff_status": "not_applicable"},
        "authority": {"pointers": [decision_ref]},
        "artifacts": [{"name": "decision", "locator": decision_ref,
            "availability": "durable_local_verified", "sha256": digest}],
        "role_use": {"official_validation": "untouched", "test_dev": "untouched",
            "test_challenge": "untouched"},
        "migration": {"migrated_at": "2026-10-22", "verification_scope": "synthetic fixture",
            "training_executed": False, "inference_executed": False,
            "scientific_reinterpretation": False},
    }
    (directory / "v5_evidence.json").write_bytes(canonical_json(evidence).encode())
    trajectory_id = case.spec.to_dict()["prospective"]["arms"][0]["trajectory_id"]
    action_id = "action-" + case.spec.to_dict()["arms"][0]["arm_id"].replace("_", "-")
    cost = {"schema": "molgap-cost-event-v1", "cost_event_id": "cost-synthetic",
        "trajectory_id": trajectory_id, "action_id": action_id,
        "run_id": "synthetic-run", "attempt_id": "v1", "platform": "synthetic",
        "hardware": "synthetic-cpu", "category": "other", "evidence_ref": acceptance_ref,
        "measurement": {field: {"value": None, "status": "measurement_missing"}
            for field in ("device_hours", "cpu_hours", "wall_hours", "queue_hours")}}
    costs = directory / "costs" / "cost-synthetic.json"
    costs.parent.mkdir(parents=True, exist_ok=True)
    costs.write_bytes(canonical_json(cost).encode())


def test_acceptance_without_diagnosis_preserves_execution_retention(tmp_path, repo):
    case = frozen_shared_acceptance_run(tmp_path, repo)
    _materialize_shared_case(case)
    retention_root = tmp_path / "execution-retention"
    sealed = _seal_shared_retention(case, retention_root)

    result = accept_run_workflow(*case.original,
        retrieved_root=case.paths.artifact_root / "retrieved",
        execution_root=retention_root)

    assert result["status"] == "MECHANICALLY_VERIFIED", result
    assert result.get("diagnosis") is None
    assert result["execution_retention"]["status"] == "EXECUTION_RETENTION_VERIFIED"
    assert result["execution_retention"]["manifest"] == sealed["manifest"]


def test_top_level_execute_restores_frozen_generated_metadata(tmp_path, repo, monkeypatch):
    case = frozen_shared_acceptance_run(tmp_path, repo)
    # Keep the local recovery attempt at v2 while the observed launch receipt
    # records physical platform version 1. The action attempt must come from
    # the frozen trajectory, independently of the receipt version.
    from molgap.experiment_launch import build_launch_receipt
    response = json.loads(json.dumps(case.receipt["response"]))
    response["platform_version"] = {"value": "1", "missing_reason": None}
    for physical in response["physical_runs"]["value"]:
        physical["run_identity"] = "kernel1/version1"
        physical["platform_version"] = {"value": "1", "missing_reason": None}
    receipt = build_launch_receipt(case.spec, case.package,
        expected_package_identity=case.manifest["package_identity"],
        response_json=canonical_json(response))
    case.run.joinpath("operations/launch_receipt.json").write_bytes(canonical_json(receipt).encode())
    journal = json.loads(case.run.joinpath("operations/submission_state.json").read_bytes())
    journal["kernel_response"]["version_number"] = 1
    case.run.joinpath("operations/submission_state.json").write_bytes(canonical_json(journal).encode())
    v2_run = case.paths.run_directory(case.spec.to_dict()["logical_run_id"], "v2")
    shutil.copytree(case.run, v2_run)
    case.run, case.prepared, case.package = v2_run, v2_run / "prepared", v2_run / "prepared/package"
    case.original = (case.profile, case.repo, case.spec.to_dict()["logical_run_id"], "v2")
    case.receipt = receipt
    declaration, arms, locations, contexts = _materialize_shared_case(case)
    _seed_reference_records(case)
    decisions = {}
    for arm_id, item in contexts.items():
        decision = case.repo / (arm_id + "-decision.md")
        decision.write_text("Infrastructure-only terminal disposition.", encoding="utf-8")
        decisions[arm_id] = decision.relative_to(case.repo).as_posix()

    # The resolver reads frozen package members.  Remove every generated
    # checkout file so terminal staging must restore only trajectory-pinned
    # metadata before execution.
    for name in case.generated:
        (case.repo / name).unlink()

    retention_root = tmp_path / "execution-retention"
    sealed = _seal_shared_retention(case, retention_root)
    import molgap.experiment_family_workflow as family_workflow
    import molgap.experiment_package as experiment_package
    counts = {"package_snapshot": 0, "capture": 0}
    original_snapshot = experiment_package._verify_package_snapshot
    original_capture = family_workflow.capture_output_inspection

    def counted_snapshot(*args, **kwargs):
        counts["package_snapshot"] += 1
        return original_snapshot(*args, **kwargs)

    def counted_capture(*args, **kwargs):
        counts["capture"] += 1
        return original_capture(*args, **kwargs)

    monkeypatch.setattr(experiment_package, "_verify_package_snapshot", counted_snapshot)
    monkeypatch.setattr(family_workflow, "capture_output_inspection", counted_capture)
    result = accept_run_workflow(*case.original,
        retrieved_root=case.paths.artifact_root / "retrieved", locations=locations,
        execute=True, execution_root=retention_root, stage_terminal=True, outcome="INCONCLUSIVE",
        decision_ref=decisions)

    assert result["status"] == "COMPLETE", result
    assert result["executed"] is True
    assert result.get("diagnosis") is None
    assert counts == {"package_snapshot": 1, "capture": len(arms)}, counts
    assert result["execution_retention"]["status"] == "EXECUTION_RETENTION_VERIFIED"
    assert result["execution_retention"]["manifest"] == sealed["manifest"]
    assert result["terminal_staging"]["status"] == "STAGED_CLOSURE_READY"
    for arm_id, where in locations.items():
        trajectory = json.loads((case.repo / where["trajectory"]).read_bytes())
        recipe = trajectory["state_at_start"]["contract_refs"][0]
        restored = case.repo / recipe
        assert restored.is_file()
        assert restored.read_bytes() == case.generated[recipe]
        staged_terminal = result["terminal_staging"]["paths"][arm_id]["terminal"]
        package = json.loads((case.repo / staged_terminal).read_bytes())
        assert package["artifact_hashes"][recipe] == case.manifest["generated_records"][recipe]
        assert {cost["attempt_id"] for cost in package["costs"]} == {
            declaration["logical_run_id"] + "-v2"}
        if "same_run_observation" in package:
            assert package["same_run_observation"]["attempt_id"] == declaration["logical_run_id"] + "-v2"
        observed = json.loads(Path(result["terminal_staging"]["paths"][arm_id]["observed"]).read_bytes())
        assert observed["context"]["platform_version"] == "1"
    validated = validate_repository_records(case.repo)
    assert validated["root"] == case.repo.resolve()
