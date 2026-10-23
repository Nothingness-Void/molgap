"""CPU-only joined run-ID acceptance path over a frozen synthetic package."""
from __future__ import annotations

import json
import hashlib
from collections import Counter
from pathlib import Path

import pytest

from molgap.experiment_allocation import AllocationLedger
from molgap.experiment_attribution import ATTRIBUTION_FORMAT
from molgap.experiment_family_workflow import RunContext
from molgap.experiment_launch import canonical_json
from molgap import experiment_cli
from molgap.experiment_run import accept_run_workflow
from molgap.experiment_run_acceptance import resolve_run_acceptance_inputs
from molgap.experiment_retention import seal_execution_retention
from molgap.research_memory.paths import artifact_locator
from molgap.family_training_outputs import SessionOutputs
from test_family_training_session import events, make_session
from run_acceptance_fixtures import frozen_shared_acceptance_run
from test_experiment_package import repo
from molgap.screen_policy import canonical_fingerprint
from molgap.research_memory.validate import validate_repository_records


def _trajectory(binding, arm, context, recipe_path):
    run_id = context.logical_run_id + ":" + context.arm_id + ":downstream"
    action_id = "action-" + context.arm_id.replace("_", "-")
    return {
        "schema": "molgap-trajectory-v1",
        "trajectory_id": binding["trajectory_id"],
        "record_mode": "prospective",
        "track": "B",
        "owner": "desktop",
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
            "attempt_ids": [context.logical_run_id + "-v7"],
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
    outputs = SessionOutputs(directory, context, contract=contract_path, adapter=entry["adapter"],
        trajectory_id=binding["trajectory_id"], train_role_identity=record["train_role_identity"])
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
    """Create the retained outputs and canonical locations once per acceptance journey."""
    declaration = case.spec.to_dict()
    workflow = json.loads((case.run / "inputs/workflow_plan.json").read_bytes())
    plan = json.loads(case.generated[workflow["acceptance_plan"]])
    entries = {item["arm_id"]: item for item in plan["arms"]}
    arms = {item["arm_id"]: item for item in declaration["arms"]}
    locations, contexts = {}, {}
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


def test_original_frozen_run_reaches_diagnosis_and_terminal_staging_cpu_only(tmp_path, repo, capsys, monkeypatch):
    case = frozen_shared_acceptance_run(tmp_path, repo)
    declaration, arms, locations, _ = _materialize_shared_case(case)

    explicit = resolve_run_acceptance_inputs(*case.original, retrieved_root=case.paths.artifact_root / "retrieved",
        locations=locations)
    assert {arm: item["trace"] for arm, item in explicit["locations"].items()} == {
        arm: item["trace"] for arm, item in locations.items()}

    # A reconciled successor changes the physical attempt without rewriting
    # the frozen v7 prospective or weakening its source/package binding.
    from copy import deepcopy
    from molgap.experiment_launch import build_launch_receipt
    successor_response = deepcopy(case.receipt["response"])
    successor_response["platform_version"]["value"] = "8"
    for run in successor_response["physical_runs"]["value"]:
        run["platform_version"]["value"] = "8"
        run["run_identity"] = "kernel1/version8"
    successor_receipt = build_launch_receipt(case.spec, case.package,
        expected_package_identity=case.manifest["package_identity"],
        response_json=canonical_json(successor_response))
    successor_path = case.paths.artifact_root / "continuation_receipt.json"
    successor_path.write_bytes(canonical_json(successor_receipt).encode())
    with pytest.raises(ValueError, match="context/source/arm mismatch: platform_version"):
        resolve_run_acceptance_inputs(*case.original,
            retrieved_root=case.paths.artifact_root / "retrieved", continuation_receipt=successor_path)
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
    assert result["status"] == "STAGED_DRAFT", result
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
    assert exit_code == 1, cli_result
    assert cli_result["diagnosis"]["status"] == "DIAGNOSED", cli_result
    assert cli_result["terminal_staging"]["status"] == "STAGED_DRAFT", cli_result
    after = sorted(path.relative_to(case.run).as_posix() for path in case.run.rglob("*") if path.is_file())
    assert after == before

    # Same-call reports cannot authenticate bytes replaced before diagnosis.
    import torch
    from molgap import experiment_attribution as attribution
    candidate = declaration["prospective"]["same_run_replay"]["candidate_arm_ids"][0]
    candidate_root = case.paths.artifact_root / "retrieved" / candidate
    manifest = json.loads((candidate_root / "output_manifest.json").read_bytes())
    load = attribution._load_arm_artifacts
    for role in ("predictions", "trace", "manifest"):
        path = (candidate_root / "output_manifest.json" if role == "manifest" else
                candidate_root / manifest["artifacts"][role]["path"])
        retained = path.read_bytes()
        changed = []

        def replace_before_loading(arm_dir, **kwargs):
            if arm_dir == candidate_root and not changed:
                if role == "predictions":
                    payload = torch.load(path, map_location="cpu", weights_only=True)
                    payload["prediction_eV"] = payload["prediction_eV"] + 0.125
                    torch.save(payload, path)
                elif role == "trace":
                    trace = json.loads(retained)
                    trace["observations"][0]["learning_rate"] = 0.125
                    path.write_bytes(canonical_json(trace).encode())
                else:
                    replaced_manifest = json.loads(retained)
                    replaced_manifest["costs"][0]["value"] += 0.125
                    path.write_bytes(canonical_json(replaced_manifest).encode())
                changed.append(True)
            return load(arm_dir, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(attribution, "_load_arm_artifacts", replace_before_loading)
            try:
                replaced = accept_run_workflow(*case.original,
                    retrieved_root=case.paths.artifact_root / "retrieved", diagnose=True)
            finally:
                path.write_bytes(retained)
        assert changed
        diagnosis = replaced["diagnosis"]
        assert diagnosis["status"] == "INSUFFICIENT_EVIDENCE", diagnosis
        assert diagnosis["comparisons"][candidate]["status"] == "INSUFFICIENT_EVIDENCE"
        blocker = ("Family output manifest changed after mechanical inspection" if role == "manifest" else
                   "Family output changed after mechanical inspection: " + role)
        assert diagnosis["arms"][candidate]["analysis_blockers"] == [blocker]

    # Replacement after a read must not change the already verified snapshot.
    from pathlib import Path
    predictions = candidate_root / manifest["artifacts"]["predictions"]["path"]
    retained = predictions.read_bytes()
    read_bytes = Path.read_bytes
    loading, replaced_paths = [], []

    def replace_after_read(path):
        raw = read_bytes(path)
        if loading and path == predictions and not replaced_paths:
            payload = torch.load(predictions, map_location="cpu", weights_only=True)
            payload["prediction_eV"] = payload["prediction_eV"] + 0.125
            torch.save(payload, predictions)
            replaced_paths.append(path)
        return raw

    def load_snapshot(arm_dir, **kwargs):
        if arm_dir != candidate_root:
            return load(arm_dir, **kwargs)
        loading.append(True)
        try:
            return load(arm_dir, **kwargs)
        finally:
            loading.clear()

    with monkeypatch.context() as patch:
        patch.setattr(attribution, "_load_arm_artifacts", load_snapshot)
        patch.setattr(Path, "read_bytes", replace_after_read)
        try:
            snapshot = accept_run_workflow(*case.original,
                retrieved_root=case.paths.artifact_root / "retrieved", diagnose=True)
        finally:
            predictions.write_bytes(retained)
    assert replaced_paths == [predictions]
    assert snapshot["diagnosis"]["status"] == "DIAGNOSED", snapshot
    assert snapshot["diagnosis"]["comparisons"] == result["diagnosis"]["comparisons"]


def _seal_shared_retention(case, root, *, prior_segments=()):
    arm_ids = [arm["arm_id"] for arm in case.spec.to_dict()["arms"]]
    arm_roots = {arm_id: root / arm_id for arm_id in arm_ids}
    clock = [0.0]
    ledger = AllocationLedger(
        spec_identity=case.spec.identity,
        hardware=["Tesla T4", "Tesla T4"],
        assignments={arm_id: index for index, arm_id in enumerate(arm_ids)},
        started=0.0,
        prior_segments=prior_segments,
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
    """Seed minimal synthetic pointers so the real repository validator can run."""
    directory = case.repo / "experiments" / "synthetic"
    directory.mkdir(parents=True, exist_ok=True)
    decision = directory / "decision.md"
    decision.write_text("Synthetic reference state.\n", encoding="utf-8")
    acceptance = directory / "acceptance.json"
    acceptance.write_bytes(canonical_json({"format": "synthetic-acceptance-v1", "status": "retained"}).encode())
    (directory / "roles.json").write_bytes(canonical_json({"format": "synthetic-role-snapshot-v1"}).encode())
    (directory / "budget.json").write_bytes(canonical_json({"format": "synthetic-budget-v1"}).encode())
    decision_ref = decision.relative_to(case.repo).as_posix()
    acceptance_ref = acceptance.relative_to(case.repo).as_posix()
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
            "availability": "durable_local_verified", "sha256": hashlib.sha256(decision.read_bytes()).hexdigest()}],
        "role_use": {"official_validation": "untouched", "test_dev": "untouched",
            "test_challenge": "untouched"},
        "migration": {"migrated_at": "2026-10-22", "verification_scope": "synthetic fixture",
            "training_executed": False, "inference_executed": False,
            "scientific_reinterpretation": False},
    }
    (directory / "v5_evidence.json").write_bytes(canonical_json(evidence).encode())
    declaration = case.spec.to_dict()
    trajectory_id = declaration["prospective"]["arms"][0]["trajectory_id"]
    action_id = "action-" + declaration["arms"][0]["arm_id"].replace("_", "-")
    cost = {"schema": "molgap-cost-event-v1", "cost_event_id": "cost-synthetic",
        "trajectory_id": trajectory_id, "action_id": action_id,
        "run_id": "synthetic-run", "attempt_id": "v1", "platform": "synthetic",
        "hardware": "synthetic-cpu", "category": "other", "evidence_ref": acceptance_ref,
        "measurement": {field: {"value": None, "status": "measurement_missing"}
            for field in ("device_hours", "cpu_hours", "wall_hours", "queue_hours")}}
    costs = directory / "costs" / "cost-synthetic.json"
    costs.parent.mkdir(parents=True, exist_ok=True)
    costs.write_bytes(canonical_json(cost).encode())


def test_top_level_execute_restores_frozen_generated_metadata(tmp_path, repo, monkeypatch):
    case = frozen_shared_acceptance_run(tmp_path, repo)
    declaration, arms, locations, contexts = _materialize_shared_case(case)
    reference_arm = declaration["prospective"]["same_run_replay"]["reference_arm_id"]
    for arm_id, where in locations.items():
        trajectory = json.loads((case.repo / where["trajectory"]).read_bytes())
        trajectory["state_at_start"]["same_run_replay"] = {
            "schema": "molgap-same-run-replay-binding-v1", "spec_identity": case.spec.identity,
            "logical_run_id": declaration["logical_run_id"], "arm_id": arm_id,
            "comparison_role": "reference" if arm_id == reference_arm else "candidate",
            "reference_arm_id": reference_arm,
            "reference_trajectory_id": locations[reference_arm]["trajectory_id"],
            "reference_trajectory_ref": locations[reference_arm]["trajectory"]}
        (case.repo / where["trajectory"]).write_bytes(canonical_json(trajectory).encode())
    _seed_reference_records(case)
    decisions, audits = {}, {}
    for arm_id, context in contexts.items():
        decision = case.repo / (arm_id + "-decision.md")
        decision.write_text("Infrastructure-only terminal disposition.", encoding="utf-8")
        decisions[arm_id] = decision.relative_to(case.repo).as_posix()
        audit = case.repo / "audits" / (arm_id + "-roles.json")
        audit.parent.mkdir(parents=True, exist_ok=True)
        audit.write_bytes(canonical_json({"context": context.to_dict(), "roles": [],
            "role_use": {role: "untouched" for role in
                ("official_validation", "test_dev", "test_challenge")}}).encode())
        audits[arm_id] = audit.relative_to(case.repo).as_posix()

    # Acceptance must recover only the metadata frozen by the package and
    # trajectories after those generated checkout files have been lost.
    for name in case.generated:
        (case.repo / name).unlink()

    retention_root = tmp_path / "execution-retention"
    sealed = _seal_shared_retention(case, retention_root, prior_segments=[{
        "format": "molgap-allocation-ledger-v1", "spec_identity": case.spec.identity,
        "status": "running"}])
    from molgap import experiment_retention as retention
    validations = []
    validate_retention = retention.validate_execution_retention

    def count_retention(*args, **kwargs):
        validations.append(True)
        return validate_retention(*args, **kwargs)

    monkeypatch.setattr(retention, "validate_execution_retention", count_retention)
    from molgap import experiment_family_workflow as family
    inspected = Counter()
    inspect = family.inspect_terminal_output

    def count_inspection(item):
        inspected[item["context"].arm_id] += 1
        return inspect(item)

    monkeypatch.setattr(family, "inspect_terminal_output", count_inspection)
    from molgap import experiment_terminal as terminal
    translations = []
    translate = terminal.translate_terminal_descriptor

    def count_translation(*args, **kwargs):
        translations.append(True)
        return translate(*args, **kwargs)

    monkeypatch.setattr(terminal, "translate_terminal_descriptor", count_translation)
    options = dict(retrieved_root=case.paths.artifact_root / "retrieved", locations=locations,
        execute=True, execution_root=retention_root,
        stage_terminal=True, outcome="INCONCLUSIVE", decision_ref=decisions, role_audit=audits)
    result = accept_run_workflow(*case.original, **options)

    assert result["status"] == "COMPLETE", result
    assert result["executed"] is True
    assert result.get("diagnosis") is None
    assert inspected == Counter({arm_id: 1 for arm_id in arms})
    assert len(translations) == 2
    assert len(validations) == 1
    assert result["execution_retention"]["status"] == "EXECUTION_RETENTION_VERIFIED"
    assert result["execution_retention"]["manifest"] == sealed["manifest"]
    assert result["terminal_staging"]["status"] == "STAGED_CLOSURE_READY"

    trajectory_contracts = set()
    for arm_id, where in locations.items():
        trajectory = json.loads((case.repo / where["trajectory"]).read_bytes())
        recipe = trajectory["state_at_start"]["contract_refs"][0]
        trajectory_contracts.add(recipe)
        restored = case.repo / recipe
        assert restored.is_file()
        assert restored.read_bytes() == case.generated[recipe]
        staged_terminal = result["terminal_staging"]["paths"][arm_id]["terminal"]
        package = json.loads((case.repo / staged_terminal).read_bytes())
        assert package["artifact_hashes"][recipe] == case.manifest["generated_records"][recipe]
        allocation_cost, = [event for event in package["costs"] if event["category"] == "other"]
        assert allocation_cost["hardware"] == "Tesla T4"
        assert allocation_cost["measurement"]["device_hours"] == {"value": 1 / 3600, "status": "measured"}
        assert all(allocation_cost["measurement"][field] == {"value": None, "status": "measurement_missing"}
                   for field in ("wall_hours", "cpu_hours", "queue_hours"))
        assert allocation_cost["attempt_id"] == declaration["logical_run_id"] + "-v7"
        objective = json.loads((case.repo / allocation_cost["evidence_ref"]).read_bytes())
        window = objective["allocation_observation"]
        assert window["devices"] == [{"device": list(arms).index(arm_id), "hardware": "Tesla T4",
            "arm_id": arm_id, "allocated_seconds": 1.0}]
        assert window["prior_segments_recorded"] == 1
        assert window["prior_unobserved_intervals"] is True
        assert window["cumulative_costs_evaluated"] is False
        assert window["event_status"] == "recorded_current_window"
        expected_allocations = {artifact_locator(case.repo, retention_root / name)
            for name in ("execution_retention.json", "allocation_ledger.json", arm_id + "/allocation_ledger.json")}
        assert set(window["artifact_refs"]) == expected_allocations
        assert expected_allocations <= package["artifact_hashes"].keys()
        finalization = case.repo / where["trajectory"]
        assert finalization.parent.joinpath("rml_finalized", "finalization.json").is_file()
        finalized_cost = finalization.parent / "rml_finalized/costs" / (allocation_cost["cost_event_id"] + ".json")
        assert json.loads(finalized_cost.read_bytes()) == allocation_cost

    assert len(trajectory_contracts) == len(arms)
    assert all(not (case.repo / name).exists() for name in set(case.generated) - trajectory_contracts)
    receipts = {arm_id: (case.repo / where["trajectory"]).parent / "rml_finalized/finalization.json"
                for arm_id, where in locations.items()}
    retained = {arm_id: path.read_bytes() for arm_id, path in receipts.items()}
    repeated = accept_run_workflow(*case.original, **options)
    assert repeated["status"] == "COMPLETE", repeated
    assert repeated["terminal_staging"] == result["terminal_staging"]
    assert all(item["finalization_status"] == "ALREADY_FINALIZED" for item in repeated["results"])
    assert inspected == Counter({arm_id: 2 for arm_id in arms})
    assert len(translations) == 4
    assert len(validations) == 2

    # A native allocated-device observation can overlap this retained window.
    # Keep both sources, without adding a second device-hour event.
    one_arm = next(iter(arms))
    manifest_path = case.paths.artifact_root / "retrieved" / one_arm / "output_manifest.json"
    original_manifest = manifest_path.read_bytes()
    try:
        for status, hardware in (("measured", "Tesla T4"), ("estimated", "Tesla T4"), ("measured", "unknown")):
            native = json.loads(original_manifest)
            native["costs"].append({"metric": "device_seconds", "unit": "seconds", "value": 0.5,
                "status": status, "semantics": "allocated_device", "hardware": hardware})
            manifest_path.write_bytes(canonical_json(native).encode())
            draft = accept_run_workflow(*case.original, **{**options, "execute": False})
            terminal_path = draft["terminal_staging"]["paths"][one_arm]["terminal"]
            terminal_package = json.loads((case.repo / terminal_path).read_bytes())
            device_cost, = [event for event in terminal_package["costs"]
                           if event["measurement"]["device_hours"]["status"] == "measured"]
            overlaps = status == "measured" and hardware == "Tesla T4"
            assert device_cost["measurement"]["device_hours"]["value"] == (0.5 if overlaps else 1) / 3600
            assert device_cost["category"] == ("training" if overlaps else "other")
            objective = json.loads((case.repo / device_cost["evidence_ref"]).read_bytes())
            assert objective["allocation_observation"]["event_status"] == (
                "not_aggregated_existing_device_measurement" if overlaps else "recorded_current_window")
            assert objective["native_costs"] == native["costs"]
    finally:
        manifest_path.write_bytes(original_manifest)

    # Same-call reuse must still reject bytes changed after terminal staging.
    from molgap import experiment_terminal_staging as staging
    stage = staging.stage_verified_terminal_packages
    output = case.paths.artifact_root / "retrieved" / next(iter(arms))
    manifest = json.loads((output / "output_manifest.json").read_bytes())
    checkpoint = output / manifest["artifacts"]["selected_model"]["path"]
    for path in (checkpoint, retention_root / "allocation_ledger.json", retention_root / "execution_retention.json"):
        retained_bytes = path.read_bytes()

        def change_after_staging(*args, **kwargs):
            staged = stage(*args, **kwargs)
            path.write_bytes(b"changed after staging")
            return staged

        monkeypatch.setattr(staging, "stage_verified_terminal_packages", change_after_staging)
        try:
            expected_error = "bound artifact SHA mismatch: " + artifact_locator(case.repo, path)
            if path == checkpoint:
                # Descriptor translation rejects its bound model bytes before
                # the per-arm terminal transactions start.
                with pytest.raises(ValueError, match="bound artifact SHA mismatch") as failed:
                    accept_run_workflow(*case.original, **options)
                assert str(failed.value) == expected_error, path
            else:
                # Allocation bindings belong to the terminal package. The
                # finalizer rejects them through its structured arm failures.
                failed = accept_run_workflow(*case.original, **options)
                assert failed["status"] == "INCOMPLETE", (path, failed)
                assert failed["executed"] is True and failed["closure_complete"] is False
                assert len(failed["results"]) == len(arms), (path, failed)
                assert all(result["pipeline_status"] == "FINALIZATION_FAILED" and
                           result["error"] == expected_error for result in failed["results"]), (path, failed)
            assert {arm_id: path.read_bytes() for arm_id, path in receipts.items()} == retained
        finally:
            path.write_bytes(retained_bytes)
    assert inspected == Counter({arm_id: 8 for arm_id in arms})
    assert len(validations) == 8
    assert {arm_id: path.read_bytes() for arm_id, path in receipts.items()} == retained
    validated = validate_repository_records(case.repo)
    assert validated["root"] == case.repo.resolve()

    # Exercise requalification after mechanical closure with real saved CPU
    # outputs and finalization. Native GPU admission and scientific comparison
    # qualification are explicit substitutes, never claims about this fixture.
    from copy import deepcopy
    from molgap import comparison_readiness as comparison_owner
    from molgap.research_memory.terminal_wiring import close_terminal_multi_arm
    from molgap.research_memory.trace import file_digest
    from test_comparison_readiness import _side
    resolved = resolve_run_acceptance_inputs(*case.original,
        retrieved_root=case.paths.artifact_root / "retrieved")
    native = {}
    for arm_id, item in resolved["outputs"].items():
        directory = Path(item["output_dir"])
        (directory / "runtime_certificate.json").write_text('{}', encoding="utf-8")
        (directory / "runtime_provenance.json").write_text(
            json.dumps({"runtime_fingerprint": "synthetic-runtime"}), encoding="utf-8")
        native[arm_id] = {"status": "VERIFIED", "dependencies": [{"path": "runtime_certificate.json",
            "local_path": str(directory / "runtime_certificate.json")}], "synthetic_substitute": True}
    assess = comparison_owner.assess_comparison_readiness
    def fixture_comparison(**kwargs):
        for side in ("candidate", "reference"):
            original_side = kwargs[side]
            fixture_side = _side(side)
            kwargs[side] = {**fixture_side, **{key: value for key, value in original_side.items()
                if key in {"artifact_bindings", "reference_bundle_id", "reference_bundle_sha256"}}}
        return assess(**kwargs)
    monkeypatch.setattr(comparison_owner, "assess_comparison_readiness", fixture_comparison)
    qualified = family.qualify_same_run_terminal_packages(case.repo, resolved,
        result["terminal_staging"]["locations"], native, reports=result["arms"])
    for arm_id, where in qualified["locations"].items():
        package = json.loads((case.repo / where["terminal"]).read_bytes())
        assert package["supersedes_finalization_ref"] == receipts[arm_id].relative_to(case.repo).as_posix()
        assert package["artifact_hashes"][package["supersedes_finalization_ref"]] == file_digest(receipts[arm_id])
    # Real custody can keep identical transforms under more than one question.
    transform = next((case.repo / "experiments").rglob("target_transform.json"))
    shared_copy = case.repo / "experiments/shared_transform/target_transform.json"
    shared_copy.parent.mkdir(parents=True)
    shared_copy.write_bytes(transform.read_bytes())
    closed = close_terminal_multi_arm(case.repo, [{key: where[key] for key in ("trajectory", "terminal", "trace")}
        for _, where in sorted(qualified["locations"].items(), key=lambda item: item[0] != reference_arm)])
    assert all(item["pipeline_status"] == "COMPLETE" for item in closed), closed
    before_reuse = {path.relative_to(case.repo).as_posix(): path.read_bytes()
        for path in (case.repo / "experiments").rglob("*") if path.is_file()}
    reused = family.qualify_same_run_terminal_packages(case.repo, resolved,
        result["terminal_staging"]["locations"], native, reports=result["arms"])
    assert reused["canonical_reused"] is True
    assert before_reuse == {path.relative_to(case.repo).as_posix(): path.read_bytes()
        for path in (case.repo / "experiments").rglob("*") if path.is_file()}
    changed = deepcopy(json.loads((case.repo / qualified["locations"][arm_id]["terminal"]).read_bytes()))
    changed["decision"]["next_allowed_actions"] = ["ALTERED"]
    changed_path = case.repo / "experiments/changed_science.json"
    changed_path.write_bytes(canonical_json(changed).encode())
    changed_locations = deepcopy(qualified["locations"])
    changed_locations[arm_id]["terminal"] = changed_path.relative_to(case.repo).as_posix()
    with pytest.raises(ValueError, match="changed accepted decision"):
        family.qualify_same_run_terminal_packages(case.repo, resolved, changed_locations, native, reports=result["arms"])
    assert all(path.read_bytes() == retained[arm] for arm, path in receipts.items())
    # A valid self-digest does not permit rebinding an existing transform ID.
    from molgap.comparison_readiness import target_transform_asset_digest
    conflicting = json.loads(shared_copy.read_bytes())
    conflicting["mean"] += 1
    conflicting["asset_sha256"] = target_transform_asset_digest(conflicting)
    shared_copy.write_bytes(canonical_json(conflicting).encode())
    with pytest.raises(ValueError, match="duplicate target_transform_asset_id"):
        validate_repository_records(case.repo)
