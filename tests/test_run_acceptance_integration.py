"""CPU-only joined run-ID acceptance path over a frozen synthetic package."""
from __future__ import annotations

import json

from molgap.experiment_attribution import ATTRIBUTION_FORMAT
from molgap.experiment_family_workflow import RunContext
from molgap.experiment_launch import canonical_json
from molgap import experiment_cli
from molgap.experiment_run import accept_run_workflow
from molgap.experiment_run_acceptance import resolve_run_acceptance_inputs
from molgap.research_memory.paths import artifact_locator
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


def test_original_frozen_run_reaches_diagnosis_and_terminal_staging_cpu_only(tmp_path, repo, capsys):
    case = frozen_shared_acceptance_run(tmp_path, repo)
    declaration = case.spec.to_dict()
    # The frozen workflow pointer lives in the original run; use it to locate
    # every packaged expectation, adapter, and recipe without mutable sources.
    workflow = json.loads((case.run / "inputs/workflow_plan.json").read_bytes())
    plan = json.loads(case.generated[workflow["acceptance_plan"]])
    entries = {item["arm_id"]: item for item in plan["arms"]}
    arms = {item["arm_id"]: item for item in declaration["arms"]}

    locations = {}
    for arm_id, arm in arms.items():
        entry = entries[arm_id]
        binding = next(b for b in declaration["prospective"]["arms"] if b["arm_id"] == arm_id)
        context, output_dir = _write_shared_output_for_arm(case, arm, binding, entry)
        manifest = json.loads((output_dir / "output_manifest.json").read_bytes())
        trace_path = output_dir / manifest["artifacts"]["trace"]["path"]
        locations[arm_id] = {
            "trajectory_id": binding["trajectory_id"],
            "run_id": declaration["logical_run_id"] + ":" + arm_id + ":downstream",
            "trajectory": binding["output"] + "/trajectory.json",
            "terminal": binding["output"] + "/terminal.json",
            "trace": artifact_locator(case.repo, trace_path),
        }

    explicit = resolve_run_acceptance_inputs(*case.original, retrieved_root=case.paths.artifact_root / "retrieved",
        locations=locations)
    assert {arm: item["trace"] for arm, item in explicit["locations"].items()} == {
        arm: item["trace"] for arm, item in locations.items()}

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
