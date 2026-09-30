"""Real local planning/staging/receipt/closure with synthetic saved state events.

No trainer, platform or model is executed. Incomplete scientific qualification
must remain replay-blocked even after the complete mechanical path succeeds.
"""
import copy
import json
import subprocess
from pathlib import Path

import random
import numpy as np
import torch

from molgap import experiment_launch as launch
from molgap.experiment_spec import ExperimentSpec, SCHEMA_VERSION_V2
from molgap.experiment_workflow import prepare_experiment_release
from molgap.experiment_family_workflow import RunContext, build_verified_terminal_descriptor
from molgap.experiment_training_hooks import bind_training_outputs
from molgap.research_memory.trace import json_bytes, file_digest
from molgap.screen_policy import canonical_fingerprint
from test_experiment_staging import staging, payload
from test_experiment_family_workflow import _recipe, SOURCE_IDX, TARGET, known, unknown
from test_terminal_trace_closure import setup_mock_repo, create_candidate_arm


def test_prospective_to_terminal_is_one_bound_flow_and_fails_closed_on_replay(staging, capsys):
    root = staging["repo_root"]
    setup_mock_repo(root)
    declaration = staging["spec"].to_dict()
    declaration["schema_version"] = SCHEMA_VERSION_V2
    recipe = _recipe("gptrans_t")
    recipe["trainer_binding"] = {"name": "pcqm_gptrans_v4", "variant": "reference"}
    (root / "recipe.json").write_bytes(json_bytes(recipe))
    declaration["arms"][0]["training"]["recipe"]["sha256"] = file_digest(root / "recipe.json")
    trajectory_id, run_id = "T-lifecycle", declaration["logical_run_id"] + ":gptrans_t:downstream"
    candidate = create_candidate_arm(root, "authority", trajectory_id, run_id, "ev-lifecycle")
    trajectory = json.loads(candidate["traj_path"].read_bytes())
    # Fixture authority is separate from the fresh prospective destination.
    candidate["traj_path"].unlink()
    trajectory["state_at_start"]["source_config_identity"] = canonical_fingerprint(declaration["arms"][0])
    trajectory["hypothesis"]["expected_native_cost_ref"] = "C-lifecycle"
    trajectory["actions"][0].update(cost_event_ids=["C-lifecycle"], evidence_refs=[])
    policy_dir = root / "research_memory/policies"
    policy_dir.mkdir()
    policy_source = Path(__file__).resolve().parents[1] / "research_memory/policies/k1-relation-resolution-bounded-research.v1.json"
    policy = json.loads(policy_source.read_bytes())
    (policy_dir / "synthetic.json").write_bytes(json_bytes(policy))
    planned_cost = {
        "schema": "molgap-cost-event-v1", "cost_event_id": "C-lifecycle",
        "action_id": "act-1", "run_id": run_id, "attempt_id": "att-1",
        "platform": "local", "hardware": "synthetic-cpu", "category": "training",
        "evidence_ref": "experiments/authority/contract.json",
        "measurement": {k: {"value": None, "status": "measurement_missing"}
                        for k in ("device_hours", "cpu_hours", "wall_hours", "queue_hours")},
    }
    plan_input = {"trajectory": trajectory, "costs": [planned_cost], "decision_state": {
        "available_actions": ["act-1"], "chosen_action": "act-1", "policy_id": policy["policy_id"],
        "policy_version": policy["version"], "state_timestamp": "2026-09-30T00:00:00Z"}}
    (root / "plan-input.json").write_bytes(json_bytes(plan_input))
    declaration["prospective"] = {"arms": [{"arm_id": "gptrans_t", "trajectory_id": trajectory_id,
        "plan_spec_ref": "plan-input.json", "plan_spec_sha256": file_digest(root / "plan-input.json"),
        "output": "experiments/planned-gptrans"}]}
    spec = ExperimentSpec(declaration)
    subprocess.run(["git", "add", "."], cwd=root, check=True, capture_output=True)
    subprocess.run(["git", "-c", "commit.gpgsign=false", "commit", "-m", "Synthetic lifecycle inputs"],
                   cwd=root, check=True, capture_output=True)
    source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    trajectory["actions"][0]["source_commit"] = source_commit
    (root / "plan-input.json").write_bytes(json_bytes(plan_input))
    declaration["prospective"]["arms"][0]["plan_spec_sha256"] = file_digest(root / "plan-input.json")
    spec = ExperimentSpec(declaration)
    config = {
        "format": "molgap-release-workflow-v1", "spec_identity": spec.identity,
        "source_paths": staging["relative_paths"],
        "artifacts": {k: {"path": str(v.path), "sha256": v.sha256} for k, v in staging["artifacts"].items()},
        "recipe_files": staging["recipe_files"], "initial_states": staging["initial_states"],
        "required_modules": staging["required_modules"], "entry_template": str(staging["entry_template"]),
        "kernel_metadata": str(staging["kernel_metadata"]), "dataset_metadata": None, "pickle_inputs": []}
    prepared, code = prepare_experiment_release(spec, root, json.dumps(config), root / "local-release")
    assert code == 0, prepared
    assert prepared["compute_released"] is prepared["submitted"] is False
    package_dir = root / "local-release/release/source"
    package_identity = prepared["staging"]["package"]["package_identity"]
    reference = "synthetic-account/lifecycle-job"
    binding = launch.build_launch_receipt(spec, package_dir, expected_package_identity=package_identity)["binding"]
    response = {"format": launch.RESPONSE_FORMAT, "version": launch.VERSION,
        "mode": "observed", "outcome": "accepted", "conflict_kind": None, "binding": binding,
        "canonical_platform_reference": known(reference), "platform_version": known("1"),
        "physical_runs": known([{"run_identity": "synthetic-job", "canonical_reference": reference,
            "platform_version": known("1"), "arm_ids": ["gptrans_t"]}]),
        "timestamp": unknown(), "monitor_paths": unknown()}
    receipt = launch.reconcile_platform_response(spec, package_dir, launch.canonical_json(response),
                                                 expected_package_identity=package_identity)
    receipt_dir = root / "receipts"
    receipt_dir.mkdir()
    receipt_path = launch.write_launch_receipt(launch.canonical_json(receipt), receipt_dir, spec,
        package_dir, expected_package_identity=package_identity)
    output = root / "retained/family_outputs"
    hooks = bind_training_outputs(spec, package_dir=package_dir, expected_package_identity=package_identity,
        arm_id="gptrans_t", account="synthetic-account", run_reference=reference,
        output=output, native_output=output.parent, contract=root / "recipe.json")
    native = {"model": {"w": torch.tensor([0.5])}, "ema": {"w": torch.tensor([0.5])},
        "optimizer": {"state": {}, "param_groups": [{"params": [0]}]},
        "scheduler": {"last_epoch": 2}, "rng_state": {"python": random.getstate(),
            "numpy": np.random.get_state(), "torch": torch.get_rng_state(), "cuda": []}}
    native["family_output_binding"] = hooks.binding
    for epoch in range(2):
        hooks.completed_epoch(epoch=epoch, optimizer_step=(epoch + 1) * 2,
            sample_presentations=(epoch + 1) * 4, train_mae_eV=1.1,
            live_dev_mae_eV=1.0, ema_dev_mae_eV=1.0, checkpoint=native,
            sampler_order_sha256="a" * 64, selected={"model_state": native["ema"], "weights": "ema",
                "prediction_eV": torch.ones(3, dtype=torch.float64), "target_eV": TARGET, "source_idx": SOURCE_IDX})
    assert hooks.complete(hardware="synthetic-cpu")["status"] == "MECHANICALLY_VERIFIED"
    context = RunContext.from_launch(spec, receipt_path, package_dir,
                                     expected_package_identity=package_identity, arm_id="gptrans_t")
    terminal = json.loads(candidate["terminal_path"].read_bytes())
    trace_ref = "retained/family_outputs/canonical_trace.json"
    artifact = next(a for a in terminal["evidence"]["artifacts"] if a["name"] == "training_trace")
    terminal["artifact_hashes"].pop(artifact["locator"])
    artifact.update(locator=trace_ref, sha256=file_digest(output / "canonical_trace.json"))
    terminal["artifact_hashes"][trace_ref] = artifact["sha256"]
    candidate["terminal_path"].write_bytes(json_bytes(terminal))
    outputs = {"gptrans_t": {"context": context, "output_dir": output,
                              "expected": copy.deepcopy(recipe["acceptance_requirements"])}}
    locations = {"gptrans_t": {"trajectory_id": trajectory_id, "run_id": run_id,
        "trajectory": "experiments/planned-gptrans/trajectory.json",
        "terminal": "experiments/authority/terminal.json", "trace": trace_ref}}
    descriptor = build_verified_terminal_descriptor(root, spec, outputs=outputs, locations=locations)
    from molgap.experiment_cli import main
    spec_path, descriptor_path, outputs_path = (root / name for name in
        ("closure-spec.json", "closure-descriptor.json", "closure-outputs.json"))
    spec_path.write_bytes(spec.to_json().encode("utf-8"))
    descriptor_path.write_bytes(descriptor.to_json().encode("utf-8"))
    outputs_path.write_bytes(json_bytes({"gptrans_t": {
        "output_dir": "retained/family_outputs", "expected": recipe["acceptance_requirements"]}}))
    capsys.readouterr()
    code = main(["accept-terminal", "--spec", str(spec_path), "--repo-root", str(root),
        "--package", str(package_dir), "--expected-package-identity", package_identity,
        "--receipt", str(receipt_path), "--descriptor", str(descriptor_path),
        "--outputs", str(outputs_path), "--execute"])
    result = json.loads(capsys.readouterr().out)
    assert code == 0, result
    assert result["status"] == "COMPLETE", result
    assert result["replay_readiness"] == "BLOCKED", result
    assert (root / "experiments/planned-gptrans/rml_finalized/prospective_snapshot.json").is_file()
