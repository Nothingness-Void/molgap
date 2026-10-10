"""Translate projection_check.close custody into reusable NO_TRAIN metadata."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import math
from pathlib import Path
import struct
import subprocess

from .evidence_pointers import load_json_object as read
from .research_memory.finalize import finalize
from .research_memory.paths import repo_local_path, verify_bound_artifact
from .research_memory.schemas import validate_trajectory
from .research_memory.trace import atomic_write
from .training_reproducibility import atomic_json, sha256_file


def _require(condition, reason):
    if not condition:
        raise ValueError(reason)


def _number(value, positive=False):
    return type(value) in (int, float) and math.isfinite(value) and (value > 0 if positive else value >= 0)


def close_cuda_execution_diagnostic(repo, attempt, *, evidence_id, trajectory_id,
                                    run_id, scope, legacy_contract, exact_controls,
                                    expected_cases, executed_rows=1920):
    """Verify retained JSON only; no model/data decoding or training release.

    Parent owns terminal_decision.md and attribution.md; source bytes must match
    both the working tree and recorded HEAD; existing snapshots are immutable.
    """
    root = Path(repo).resolve()
    folder = repo_local_path(root, attempt)
    folder.relative_to(root / "experiments")
    rel = folder.relative_to(root).as_posix()
    if (folder / "terminal.json").exists():
        package = read(folder / "terminal.json")
        _require((package["trajectory_id"], package["run_id"], package["evidence"]["evidence_id"],
                  package["evidence"]["scope"], package["evidence"]["legacy_contract"]) ==
                 (trajectory_id, run_id, evidence_id, scope, legacy_contract) and package["decision"]["outcome"] == "NO_TRAIN",
                 "Existing terminal closure identity mismatch")
        return finalize(root, f"{rel}/rml", f"{rel}/terminal.json")
    output = repo_local_path(root, folder / "results")
    inputs, result, runtime = (read(p) for p in (folder / "inputs.json", output / "result.json", output / "runtime.json"))
    trajectory = validate_trajectory(read(folder / "rml/trajectory.json"))
    completion, observation = read(output / "completion.json"), read(output / "cost_observation.json")
    inventory = {p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()} - {"completion.json"}
    required = {"result.json", "runtime.json", "cost_observation.json"}
    _require(completion["status"] == result["status"] == "complete" and required <= inventory <= required | {"contrasts.json"}
             and set(completion["artifacts"]) == inventory, "Exact JSON completion inventory required")
    for name, digest in completion["artifacts"].items():
        verify_bound_artifact(root, (output / name).relative_to(root).as_posix(), digest)
        read(output / name)
    _require(trajectory["record_mode"] == "prospective" and trajectory["decision"]["outcome"] == "ACTIVE"
             and trajectory["trajectory_id"] == trajectory_id, "Active prospective identity required")
    actions = [a for a in trajectory["actions"] if run_id in a["run_ids"]]
    _require(len(actions) == 1, "Unique prospective run/action required")
    action = actions[0]
    commit, inputs_hash = inputs["source_commit"], sha256_file(folder / "inputs.json")
    _require(commit == runtime["source_commit"] == action["source_commit"] == trajectory["state_at_start"]["source_commit"]
             and inputs_hash == runtime["inputs_sha256"] == trajectory["state_at_start"]["source_config_identity"]
             and runtime["prospective_sha256"] == sha256_file(folder / "rml/trajectory.json"), "Runtime/source/prospective identity mismatch")
    _require(runtime["scientific_training"] is False and result["contract_promotion"] is False, "No scientific training/promotion allowed")
    idle = runtime["idle_observations"]
    _require(isinstance(idle, list) and len(idle) >= 2 and all(o["idle"] is True for o in idle), "Two idle observations required")
    rows = inputs["source_idx"]
    _require(isinstance(rows, list) and all(type(i) is int and 0 <= i < 500000 for i in rows)
             and len(set(rows)) == len(rows) and type(executed_rows) is int and 0 < executed_rows <= len(rows)
             and runtime["train_decoded_source_idx"] == rows and runtime["train_executed_source_idx"] == rows[:executed_rows], "Runtime train row/count mismatch")
    expected, controls = set(expected_cases), set(exact_controls)
    _require(len(expected) == len(expected_cases) and len(controls) == len(exact_controls)
             and {"baseline", "non_blocking", "no_redundant_sync", "transfer_and_sync", "cpu_layout"} <= controls <= expected
             and set(result["cases"]) == expected and result["equivalence_updates"] == 3, "Case inventory/three updates mismatch")
    if expected == controls | {"fused_layout", "fused_tf32_layout"}:
        from .k1_precision_diagnostic import qualify_combined_report
        qualify_combined_report(result)
    for name, case in result["cases"].items():
        exact, changed = case["equivalence"]["exact"], case["optimizer_contract_change"]
        _require(type(exact) is bool and type(changed) is bool and case["status"] == ("qualified" if exact else "numerical_mismatch")
                 and len(case["timings"]) == 4 and all(_number(t["seconds"], True) and _number(t["seconds_per_step"], True) for t in case["timings"])
                 and _number(case["median_seconds_per_step"], True) and _number(case["equivalence"]["max_abs_error"]), "Invalid case/status/timing windows")
        _require(case["execution_only_admissible"] is (exact and not changed)
                 and (not changed or case["speed_candidate"] is False), "Optimizer contract changes are never admissible")
        _require(name not in controls or (exact and not changed), "Exact control mismatch")
    wall, cpu, limit = observation["worker_wall_seconds"], observation["process_cpu_seconds"], inputs["worker_wall_limit_seconds"]
    _require(_number(limit, True) and limit <= 600 and _number(wall, True) and wall <= limit
             and _number(cpu) and observation["gpu_busy_seconds"] is None, "Invalid worker/CPU bound or fabricated GPU busy time")
    _require(all((folder / n).is_file() and (folder / n).stat().st_size for n in ("README.md", "terminal_decision.md", "attribution.md")), "Parent terminal decision/attribution required")
    _require({"train_probe", "initial", "normalization_checkpoint"} <= inputs["bindings"].keys(), "Input bindings required")
    for entry in inputs["bindings"].values():
        path = Path(entry["path"])
        _require(sha256_file(path if path.is_absolute() else repo_local_path(root, path)) == entry["sha256"], "Input binding SHA mismatch")
    _require(subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip() == commit, "Recorded source commit differs from HEAD")
    _require(bool(inputs["files"]), "Executed source inventory required")
    frozen = {}
    for name, digest in inputs["files"].items():
        source = repo_local_path(root, name)
        _require(source.relative_to(root).as_posix() == name, "Canonical source path required")
        verify_bound_artifact(root, name, digest)
        payload = subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=root)
        _require(hashlib.sha256(payload).hexdigest() == digest, "Git recorded source bytes mismatch")
        destination = repo_local_path(root, folder / "executed_source" / name)
        _require(not destination.exists() or sha256_file(destination) == digest, "Immutable source snapshot mismatch")
        frozen[destination] = payload
    cost = deepcopy(read(folder / "plan_input.json")["costs"][0])
    _require((cost["trajectory_id"], cost["run_id"], cost["action_id"]) == (trajectory_id, run_id, action["action_id"])
             and cost["attempt_id"] in action["attempt_ids"], "Cost attempt identity mismatch")
    for path, payload in frozen.items():
        if not path.exists():
            atomic_write(path, payload)
    decision = dict(outcome="NO_TRAIN", decision_ref=f"{rel}/terminal_decision.md", next_allowed_actions=[],
                    reopen_conditions=["Separately authorized numerical/native/quality qualification"])
    outcome = dict(execution_status="complete_cuda_execution_diagnostic", artifact_status="local_hash_verified",
                   comparison_status="descriptive_matched_cuda_scratch_only", scientific_status="NO_TRAIN",
                   transfer_status="not_evaluated", budget_decision="bounded_local_diagnostic_complete", full_handoff_status="not_applicable")
    cost.update(cost_event_id=f"cost-{evidence_id}-observed", category="other", evidence_ref=f"{rel}/acceptance.json",
                measurement=dict(wall_hours=dict(value=wall/3600, status="measured"), cpu_hours=dict(value=cpu/3600, status="measured"),
                                 device_hours=dict(value=None, status="measurement_missing"), queue_hours=dict(value=None, status="not_applicable")))
    roles = []
    for role, members, accesses in (("train_decoded", runtime["train_decoded_source_idx"], ("labels_read",)),
                                   ("train_executed", runtime["train_executed_source_idx"], ("prediction_input", "metric_computed"))):
        digest = hashlib.sha256(b"".join(struct.pack("<q", i) for i in members)).hexdigest()
        for access in accesses:
            roles.append(dict(schema="molgap-role-event-v1", role_event_id=f"role-{evidence_id}-{role}-{access}", trajectory_id=trajectory_id,
                              action_id=action["action_id"], run_id=run_id, dataset_identity=inputs.get("dataset_identity", "pcqm4mv2-ogb-fixed-500k-scnet-v1"),
                              row_manifest_hash=digest, role_name=role, access_kind=access, selection_used=False, evidence_ref=f"{rel}/acceptance.json"))
    role_use = dict(train_decoded="labels_read", train_executed="prediction_input", development="untouched",
                    official_validation="untouched", test_dev="untouched", test_challenge="untouched")
    atomic_json(folder / "acceptance.json", dict(evidence_id=evidence_id, run_id=run_id, outcome=outcome, trajectory_decision=decision,
                role_use=role_use, costs=[cost], roles=roles, scientific_training=False, training_replay_ready=False, generalization_accuracy_qualified=False))
    hashes = {p.relative_to(root).as_posix(): sha256_file(repo_local_path(root, p)) for p in folder.rglob("*") if p.is_file()
              and not {"rml", "rml_finalized", "__pycache__"}.intersection(p.relative_to(folder).parts)
              and p.name not in {"terminal.json", "closure_receipt.json"}}
    now = datetime.now(timezone.utc)
    evidence = dict(format="molgap-v5-evidence-envelope-v1", contract="MOLGAP-COMMON-V5-FINAL", evidence_id=evidence_id,
        track=trajectory["track"], scope=scope, legacy_contract=legacy_contract, outcome=outcome, role_use=role_use,
        authority=dict(pointers=[f"{rel}/{n}" for n in ("README.md", "terminal_decision.md", "attribution.md", "acceptance.json")]),
        migration=dict(migrated_at=now.date().isoformat(), training_executed=False, inference_executed=False, scientific_reinterpretation=False,
                       verification_scope="Metadata closure of separately executed prospective CUDA scratch diagnostic"),
        observed_execution=dict(training_executed=False, inference_executed=True, training_replay_ready=False, execution_ref=f"{rel}/results/result.json"),
        artifacts=[dict(name=p.removeprefix(rel+"/"), locator=p, sha256=h, availability="locally_retained_hash_verified") for p, h in sorted(hashes.items())])
    atomic_json(folder / "terminal.json", dict(format="molgap-rml-terminal-package-v1", trajectory_id=trajectory_id, run_id=run_id,
        action_id=action["action_id"], finalized_at=now.isoformat(), acceptance_ref=f"{rel}/acceptance.json", artifact_hashes=hashes,
        evidence=evidence, decision=decision, costs=[cost], roles=roles))
    receipt = finalize(root, f"{rel}/rml", f"{rel}/terminal.json")
    atomic_json(folder / "closure_receipt.json", receipt)
    return receipt
