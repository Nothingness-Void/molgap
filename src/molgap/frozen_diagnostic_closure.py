"""Translate completed local frozen diagnostics into the existing RML finalizer.

The caller owns interpretation. This adapter neither executes a model nor adds
a scientific gate; it binds observed rows, native CPU timers and result bytes.
"""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import struct

from .research_memory.finalize import finalize
from .training_reproducibility import atomic_json, sha256_file


def close_local_diagnostic(repo: Path, experiment: Path, *, evidence_id: str,
                           policy_id: str, role_rows: dict):
    repo, experiment = repo.resolve(), experiment.resolve()
    rel = experiment.relative_to(repo).as_posix()
    read = lambda p: json.loads(p.read_text(encoding="utf-8"))
    inputs = read(experiment / "inputs.json")
    result = read(experiment / "results/result.json")
    completion = read(experiment / "results/completion.json")
    trajectory = read(experiment / "rml/trajectory.json")
    if completion.get("complete") is not True or result.get("status") != "complete":
        raise ValueError("Diagnostic is incomplete")
    if not result.get("checks") or not all(v is True for v in result["checks"].values()):
        raise ValueError("Diagnostic checks failed")
    runtime = result.get("runtime", {})
    if runtime.get("device") != "cpu" or runtime.get("cpu_threads") != inputs.get("cpu_threads") or runtime.get("cpu_threads") != 4:
        raise ValueError("This adapter owns local CPU diagnostics only")
    if result.get("optimizer_created") is not False or result.get("gradients_computed") is not False or result.get("training_executed") is not False:
        raise ValueError("Unexpected training or gradients")
    for key, path in [("inputs_sha256", experiment / "inputs.json"),
                      ("prospective_sha256", experiment / "rml/trajectory.json")]:
        if result.get(key) != sha256_file(path):
            raise ValueError(f"Result identity differs: {key}")
    if completion.get("inputs_sha256") != result["inputs_sha256"]:
        raise ValueError("Completion input identity differs")
    train, dev = inputs["train_source_idx"], inputs["development_source_idx"]
    if result["format"] == "molgap-k1-endpoint-average-diagnostic-v1":
        expected_roles = {"train_features": {"labels_read": train, "prediction_input": train},
                          "internal_development": {a:dev for a in ("labels_read", "prediction_input", "metric_computed", "selection_used")}}
    elif result["format"] == "molgap-k1-clean-fit-diagnostic-v1":
        if inputs["train_decoded_source_bounds"] != [0, 500000]:
            raise ValueError("Clean-fit decoded training bounds differ")
        expected_roles = {"train_decoded": {"labels_read": list(range(500000))},
                          "train_descriptive": {"prediction_input": train, "metric_computed": train},
                          "internal_development": {a:dev for a in ("labels_read", "prediction_input", "metric_computed", "selection_used")}}
    elif result["format"] == "molgap-k1-component-diagnostic-v1":
        sample = result["sample_source_idx"]
        sample_dev, sample_train = [i for i in sample if i >= 500000], [i for i in sample if i < 500000]
        if not set(sample_dev) <= set(dev) or not set(sample_train) <= set(train):
            raise ValueError("Observed sample escaped its declared roles")
        expected_roles = {"train_decoded": {"labels_read": train},
                          "train_descriptive": {"prediction_input": sample_train, "metric_computed": sample_train},
                          "internal_development_decoded": {"labels_read": dev},
                          "internal_development_retained_predictions": {a:dev for a in ("labels_read", "metric_computed")},
                          "internal_development": {a:sample_dev for a in ("prediction_input", "metric_computed", "selection_used")}}
    else:
        raise ValueError("Unsupported diagnostic result format")
    if role_rows != expected_roles:
        raise ValueError("Caller role events differ from observed diagnostic membership")
    actual = {p.name for p in (experiment / "results").iterdir() if p.is_file() and p.name != "completion.json" and p.suffix in {".json", ".pt"}}
    if actual != set(completion["files"]):
        raise ValueError("Completion inventory differs")
    for name, digest in completion["files"].items():
        path = (experiment / "results" / name).resolve()
        if not path.is_relative_to(experiment / "results") or sha256_file(path) != digest:
            raise ValueError(f"Result bytes differ: {name}")
    for name, digest in inputs["executed_source_files"].items():
        if sha256_file(repo / name) != digest:
            raise ValueError(f"Executed source differs: {name}")
    wall, cpu = float(result["wall_seconds"]), float(result["process_cpu_seconds"])
    if not math.isfinite(wall) or not math.isfinite(cpu) or wall < 0 or cpu < 0 or wall > inputs["ceiling_seconds"]:
        raise ValueError("Invalid native cost or exceeded ceiling")
    for name in ("terminal_decision.md", "attribution.md"):
        if not (experiment / name).is_file():
            raise ValueError(f"Parent-written interpretation required: {name}")
    tid = trajectory["trajectory_id"]
    action = trajectory["actions"][0]
    run = action["run_ids"][0]
    short = evidence_id.removeprefix("pcqm-")
    outcome = dict(execution_status="complete_no_training", artifact_status="local_hash_verified",
                   comparison_status="consumed_role_frozen_diagnostic", scientific_status="NO_TRAIN",
                   transfer_status="not_evaluated", budget_decision="bounded_local_diagnostic_complete",
                   full_handoff_status="not_applicable")
    decision = dict(outcome="NO_TRAIN", decision_ref=f"{rel}/terminal_decision.md", next_allowed_actions=[],
                    reopen_conditions=["Separately prospective and authorized training or independent-role validation"])
    cost = dict(schema="molgap-cost-event-v1", cost_event_id=f"cost-{short}-observed", trajectory_id=tid,
                action_id=action["action_id"], run_id=run, attempt_id="attempt-001", platform="local-windows",
                hardware="CPU four intra-op threads; no accelerator", category="inference", evidence_ref=f"{rel}/acceptance.json",
                measurement={"wall_hours": {"value": wall / 3600, "status": "measured"},
                             "cpu_hours": {"value": cpu / 3600, "status": "measured"},
                             "device_hours": {"value": None, "status": "not_applicable"},
                             "queue_hours": {"value": None, "status": "not_applicable"}})
    roles = []
    costs = [cost]
    if result["format"] == "molgap-k1-component-diagnostic-v1":
        summary = read(experiment / "retained_prediction_summary.json")
        if summary.get("inference_executed") is not False or summary.get("training_executed") is not False or summary.get("rows") != len(dev):
            raise ValueError("Retained-prediction analysis scope differs")
        for name, digest in summary["source_hashes"].items():
            if digest != inputs["artifacts"][name]["sha256"]:
                raise ValueError("Retained-prediction analysis input differs")
        summary_cost = dict(cost, cost_event_id=f"cost-{short}-saved-analysis", category="other")
        summary_cost["measurement"] = dict(cost["measurement"],
            wall_hours={"value":summary["wall_seconds"]/3600,"status":"measured"},
            cpu_hours={"value":summary["process_cpu_seconds"]/3600,"status":"measured"})
        costs.append(summary_cost)
    role_use = dict.fromkeys(("official_validation", "test_dev", "test_challenge", "common", "ood"), "untouched")
    for role, observations in role_rows.items():
        role_use[role] = "consumed"
        for access, rows in observations.items():
            rows = [int(i) for i in rows]
            if not rows or len(set(rows)) != len(rows):
                raise ValueError("Invalid observed role membership")
            digest = hashlib.sha256(b"".join(struct.pack("<q", i) for i in rows)).hexdigest()
            roles.append(dict(schema="molgap-role-event-v1", role_event_id=f"role-{short}-{role}-{access}",
                              trajectory_id=tid, action_id=action["action_id"], run_id=run,
                              dataset_identity="pcqm4mv2-ogb-fixed-500k-scnet-v1", row_manifest_hash=digest,
                              role_name=role, access_kind=access, selection_used=role.startswith("internal_development"),
                              evidence_ref=f"{rel}/acceptance.json"))
    acceptance = dict(format="molgap-local-frozen-diagnostic-acceptance-v1", evidence_id=evidence_id,
                      run_id=run, outcome=outcome, trajectory_decision=decision, role_use=role_use,
                      costs=costs, roles=roles, checks=result["checks"], comparison_class="CONTEXT_ONLY",
                      analysis_ref=f"{rel}/results/result.json", cost_scope="Worker hashing/loading/calibration/inference/analysis; excludes preparation/tests/Git and historical comparator execution",
                      execution_scope="Frozen CPU inference, no optimizer, training trace or training replay-ready claim")
    atomic_json(experiment / "acceptance.json", acceptance)
    paths = [p for p in experiment.rglob("*") if p.is_file() and "rml" not in p.relative_to(experiment).parts
             and "__pycache__" not in p.parts and p.name != "terminal.json"]
    paths += [repo / n for n in inputs["executed_source_files"]]
    paths += [repo / "src/molgap/frozen_diagnostic_closure.py", repo / f"research_memory/policies/{policy_id}.1.json"]
    hashes = {p.relative_to(repo).as_posix(): sha256_file(p) for p in paths}
    evidence = dict(format="molgap-v5-evidence-envelope-v1", contract="MOLGAP-COMMON-V5-FINAL", evidence_id=evidence_id,
                    track="B", scope=policy_id, legacy_contract=policy_id + "-v1", outcome=outcome,
                    authority={"pointers": [f"{rel}/{n}" for n in ("protocol.md", "terminal_decision.md", "attribution.md", "acceptance.json", "results/result.json")]},
                    role_use=role_use,
                    migration=dict(migrated_at=datetime.now(timezone.utc).date().isoformat(), training_executed=False,
                                   inference_executed=False, scientific_reinterpretation=False, verification_scope="Metadata finalization of separately executed local diagnostic; no model execution in closure"),
                    observed_execution=dict(training_executed=False, inference_executed=True, training_replay_ready=False,
                                            execution_ref=f"{rel}/results/result.json"),
                    artifacts=[dict(name=n.removeprefix(rel + "/"), locator=n, sha256=h, availability="locally_retained_hash_verified")
                               for n, h in hashes.items() if n.startswith(rel + "/")])
    terminal = dict(format="molgap-rml-terminal-package-v1", trajectory_id=tid, run_id=run, action_id=action["action_id"],
                    finalized_at=datetime.now(timezone.utc).isoformat(), acceptance_ref=f"{rel}/acceptance.json",
                    artifact_hashes=hashes, evidence=evidence, decision=decision, costs=costs, roles=roles)
    atomic_json(experiment / "terminal.json", terminal)
    return finalize(repo, f"{rel}/rml", f"{rel}/terminal.json")
