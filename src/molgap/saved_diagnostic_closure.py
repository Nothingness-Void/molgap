"""Translate saved arithmetic to RML; retain frozen source, not mutable code refs."""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess

from .research_memory.finalize import finalize
from .research_memory.paths import repo_local_path
from .training_reproducibility import atomic_json, sha256_file


def close_saved_diagnostic(repo: Path, experiment: Path):
    repo, experiment = repo.resolve(), experiment.resolve()
    rel = experiment.relative_to(repo).as_posix()
    if (experiment/"rml/rml_finalized/finalization.json").is_file():
        return finalize(repo,f"{rel}/rml",f"{rel}/terminal.json")
    read = lambda name: json.loads((experiment/name).read_text(encoding="utf-8"))
    inputs, trajectory = read("inputs.json"), read("rml/trajectory.json")
    for name,digest in inputs["snapshot_files"].items():
        path = repo_local_path(experiment/"source_snapshot",name)
        if sha256_file(path) != digest:
            raise ValueError("Frozen metadata hash differs")
    failed = (experiment/"failure.json").is_file()
    if failed and (experiment/"results/result.json").exists():
        raise ValueError("Failure and completed result cannot coexist")
    result = read("failure.json" if failed else "results/result.json")
    if result["inputs_sha256"] != sha256_file(experiment/"inputs.json") or result["prospective_sha256"] != sha256_file(experiment/"rml/trajectory.json"):
        raise ValueError("Prospective/input identity differs")
    if any(result.get(n) is not False for n in ("training_executed","inference_executed","optimizer_created","gradients_computed")):
        raise ValueError("Saved arithmetic cannot perform model work")
    if not failed:
        completion = read("results/completion.json")
        if result.get("format") != "molgap-k1-saved-analysis-v1" or result.get("status") != "complete" or not result.get("checks") or not all(v is True for v in result["checks"].values()):
            raise ValueError("Saved analysis not accepted")
        if completion.get("complete") is not True or completion.get("inputs_sha256") != result["inputs_sha256"]:
            raise ValueError("Completion differs")
        files = {p.name for p in (experiment/"results").iterdir() if p.is_file() and p.name != "completion.json"}
        if files != set(completion["files"]):
            raise ValueError("Output inventory differs")
        for name,digest in completion["files"].items():
            path = (experiment/"results"/name).resolve()
            if not path.is_relative_to(experiment/"results") or sha256_file(path) != digest:
                raise ValueError("Output hash differs")
        wall, cpu = result["wall_seconds"], result["process_cpu_seconds"]
        if not all(math.isfinite(x) and x >= 0 for x in (wall,cpu)) or wall > inputs["ceiling_seconds"]:
            raise ValueError("Invalid CPU cost")
        if result.get("kind") != inputs["kind"] or result["runtime"]["device"] != "cpu" or result["runtime"]["cpu_threads"] != 4:
            raise ValueError("Runtime/kind differs")
    elif result.get("status") != "infrastructure_failed_before_metric_analysis" or inputs["kind"] != "trace_pair":
        raise ValueError("Unsupported failure closure")
    for name in ("terminal_decision.md","attribution.md"):
        if not (experiment/name).is_file():
            raise ValueError("Parent interpretation missing")
    # The executed commit is immutable; snapshots keep old receipts valid after reuse evolves.
    commit = trajectory["actions"][0]["source_commit"]
    for name,digest in inputs["executed_source_files"].items():
        data = subprocess.check_output(["git","show",f"{commit}:{name}"],cwd=repo)
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError("Executed source is not retained by its recorded commit")
        path = repo_local_path(experiment/"executed_source",name)
        path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists() and path.read_bytes() != data:
            raise ValueError("Frozen executed source changed")
        if not path.exists():
            path.write_bytes(data)
    tid, action = trajectory["trajectory_id"], trajectory["actions"][0]
    run, eid = action["run_ids"][0], tid.removeprefix("TB-")+"-evidence"
    decision = dict(outcome="NO_TRAIN",decision_ref=f"{rel}/terminal_decision.md",next_allowed_actions=[],
        reopen_conditions=["Separately prospective authorized diagnostic; original training gate remains independent"])
    outcome = dict(execution_status=result["status"],artifact_status="local_hash_verified",comparison_status="descriptive_saved_evidence_only",
        scientific_status="NO_TRAIN",transfer_status="not_evaluated",budget_decision="local_analysis_complete" if not failed else "infrastructure_attempt_closed",
        full_handoff_status="not_applicable")
    cost = dict(schema="molgap-cost-event-v1",cost_event_id="cost-"+tid+"-observed",trajectory_id=tid,action_id=action["action_id"],run_id=run,
        attempt_id="attempt-002" if rel.endswith("attempt_002") else "attempt-001",platform="local-windows",hardware="CPU saved-artifact arithmetic; no accelerator",
        category="other",evidence_ref=f"{rel}/acceptance.json",measurement=dict(
            wall_hours=dict(value=None if failed else result["wall_seconds"]/3600,status="measurement_missing" if failed else "measured"),
            cpu_hours=dict(value=None if failed else result["process_cpu_seconds"]/3600,status="measurement_missing" if failed else "measured"),
            device_hours=dict(value=None,status="not_applicable"),queue_hours=dict(value=None,status="not_applicable")))
    expected = list(range(*inputs["development_bounds"])) if inputs["kind"] == "bn_rows" else []
    if result.get("observed_source_idx") != expected:
        raise ValueError("Observed membership differs")
    roles = []
    role_use = dict.fromkeys(("official_validation","test_dev","test_challenge","common","ood"),"untouched")
    if expected:
        role_use["internal_development"] = "consumed"
        digest = hashlib.sha256(b"".join(struct.pack("<q",i) for i in expected)).hexdigest()
        for access in ("labels_read","metric_computed"):
            roles.append(dict(schema="molgap-role-event-v1",role_event_id=f"role-{tid}-{access}",trajectory_id=tid,action_id=action["action_id"],run_id=run,
                dataset_identity="pcqm4mv2-ogb-fixed-500k-scnet-v1",row_manifest_hash=digest,role_name="internal_development",access_kind=access,
                selection_used=False,evidence_ref=f"{rel}/acceptance.json"))
    acceptance = dict(evidence_id=eid,run_id=run,outcome=outcome,trajectory_decision=decision,role_use=role_use,costs=[cost],roles=roles,
        checks=result.get("checks",{}),analysis_ref=f"{rel}/"+("failure.json" if failed else "results/result.json"),
        execution_scope="Saved metadata/tensor arithmetic only; no model, new selection, training or replay-ready claim")
    atomic_json(experiment/"acceptance.json",acceptance)
    paths = [p for p in experiment.rglob("*") if p.is_file() and "rml" not in p.relative_to(experiment).parts
             and "rml_finalized" not in p.parts and "__pycache__" not in p.parts and p.name != "terminal.json"]
    # A parent attempt never hashes its independently planned child attempt.
    paths = [p for p in paths if not p.is_relative_to(experiment/"attempt_002")]
    hashes = {p.relative_to(repo).as_posix():sha256_file(p) for p in paths}
    evidence = dict(format="molgap-v5-evidence-envelope-v1",contract="MOLGAP-COMMON-V5-FINAL",evidence_id=eid,track="B",scope=inputs["kind"],
        legacy_contract="pcqm-k1-saved-"+inputs["kind"].replace("_","-")+"-v1",
        migration=dict(migrated_at=datetime.now(timezone.utc).date().isoformat(),training_executed=False,
            inference_executed=False,scientific_reinterpretation=False,
            verification_scope="Metadata translation of separately recorded saved arithmetic or infrastructure failure; no model work in closure"),
        outcome=outcome,authority=dict(pointers=[f"{rel}/{n}" for n in ("protocol.md","terminal_decision.md","attribution.md","acceptance.json")]),
        role_use=role_use,observed_execution=dict(training_executed=False,inference_executed=False,training_replay_ready=False),
        artifacts=[dict(name=n.removeprefix(rel+"/"),locator=n,sha256=h,availability="locally_retained_hash_verified") for n,h in hashes.items()])
    terminal = dict(format="molgap-rml-terminal-package-v1",trajectory_id=tid,run_id=run,action_id=action["action_id"],
        finalized_at=datetime.now(timezone.utc).isoformat(),acceptance_ref=f"{rel}/acceptance.json",artifact_hashes=hashes,evidence=evidence,
        decision=decision,costs=[cost],roles=roles)
    atomic_json(experiment/"terminal.json",terminal)
    return finalize(repo,f"{rel}/rml",f"{rel}/terminal.json")
