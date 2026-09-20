"""Freeze supplied scientific content and the knowledge available at planning."""

from __future__ import annotations

import copy
import hashlib
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from molgap.evidence_pointers import load_json_object
from .paths import repo_local_path, resolve_repo_pointer
from .discovery import discover_records
from .schemas import validate_cost_event, validate_trajectory
from .trace import atomic_write, file_digest, json_bytes, sync_directory


def validate_decision_state(state: dict[str, Any]) -> None:
    for key in ("known_trajectory_ids", "known_evidence_ids", "active_reference_ids",
                "available_actions", "role_snapshot_refs"):
        values = state.get(key)
        if not isinstance(values, list) or any(not isinstance(v, str) or not v for v in values):
            raise ValueError(f"decision_state.{key} must be a string array")
        if len(set(values)) != len(values):
            raise ValueError(f"duplicate decision_state.{key}")
    for key in ("chosen_action", "policy_id", "policy_version", "budget_snapshot_ref",
                "state_timestamp", "source_commit"):
        if not isinstance(state.get(key), str) or not state[key]:
            raise ValueError(f"missing decision_state.{key}")
    if state["chosen_action"] not in state["available_actions"]:
        raise ValueError("chosen action unavailable at decision time")
    if not set(state["active_reference_ids"]) <= set(state["known_evidence_ids"]):
        raise ValueError("reference unknown at decision time")


def plan(repo_root: str | Path, spec: dict[str, Any], output: str | Path) -> dict[str, Any]:
    from .policy import load_policy_registry

    root = Path(repo_root).resolve()
    destination = repo_local_path(root, output)
    destination.relative_to(root / "experiments")
    if destination.exists():
        raise ValueError("plan output must be a new experiment directory")
    trajectory = copy.deepcopy(spec["trajectory"])
    token = hashlib.sha256(json_bytes(spec)).hexdigest()[:20]
    trajectory.setdefault("schema", "molgap-trajectory-v1")
    trajectory.setdefault("record_mode", "prospective")
    trajectory.setdefault("owner", "server")
    trajectory.setdefault("trajectory_id", "T-" + token)
    trajectory["hypothesis"].setdefault("hypothesis_id", "H-" + token)
    trajectory.setdefault("result", {"evidence_ids": [], "evidence_refs": []})
    if trajectory["owner"] != "server" or trajectory["record_mode"] != "prospective":
        raise ValueError("plan creates only server prospective trajectories")
    if trajectory["decision"]["outcome"] != "ACTIVE" or trajectory["result"]["evidence_ids"]:
        raise ValueError("new plan must be ACTIVE with no terminal evidence")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    discovered = discover_records(root)
    known_trajectories = sorted(load_json_object(p)["trajectory_id"] for p in discovered.trajectories)
    known_evidence = sorted(load_json_object(p)["evidence_id"] for p in discovered.evidence)
    if trajectory["trajectory_id"] in known_trajectories:
        raise ValueError("trajectory ID already exists")
    if trajectory["hypothesis"]["hypothesis_id"] in {load_json_object(p)["hypothesis"]["hypothesis_id"] for p in discovered.trajectories}:
        raise ValueError("hypothesis ID already exists")
    state = trajectory["state_at_start"]
    state["source_commit"] = commit
    decision_state = copy.deepcopy(spec["decision_state"])
    decision_state.update(known_trajectory_ids=known_trajectories, known_evidence_ids=known_evidence,
                          active_reference_ids=state["reference_ids"],
                          budget_snapshot_ref=state["budget_snapshot_ref"],
                          role_snapshot_refs=state["role_snapshot_refs"], source_commit=commit)
    validate_decision_state(decision_state)
    for field in ("parent_trajectory_ids", "prior_trajectory_ids"):
        if not set(state.get(field, [])) <= set(known_trajectories):
            raise ValueError("plan references unknown prior trajectories")
    for values in (state["prior_evidence_ids"], trajectory["hypothesis"]["supporting_evidence_ids"]):
        if not set(values) <= set(known_evidence):
            raise ValueError("plan references unknown evidence")
    policies = load_policy_registry(root)
    policy = next((p for p in policies if (p["policy_id"], p["version"]) ==
                   (decision_state["policy_id"], decision_state["policy_version"])), None)
    if policy is None:
        raise ValueError("plan requires a registered explicit policy version")
    pointers = [*state["contract_refs"], *state["role_snapshot_refs"], state["budget_snapshot_ref"],
                trajectory["decision"]["decision_ref"]]
    pointers += [str(p.relative_to(root)) for p in (*discovered.trajectories, *discovered.evidence)]
    bindings = {}
    for pointer in sorted(set(pointers)):
        path = resolve_repo_pointer(root, pointer)
        if path is None:
            raise ValueError("planning state requires locally frozen bindings")
        bindings[pointer] = file_digest(path)
    decision_state["source_hashes"] = bindings
    decision_state["policy_sha256"] = hashlib.sha256(json_bytes(policy)).hexdigest()
    if spec.get("action_inputs_ref") is not None:
        action_path = resolve_repo_pointer(root, spec["action_inputs_ref"])
        if action_path is None:
            raise ValueError("action inputs must be retained locally")
        action_inputs = load_json_object(action_path)
        if action_inputs.get("trajectory_id") != trajectory["trajectory_id"] or action_inputs.get("state_timestamp") != decision_state["state_timestamp"]:
            raise ValueError("action input identity differs from prospective decision state")
        if not set(action_inputs["evidence_ids"]) <= set(known_evidence):
            raise ValueError("action input uses unknown evidence")
        decision_state["action_inputs_sha256"] = file_digest(action_path)
        decision_state["source_hashes"][spec["action_inputs_ref"]] = file_digest(action_path)
    trajectory["decision_state"] = decision_state
    validate_trajectory(trajectory)
    files = {"trajectory.json": json_bytes(trajectory),
             "decision_state.json": json_bytes(decision_state), "policy_snapshot.json": json_bytes(policy)}
    costs = spec.get("costs", [])
    cost_ids = set()
    existing_cost_ids = {load_json_object(p)["cost_event_id"] for p in discovered.costs}
    action_ids = {a["action_id"] for a in trajectory["actions"]}
    for event in costs:
        event = copy.deepcopy(event)
        event.setdefault("trajectory_id", trajectory["trajectory_id"])
        validate_cost_event(event)
        if event["trajectory_id"] != trajectory["trajectory_id"] or event["action_id"] not in action_ids:
            raise ValueError("plan cost identity mismatch")
        if event["cost_event_id"] in cost_ids | existing_cost_ids:
            raise ValueError("duplicate plan cost ID")
        if any(m["status"] == "measured" for m in event["measurement"].values()):
            raise ValueError("prospective cost cannot be measured")
        resolve_repo_pointer(root, event["evidence_ref"])
        cost_ids.add(event["cost_event_id"])
        files[f"costs/{event['cost_event_id']}.json"] = json_bytes(event)
    required_costs = {trajectory["hypothesis"]["expected_native_cost_ref"]}
    required_costs.update(c for a in trajectory["actions"] for c in a["cost_event_ids"])
    if not required_costs <= cost_ids:
        raise ValueError("plan spec must supply its referenced prospective cost events")
    staging_root = repo_local_path(root, "research_memory/.staging")
    staging_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="plan-", dir=staging_root))
    try:
        for name, data in files.items():
            atomic_write(staging / name, data)
        for pointer, digest in decision_state["source_hashes"].items():
            if file_digest(resolve_repo_pointer(root, pointer)) != digest:
                raise ValueError("decision source changed during planning")
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.rename(staging, destination)
        sync_directory(destination.parent)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return {"trajectory_id": trajectory["trajectory_id"], "status": "PLANNED",
            "path": destination.relative_to(root).as_posix()}
