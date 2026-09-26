"""Freeze supplied scientific content and the knowledge available at planning."""

from __future__ import annotations

import copy
import hashlib
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any

from molgap.evidence_pointers import load_json_object
from .paths import repo_local_path, resolve_repo_pointer
from .discovery import DiscoveredRecords, discover_records
from .paired import pair_binding
from .schemas import validate_cost_event, validate_id, validate_trajectory
from .trace import atomic_write, file_digest, json_bytes, sync_directory


@dataclass(frozen=True)
class _PlanningSnapshot:
    discovered: DiscoveredRecords
    commit: str
    known_trajectories: list[str]
    known_evidence: list[str]
    existing_hypothesis_ids: frozenset[str]
    existing_cost_ids: frozenset[str]
    policies: list[dict[str, Any]]
    source_hashes: dict[str, str]


class PlanBatchError(RuntimeError):
    """A batch stopped after zero or more plans were already published."""

    def __init__(self, index: int, size: int, completed_results: list[dict[str, Any]], cause: Exception):
        self.failed_index = index
        self.completed_results = tuple(completed_results)
        super().__init__(
            f"batch plan failed at item {index + 1}/{size}: {cause}; "
            f"{len(completed_results)} earlier plan(s) remain published"
        )


def _prepare_trajectory(spec: dict[str, Any]) -> dict[str, Any]:
    trajectory = copy.deepcopy(spec["trajectory"])
    token = hashlib.sha256(json_bytes(spec)).hexdigest()[:20]
    trajectory.setdefault("schema", "molgap-trajectory-v1")
    trajectory.setdefault("record_mode", "prospective")
    trajectory.setdefault("owner", "server")
    trajectory.setdefault("trajectory_id", "T-" + token)
    trajectory["hypothesis"].setdefault("hypothesis_id", "H-" + token)
    trajectory.setdefault("result", {"evidence_ids": [], "evidence_refs": []})
    return trajectory


def _source_pointers(root: Path, trajectory: dict[str, Any], discovered: DiscoveredRecords) -> list[str]:
    state = trajectory["state_at_start"]
    pointers = [*state["contract_refs"], *state["role_snapshot_refs"],
                state["budget_snapshot_ref"], trajectory["decision"]["decision_ref"]]
    pointers += [str(p.relative_to(root)) for p in (*discovered.trajectories, *discovered.evidence)]
    return pointers


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


def plan(repo_root: str | Path, spec: dict[str, Any], output: str | Path,
         *, _snapshot: _PlanningSnapshot | None = None) -> dict[str, Any]:
    from .policy import load_policy_registry

    root = Path(repo_root).resolve()
    destination = repo_local_path(root, output)
    destination.relative_to(root / "experiments")
    if destination.exists():
        raise ValueError("plan output must be a new experiment directory")
    trajectory = _prepare_trajectory(spec)
    if pair_binding(trajectory) is not None and _snapshot is None:
        raise ValueError("same-run replay must be planned in one frozen multi-arm batch")
    if trajectory["owner"] != "server" or trajectory["record_mode"] != "prospective":
        raise ValueError("plan creates only server prospective trajectories")
    if trajectory["decision"]["outcome"] != "ACTIVE" or trajectory["result"]["evidence_ids"]:
        raise ValueError("new plan must be ACTIVE with no terminal evidence")
    commit = (_snapshot.commit if _snapshot is not None else
              subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip())
    discovered = _snapshot.discovered if _snapshot is not None else discover_records(root)
    known_trajectories = (_snapshot.known_trajectories if _snapshot is not None else
                          sorted(load_json_object(p)["trajectory_id"] for p in discovered.trajectories))
    known_evidence = (_snapshot.known_evidence if _snapshot is not None else
                      sorted(load_json_object(p)["evidence_id"] for p in discovered.evidence))
    if trajectory["trajectory_id"] in known_trajectories:
        raise ValueError("trajectory ID already exists")
    existing_hypothesis_ids = (_snapshot.existing_hypothesis_ids if _snapshot is not None else
                               {load_json_object(p)["hypothesis"]["hypothesis_id"] for p in discovered.trajectories})
    if trajectory["hypothesis"]["hypothesis_id"] in existing_hypothesis_ids:
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
    policies = _snapshot.policies if _snapshot is not None else load_policy_registry(root)
    policy = next((p for p in policies if (p["policy_id"], p["version"]) ==
                   (decision_state["policy_id"], decision_state["policy_version"])), None)
    if policy is None:
        raise ValueError("plan requires a registered explicit policy version")
    pointers = _source_pointers(root, trajectory, discovered)
    if _snapshot is not None:
        if not set(pointers) <= set(_snapshot.source_hashes):
            raise ValueError("plan source is absent from the frozen batch snapshot")
        bindings = dict(_snapshot.source_hashes)
    else:
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
        if _snapshot is not None:
            digest = _snapshot.source_hashes.get(spec["action_inputs_ref"])
            if digest is None or file_digest(action_path) != digest:
                raise ValueError("action inputs changed since the frozen batch snapshot")
            decision_state["action_inputs_sha256"] = digest
            decision_state["source_hashes"][spec["action_inputs_ref"]] = digest
        else:
            decision_state["action_inputs_sha256"] = file_digest(action_path)
            decision_state["source_hashes"][spec["action_inputs_ref"]] = file_digest(action_path)
    trajectory["decision_state"] = decision_state
    validate_trajectory(trajectory)
    files = {"trajectory.json": json_bytes(trajectory),
             "decision_state.json": json_bytes(decision_state), "policy_snapshot.json": json_bytes(policy)}
    costs = spec.get("costs", [])
    cost_ids = set()
    existing_cost_ids = (_snapshot.existing_cost_ids if _snapshot is not None else
                         {load_json_object(p)["cost_event_id"] for p in discovered.costs})
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


def plan_many(repo_root: str | Path, plans: list[dict[str, Any]]) -> dict[str, Any]:
    """Publish independent prospective plans against one pre-publication snapshot."""
    from .policy import load_policy_registry

    if not isinstance(plans, list) or not plans:
        raise ValueError("plans must be a non-empty array")
    root = Path(repo_root).resolve()
    experiment_root = root / "experiments"
    prepared = []
    destinations = set()
    for index, item in enumerate(plans):
        if not isinstance(item, dict) or set(item) != {"spec", "output"} or not isinstance(item["spec"], dict):
            raise ValueError(f"batch item {index + 1} must contain exactly spec and output")
        output = item["output"]
        if (not isinstance(output, (str, Path)) or Path(output).is_absolute()
                or Path(output).drive or PureWindowsPath(output).root or PureWindowsPath(output).drive):
            raise ValueError(f"batch item {index + 1} output must be repository-relative")
        destination = repo_local_path(root, output)
        try:
            relative = destination.relative_to(experiment_root)
        except ValueError as exc:
            raise ValueError(f"batch item {index + 1} output must be under experiments") from exc
        if not relative.parts:
            raise ValueError(f"batch item {index + 1} output must be a new experiment directory")
        if destination in destinations:
            raise ValueError(f"duplicate batch output: {output}")
        if destination.exists():
            raise ValueError(f"batch output already exists: {output}")
        destinations.add(destination)
        spec = copy.deepcopy(item["spec"])
        prepared.append((spec, destination.relative_to(root), _prepare_trajectory(spec)))

    paired = [(trajectory, output, pair_binding(trajectory)) for _, output, trajectory in prepared]
    paired = [(trajectory, output, binding) for trajectory, output, binding in paired if binding is not None]
    if paired:
        groups: dict[tuple[str, str], list[tuple[dict[str, Any], Path, dict[str, str]]]] = {}
        for trajectory, output, binding in paired:
            groups.setdefault((binding["spec_identity"], binding["logical_run_id"]), []).append(
                (trajectory, output, binding)
            )
        for group in groups.values():
            references = [row for row in group if row[2]["comparison_role"] == "reference"]
            if len(references) != 1 or len(group) < 2:
                raise ValueError("same-run replay batch requires one reference and at least one candidate")
            if len({binding["arm_id"] for _, _, binding in group}) != len(group):
                raise ValueError("same-run replay batch repeats an arm identity")
            source_commits = {
                trajectory["state_at_start"].get("source_commit") for trajectory, _, _ in group
            }
            if len(source_commits) > 1:
                raise ValueError("same-run arms must freeze one source commit")
            reference, reference_output, reference_binding = references[0]
            expected_ref = (reference_output / "trajectory.json").as_posix()
            if reference_binding["reference_trajectory_ref"] != expected_ref:
                raise ValueError("same-run reference path differs from batch output")
            for trajectory, _, binding in group:
                if (binding["reference_trajectory_id"], binding["reference_trajectory_ref"],
                        binding["reference_arm_id"]) != (
                            reference["trajectory_id"], expected_ref, reference_binding["arm_id"]):
                    raise ValueError("same-run candidate does not bind the batch reference")

    discovered = discover_records(root)
    existing_trajectories = [load_json_object(p) for p in discovered.trajectories]
    known_trajectories = sorted(record["trajectory_id"] for record in existing_trajectories)
    known_evidence = sorted(load_json_object(p)["evidence_id"] for p in discovered.evidence)
    existing_hypotheses = frozenset(record["hypothesis"]["hypothesis_id"] for record in existing_trajectories)
    existing_costs = frozenset(load_json_object(p)["cost_event_id"] for p in discovered.costs)
    trajectory_ids = set(known_trajectories)
    hypothesis_ids = set(existing_hypotheses)
    cost_ids = set(existing_costs)
    policy_key = None
    for index, (spec, _, trajectory) in enumerate(prepared):
        for label, value, seen in (
            ("trajectory_id", trajectory["trajectory_id"], trajectory_ids),
            ("hypothesis_id", trajectory["hypothesis"]["hypothesis_id"], hypothesis_ids),
        ):
            validate_id(value, f"batch item {index + 1} {label}")
            if value in seen:
                raise ValueError(f"batch item {index + 1} duplicate {label}: {value}")
            seen.add(value)
        for event in spec.get("costs", []):
            cost_id = validate_id(event["cost_event_id"], f"batch item {index + 1} cost_event_id")
            if cost_id in cost_ids:
                raise ValueError(f"batch item {index + 1} duplicate cost_event_id: {cost_id}")
            cost_ids.add(cost_id)
        state = spec["decision_state"]
        item_policy_key = state["policy_id"], state["policy_version"]
        if policy_key is None:
            policy_key = item_policy_key
        elif item_policy_key != policy_key:
            raise ValueError("batch plans must use the same policy version")

    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    policies = load_policy_registry(root)
    policy = next((p for p in policies if (p["policy_id"], p["version"]) == policy_key), None)
    if policy is None:
        raise ValueError("batch plans require a registered explicit policy version")
    pointers = set()
    for spec, _, trajectory in prepared:
        pointers.update(_source_pointers(root, trajectory, discovered))
        if spec.get("action_inputs_ref") is not None:
            pointers.add(spec["action_inputs_ref"])
    bindings = {}
    for pointer in sorted(pointers):
        path = resolve_repo_pointer(root, pointer)
        if path is None:
            raise ValueError("planning state requires locally frozen bindings")
        bindings[pointer] = file_digest(path)
    snapshot = _PlanningSnapshot(
        discovered=discovered,
        commit=commit,
        known_trajectories=known_trajectories,
        known_evidence=known_evidence,
        existing_hypothesis_ids=existing_hypotheses,
        existing_cost_ids=existing_costs,
        policies=policies,
        source_hashes=bindings,
    )
    results = []
    for index, (spec, output, _) in enumerate(prepared):
        try:
            results.append(plan(root, spec, output, _snapshot=snapshot))
        except Exception as exc:
            raise PlanBatchError(index, len(prepared), results, exc) from exc
    return {
        "status": "PLANNED",
        "results": results,
        "batch": {
            "size": len(results),
            "source_commit": commit,
            "policy_sha256": hashlib.sha256(json_bytes(policy)).hexdigest(),
            "source_hashes_sha256": hashlib.sha256(json_bytes(bindings)).hexdigest(),
        },
    }
