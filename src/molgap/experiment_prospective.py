"""Per-arm prospective planning boundary for the thin ExperimentSpec CLI."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import stat

from .experiment_launch import _safe_local
from .common.identity import validate_execution_identity
from .experiment_spec import ExperimentSpec, SCHEMA_VERSION_V2, _canonical, _unique_object
from .evidence_pointers import load_json_object
from .research_memory.discovery import discover_records
from .research_memory.paths import repo_local_path, resolve_repo_pointer, verify_bound_artifact
from .research_memory.plan import PlanBatchError, plan_many
from .research_memory.paired import PAIR_SCHEMA
from .screen_policy import canonical_fingerprint
from .v5_common import validate_comparison_readiness, validate_v5_evidence_envelope


def _check_referenced_evidence(repo_root: Path, plans: list[dict]) -> None:
    """Validate this question's evidence without hydrating unrelated history."""
    referenced = set()
    pointers = set()
    for item in plans:
        plan = item["spec"]
        trajectory = plan["trajectory"]
        state = trajectory["state_at_start"]
        for values in (state.get("reference_ids", []), state.get("prior_evidence_ids", []),
                       trajectory.get("hypothesis", {}).get("supporting_evidence_ids", [])):
            referenced.update(values)
        pointers.update(state.get("contract_refs", []))
        if trajectory.get("comparison_readiness_ref"):
            pointer = trajectory["comparison_readiness_ref"]
            resolve_repo_pointer(repo_root, pointer)
            pointers.add(pointer)
        for action in trajectory.get("actions", []):
            for pointer in action.get("evidence_refs", []):
                resolve_repo_pointer(repo_root, pointer)
                pointers.add(pointer)
        for pointer in trajectory.get("result", {}).get("evidence_refs", []):
            resolve_repo_pointer(repo_root, pointer)
            pointers.add(pointer)
        if plan.get("action_inputs_ref") is not None:
            path = resolve_repo_pointer(repo_root, plan["action_inputs_ref"])
            referenced.update(load_json_object(path)["evidence_ids"])
    if not referenced and not pointers:
        return
    discovered = discover_records(repo_root)
    found = set()
    for path in discovered.evidence:
        record = load_json_object(path)
        evidence_id = record["evidence_id"]
        if evidence_id in referenced:
            if evidence_id in found:
                raise ValueError(f"duplicate referenced evidence ID: {evidence_id}")
            validate_v5_evidence_envelope(record, repo_root=repo_root)
            pointers.update(record["authority"]["pointers"])
            pointers.update(artifact["locator"] for artifact in record["artifacts"])
            found.add(evidence_id)
    if referenced - found:
        raise ValueError(f"plan references unknown evidence: {sorted(referenced - found)}")
    for path in discovered.comparison_readiness:
        relative = path.relative_to(repo_root.resolve()).as_posix()
        if relative in pointers or "repo://" + relative in pointers:
            validate_comparison_readiness(load_json_object(path),
                evidence_verifier=lambda pointer, digest: verify_bound_artifact(repo_root, pointer, digest))


def _prospective_inputs(spec: ExperimentSpec, repo_root: Path, *, execution_source_commit=None):
    declaration = spec.to_dict()
    if declaration["schema_version"] != SCHEMA_VERSION_V2:
        raise ValueError("plan-prospective requires molgap-experiment-spec-v2")
    if not repo_root.is_dir():
        raise ValueError("repo-root must be an existing local directory")

    arms = {arm["arm_id"]: arm for arm in declaration["arms"]}
    bindings = declaration["prospective"]["arms"]
    plans = []
    for binding in bindings:
        path = repo_root / binding["plan_spec_ref"]
        _safe_local(path)
        path = repo_local_path(repo_root, binding["plan_spec_ref"])
        if not path.is_file() or not stat.S_ISREG(path.stat().st_mode) or path.stat().st_nlink != 1:
            raise ValueError(f"plan input must be an unlinked regular file: {binding['plan_spec_ref']}")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != binding["plan_spec_sha256"]:
            raise ValueError(f"plan input SHA mismatch: {binding['plan_spec_ref']}")
        plan_spec = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object)
        if type(plan_spec) is not dict:
            raise ValueError("RML plan input must be a JSON object")
        _canonical(plan_spec)
        trajectory = plan_spec.get("trajectory")
        state = trajectory.get("state_at_start") if type(trajectory) is dict else None
        if type(state) is not dict or trajectory.get("trajectory_id") != binding["trajectory_id"]:
            raise ValueError(f"RML plan trajectory identity mismatch: {binding['arm_id']}")
        if state.get("source_config_identity") != canonical_fingerprint(arms[binding["arm_id"]]):
            raise ValueError(f"RML plan arm identity mismatch: {binding['arm_id']}")
        if execution_source_commit is not None:
            validate_execution_identity(spec, binding["arm_id"], trajectory, execution_source_commit)
        output = repo_root / binding["output"]
        _safe_local(output)
        repo_local_path(repo_root, binding["output"])
        if output.exists():
            raise ValueError(f"prospective output already exists: {binding['output']}")
        plans.append({"spec": plan_spec, "output": binding["output"]})

    same_run = declaration["prospective"].get("same_run_replay")
    if same_run is not None:
        if execution_source_commit is not None:
            attempts = {validate_execution_identity(spec, binding["arm_id"], item["spec"]["trajectory"],
                        execution_source_commit) for binding, item in zip(bindings, plans)}
            if len(attempts) != 1:
                raise ValueError("Same-run prospective arms must share one physical attempt version")
        by_arm = {entry["arm_id"]: entry for entry in bindings}
        reference = by_arm[same_run["reference_arm_id"]]
        reference_ref = reference["output"] + "/trajectory.json"
        for entry, item in zip(bindings, plans):
            if entry["arm_id"] == reference["arm_id"]:
                role = "reference"
            elif entry["arm_id"] in same_run["candidate_arm_ids"]:
                role = "candidate"
            else:
                continue
            state = item["spec"]["trajectory"]["state_at_start"]
            if "same_run_replay" in state:
                raise ValueError("plan input cannot override the spec-derived same-run replay binding")
            state["same_run_replay"] = {
                "schema": PAIR_SCHEMA,
                "spec_identity": spec.identity,
                "logical_run_id": declaration["logical_run_id"],
                "arm_id": entry["arm_id"],
                "comparison_role": role,
                "reference_arm_id": reference["arm_id"],
                "reference_trajectory_id": reference["trajectory_id"],
                "reference_trajectory_ref": reference_ref,
            }

    _check_referenced_evidence(repo_root, plans)
    return plans, bindings


def check_prospective(spec: ExperimentSpec, repo_root: Path) -> dict:
    """Run the existing RML validators without publishing or rebuilding indexes."""
    plans, _ = _prospective_inputs(spec, Path(repo_root))
    result = plan_many(repo_root, plans, validate_only=True)
    if result.get("status") != "VALIDATED" or len(result.get("results", [])) != len(plans):
        raise ValueError("Prospective dry-validation result mismatch")
    return {"status": "PROSPECTIVE_VALIDATED_ONLY", "batch": result["batch"],
            "records": result["results"], "published": False, "rml_rebuilt": False,
            "submission_authorized": False}


def plan_prospective(spec: ExperimentSpec, repo_root: Path, *, execution_source_commit=None) -> tuple[dict, int]:
    plans, bindings = _prospective_inputs(spec, repo_root, execution_source_commit=execution_source_commit)
    try:
        result = plan_many(repo_root, plans)
    except PlanBatchError as exc:
        return {
            "status": "PARTIAL_PROSPECTIVE_PLAN_REQUIRES_RECONCILIATION",
            "completed_records": list(exc.completed_results),
            "failed_index": exc.failed_index,
            "failed_arm_id": bindings[exc.failed_index]["arm_id"],
            "error": str(exc),
            "rml_rebuilt": False,
            "submission_authorized": False,
        }, 1

    records = result.get("results") if type(result) is dict else None
    expected = [
        (binding["trajectory_id"], binding["output"]) for binding in bindings
    ]
    if (type(result) is not dict or result.get("status") != "PLANNED"
            or type(result.get("batch")) is not dict or type(records) is not list
            or len(records) != len(expected)
            or result["batch"].get("size") != len(expected)
            or any(type(record) is not dict or record.get("status") != "PLANNED"
                   or (record.get("trajectory_id"), record.get("path")) != wanted
                   for record, wanted in zip(records, expected))):
        return {
            "status": "PROSPECTIVE_PLAN_RESULT_MISMATCH_REQUIRES_RECONCILIATION",
            "plan_result": result,
            "rml_rebuilt": False,
            "submission_authorized": False,
        }, 1
    return {
        "status": "PROSPECTIVE_PLANNED_RML_PENDING",
        "completed_records": records,
        "batch": result["batch"],
        "rml_rebuilt": False,
        "rml_rebuild_pending": True,
        "next_action": "Derived indexes are deferred to accepted milestones or handoff; use the existing RML rebuild CLI then.",
        "scope": "prospective_planning_only",
        "submission_authorized": False,
        "training_authorized": False,
        "ready_for_desktop": False,
    }, 0
