"""Per-arm prospective planning boundary for the thin ExperimentSpec CLI."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import stat

from .experiment_launch import _safe_local
from .experiment_spec import ExperimentSpec, SCHEMA_VERSION_V2, _canonical, _unique_object
from .research_memory.compiler import rebuild_research_memory
from .research_memory.paths import repo_local_path
from .research_memory.plan import PlanBatchError, plan_many
from .research_memory.paired import PAIR_SCHEMA
from .screen_policy import canonical_fingerprint


def plan_prospective(spec: ExperimentSpec, repo_root: Path) -> tuple[dict, int]:
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
        output = repo_root / binding["output"]
        _safe_local(output)
        repo_local_path(repo_root, binding["output"])
        if output.exists():
            raise ValueError(f"prospective output already exists: {binding['output']}")
        plans.append({"spec": plan_spec, "output": binding["output"]})

    same_run = declaration["prospective"].get("same_run_replay")
    if same_run is not None:
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
    try:
        derived = rebuild_research_memory(repo_root)
    except Exception as exc:
        return {
            "status": "PROSPECTIVE_PLANNED_RML_REBUILD_FAILED",
            "completed_records": records,
            "error": {"type": type(exc).__name__, "message": str(exc)},
            "rml_rebuilt": False,
            "submission_authorized": False,
        }, 1
    return {
        "status": "PROSPECTIVE_PLANNED_AND_RML_REBUILT",
        "completed_records": records,
        "batch": result["batch"],
        "rml_derived_files": sorted(derived),
        "rml_rebuilt": True,
        "scope": "prospective_planning_only",
        "submission_authorized": False,
        "training_authorized": False,
        "ready_for_desktop": False,
    }, 0
