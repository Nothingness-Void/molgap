"""Per-arm prospective planning boundary for the thin ExperimentSpec CLI."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import stat

from .common.identity import validate_execution_identity
from .experiment_launch import _safe_local
from .experiment_spec import ExperimentSpec, SCHEMA_VERSION_V2, _canonical, _unique_object
from .research_memory.compiler import rebuild_research_memory
from .research_memory.paths import repo_local_path
from .research_memory.plan import PlanBatchError, plan_many
from .research_memory.paired import PAIR_SCHEMA
from .screen_policy import canonical_fingerprint


def _prospective_inputs(spec: ExperimentSpec, repo_root: Path) -> tuple[list, list]:
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

    return plans, bindings


def check_prospective(spec: ExperimentSpec, repo_root: Path) -> dict:
    """Use actual RML validation for every arm, without publication or rebuild."""
    plans, _ = _prospective_inputs(spec, repo_root)
    result = plan_many(repo_root, plans, validate_only=True)
    if result.get("status") != "VALIDATED" or len(result.get("results", [])) != len(plans):
        raise ValueError("Prospective dry-validation result mismatch")
    return {"status": "PROSPECTIVE_VALIDATED_ONLY", "batch": result["batch"],
            "records": result["results"], "published": False, "rml_rebuilt": False,
            "submission_authorized": False}


def plan_prospective(spec: ExperimentSpec, repo_root: Path, *, execution_source_commit=None) -> tuple[dict, int]:
    # Derived views are regenerable, but a user's uncommitted edits are not ours
    # to overwrite as a side effect of publishing a new scientific trajectory.
    import subprocess
    root = Path(repo_root).absolute()
    protect_derived = False
    if (root / ".git").exists():
        status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all", "--",
            "research_memory/derived"], cwd=root, text=True, capture_output=True)
        protect_derived = status.returncode != 0 or bool(status.stdout.strip())
    if execution_source_commit is not None:
        import subprocess
        observed = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
        if observed != execution_source_commit:
            raise ValueError("Packaged execution commit differs from prospective planning HEAD")
    plans, bindings = _prospective_inputs(spec, repo_root)

    if execution_source_commit is not None:
        # Authored plan files predate their packaging commit. Bind executable
        # provenance at the existing publication boundary, not by editing RML.
        for item in plans:
            trajectory = item["spec"]["trajectory"]
            trajectory["state_at_start"]["source_commit"] = execution_source_commit
            for action in trajectory.get("actions", []):
                action["source_commit"] = execution_source_commit
            if "decision_state" in item["spec"]:
                item["spec"]["decision_state"]["source_commit"] = execution_source_commit
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
    if protect_derived:
        return {"status": "PROSPECTIVE_PLANNED_DERIVED_DEFERRED_USER_CHANGES",
            "completed_records": records, "batch": result["batch"],
            "rml_derived_files": [], "rml_rebuilt": False,
            "scope": "prospective_planning_only", "submission_authorized": False,
            "training_authorized": False, "ready_for_desktop": False,
            "derived_rebuild_reason": "Preserved existing uncommitted derived views; canonical prospective records were published."}, 0
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
