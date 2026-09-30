"""Compose existing prospective and release staging APIs, without a submitter."""
from __future__ import annotations

import json
from pathlib import Path

from .experiment_launch import _safe_local
from .experiment_package import _allowlist, _name, _spec
from .experiment_prospective import plan_prospective
from .experiment_spec import SCHEMA_VERSION_V2, _digest, _unique_object
from .experiment_staging import UploadArtifact, _artifact_name, stage_release_inputs
from .research_memory.trace import atomic_write, json_bytes

WORKFLOW_FORMAT = "molgap-release-workflow-v1"
_FIELDS = {
    "format", "spec_identity", "source_paths", "artifacts", "recipe_files",
    "initial_states", "required_modules", "entry_template", "kernel_metadata",
    "dataset_metadata", "pickle_inputs",
}


def _path(root: Path, value: str) -> Path:
    if type(value) is not str or not value:
        raise ValueError("Workflow paths must be explicit nonempty strings")
    path = Path(value)
    path = path if path.is_absolute() else root / path
    _safe_local(path)
    return path.absolute()


def release_workflow_inputs(spec, repo_root: Path, raw: str) -> dict:
    """Parse transport configuration, never infer a protocol or fabricate RML."""
    spec = _spec(spec)
    config = json.loads(raw, object_pairs_hook=_unique_object)
    if type(config) is not dict or set(config) != _FIELDS or config["format"] != WORKFLOW_FORMAT:
        raise ValueError("Missing/unknown release workflow fields or format")
    json_bytes(config)  # Reject NaN/Infinity before publishing prospective records.
    if config["spec_identity"] != spec.identity:
        raise ValueError("Release workflow differs from the frozen Spec")
    for name in ("source_paths", "required_modules", "pickle_inputs"):
        values = config[name]
        if type(values) is not list or any(type(value) is not str or not value for value in values):
            raise ValueError(f"Invalid workflow list: {name}")
        if name != "pickle_inputs" and not values:
            raise ValueError(f"Empty workflow list: {name}")
    for name in ("artifacts", "recipe_files", "initial_states"):
        if type(config[name]) is not dict:
            raise ValueError(f"Invalid workflow mapping: {name}")
    _allowlist(config["source_paths"])
    arms = {arm["arm_id"] for arm in spec.to_dict()["arms"]}
    if set(config["recipe_files"]) != arms or set(config["initial_states"]) - arms:
        raise ValueError("Workflow recipe/initialization arm bindings differ from Spec")
    for value in config["recipe_files"].values():
        if _name(value) not in config["source_paths"]:
            raise ValueError("Workflow recipes must be explicit packaged source")
    if any(type(name) is not str or name not in config["artifacts"] for name in config["initial_states"].values()):
        raise ValueError("Workflow initialization must name a staged artifact")
    if config["dataset_metadata"] is not None and type(config["dataset_metadata"]) is not dict:
        raise ValueError("Dataset metadata must be explicit object or null")
    artifacts = {}
    for name, value in config["artifacts"].items():
        _artifact_name(name)
        if type(value) is not dict or set(value) != {"path", "sha256"}:
            raise ValueError("Each artifact requires path and SHA256")
        _digest(value["sha256"], "workflow.artifact.sha256")
        artifacts[name] = UploadArtifact(_path(repo_root, value["path"]), value["sha256"])
    return {
        "relative_paths": config["source_paths"], "artifacts": artifacts,
        "recipe_files": config["recipe_files"], "initial_states": config["initial_states"],
        "required_modules": config["required_modules"],
        "entry_template": _path(repo_root, config["entry_template"]),
        "kernel_metadata": _path(repo_root, config["kernel_metadata"]),
        "dataset_metadata": config["dataset_metadata"],
        "pickle_inputs": [_path(repo_root, value) for value in config["pickle_inputs"]],
    }


def prepare_experiment_release(spec, repo_root: Path, workflow_raw: str, output: Path) -> tuple[dict, int]:
    """Plan all arms, assemble upload inputs and retain a single local receipt.

    A partial plan or a failed staging attempt is retained, never deleted or
    automatically replanned. Caller must reconcile before a retry. This does
    not select an account, POST a kernel, release compute or finalize evidence.
    """
    spec = _spec(spec)
    if spec.to_dict()["schema_version"] != SCHEMA_VERSION_V2:
        raise ValueError("prepare-release requires per-arm Spec v2 planning")
    repo_root, output = Path(repo_root).absolute(), Path(output).absolute()
    _safe_local(repo_root)
    _safe_local(output)
    inputs = release_workflow_inputs(spec, repo_root, workflow_raw)
    output.mkdir(parents=True, exist_ok=False)
    result = {"spec_identity": spec.identity, "submitted": False, "compute_released": False}
    try:
        planned, code = plan_prospective(spec, repo_root)
        result["prospective"] = planned
        if code:
            result["status"] = "PLANNING_BLOCKED_RECONCILE_BEFORE_RETRY"
        else:
            staged = stage_release_inputs(spec, repo_root, output=output / "release", **inputs)
            result["staging"] = staged
            code = 1 if staged["errors"] else 0
            result["status"] = "LOCAL_PREPARATION_BLOCKED" if code else "LOCAL_PREPARATION_COMPLETE"
        atomic_write(output / "workflow.json", json_bytes(result))
        return result, code
    except Exception as exc:
        atomic_write(output / "workflow.json", json_bytes({
            **result, "status": "PREPARATION_ERROR_RECONCILE_BEFORE_RETRY", "error": str(exc),
        }))
        raise
