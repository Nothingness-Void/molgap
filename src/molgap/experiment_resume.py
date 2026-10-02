"""Portable, hash-bound transport for incomplete per-arm screen outputs.

This module owns transport and identity checks only.  It never selects a
trainer, imports a caller supplied callback, or loads a checkpoint with
pickle execution enabled.  Family-specific checkpoint/trace rules are
dispatched through the reviewed ``TrainingAdapter`` registry to the owning
module's ``validate_screen_resume`` hook.

Public transport API:
``create_resume_bundle(source_dir, context, trajectory, output, ...)``;
``validate_resume_bundle(bundle_path, spec=None, ...)``; and
``restore_resume_bundle(bundle_path, output_dir, spec=None, ...)``.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
from pathlib import Path
from typing import Any, Mapping

from .experiment_launch import _safe_local, publish_immutable_bytes
from .training_reproducibility import sha256_file
from .v4_runtime import torch_load_compat


RESUME_BUNDLE_FORMAT = "molgap-resume-bundle-v1"
RESUME_BUNDLE_FILE = "resume_bundle.json"
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_REQUIRED_ARTIFACTS = ("checkpoint", "trace", "trajectory")
_OPTIONAL_ARTIFACTS = ("selected_model", "predictions")
_ROOT_KEYS = {
    "format", "status", "source", "spec", "arm", "trajectory", "run",
    "cursor", "artifacts", "runtime", "provenance", "context",
    "cost_segments",
}


def _strict_json(path: Path) -> dict:
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"Duplicate JSON key: {key}")
            value[key] = item
        return value

    value = json.loads(path.read_bytes(), object_pairs_hook=unique,
                       parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))
    if type(value) is not dict:
        raise ValueError("Expected a JSON object")
    return value


def _strict_json_bytes(raw: bytes) -> dict:
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"Duplicate JSON key: {key}")
            value[key] = item
        return value

    value = json.loads(raw.decode("utf-8"), object_pairs_hook=unique,
                       parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))
    if type(value) is not dict:
        raise ValueError("Expected a JSON object")
    return value


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _digest(value: Any, label: str) -> str:
    if type(value) is not str or not _DIGEST.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase SHA256")
    return value


def _commit(value: Any, label: str) -> str:
    if type(value) is not str or not _COMMIT.fullmatch(value):
        raise ValueError(f"{label} must be a full source commit")
    return value


def _text(value: Any, label: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{label} must be nonempty text")
    return value


def _relative(value: Any, label: str = "artifact path") -> str:
    if (type(value) is not str or not value or value.startswith("/")
            or "\\" in value or ":" in value):
        raise ValueError(f"Unsafe {label}")
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"Unsafe {label}")
    return value


def _path(path: Path | str, label: str, *, regular: bool = False) -> Path:
    value = Path(path)
    if not value.is_absolute():
        value = value.absolute()
    _safe_local(value)
    if regular and (not value.exists() or not stat.S_ISREG(value.stat().st_mode)):
        raise ValueError(f"{label} must be a regular file")
    return value


def _context_dict(context: Any) -> dict:
    if hasattr(context, "to_dict"):
        value = context.to_dict()
    elif isinstance(context, Mapping):
        value = dict(context)
    else:
        raise TypeError("context must be RunContext or a mapping")
    if type(value) is not dict:
        raise ValueError("context must serialize to an object")
    required = {
        "experiment_id", "logical_run_id", "arm_id", "arm_identity",
        "spec_identity", "family_name", "family_version", "source_commit",
        "source_archive_sha256", "package_identity", "training_recipe_sha256",
        "platform", "account", "run_reference", "platform_version",
    }
    if set(value) != required:
        raise ValueError("context has missing or unknown identity fields")
    for key in ("experiment_id", "logical_run_id", "arm_id", "family_name",
                "family_version", "platform", "account", "run_reference"):
        _text(value[key], "context." + key)
    for key in ("arm_identity", "spec_identity", "source_archive_sha256",
                "package_identity", "training_recipe_sha256"):
        _digest(value[key], "context." + key)
    _commit(value["source_commit"], "context.source_commit")
    if value["platform_version"] is not None:
        _text(value["platform_version"], "context.platform_version")
    return value


def _load_trajectory(value: Any) -> tuple[dict, bytes, str | None]:
    if type(value) is bytes:
        return _strict_json_bytes(value), value, None
    if isinstance(value, (str, Path)):
        path = _path(value, "trajectory", regular=True)
        raw = path.read_bytes()
        parsed = _strict_json_bytes(raw)
        return parsed, raw, path.as_posix()
    if isinstance(value, Mapping):
        parsed = json.loads(_json_bytes(dict(value)).decode("utf-8"))
        return parsed, _json_bytes(parsed), None
    raise TypeError("trajectory must be JSON bytes, a path or mapping")


def _prospective_attempt(trajectory: dict, context: dict, *, run_id=None,
                         action_id=None, attempt_id=None) -> dict:
    if trajectory.get("trajectory_id") is None:
        raise ValueError("Prospective trajectory has no trajectory_id")
    if trajectory.get("record_mode") not in {"prospective", "prospective_partial"}:
        raise ValueError("Resume requires an active prospective trajectory")
    if trajectory.get("decision", {}).get("outcome") not in {None, "ACTIVE"}:
        raise ValueError("Resume trajectory is terminal")
    state = trajectory.get("state_at_start")
    if type(state) is not dict:
        raise ValueError("Prospective trajectory has no state_at_start")
    if state.get("source_commit") not in {None, context["source_commit"]}:
        raise ValueError("Prospective trajectory/source commit mismatch")
    if state.get("source_config_identity") not in {None, context["arm_identity"]}:
        raise ValueError("Prospective trajectory/arm identity mismatch")
    actions = trajectory.get("actions")
    if type(actions) is not list or not actions:
        raise ValueError("Prospective trajectory has no actions")
    candidates = []
    for action in actions:
        if type(action) is not dict:
            continue
        if action_id is not None and action.get("action_id") != action_id:
            continue
        run_ids = action.get("run_ids")
        attempts = action.get("attempt_ids")
        if type(run_ids) is not list or type(attempts) is not list:
            continue
        if run_id is not None and run_id not in run_ids:
            continue
        if run_id is None and len(run_ids) != 1:
            continue
        if attempt_id is not None and attempt_id not in attempts:
            continue
        if len(attempts) != 1:
            continue
        candidates.append(action)
    if len(candidates) != 1:
        raise ValueError("Resume requires one exact prospective action/run/attempt")
    action = candidates[0]
    if action.get("source_commit") not in {None, context["source_commit"]}:
        raise ValueError("Prospective action/source commit mismatch")
    resolved_run = run_id or action["run_ids"][0]
    resolved_attempt = attempt_id or action["attempt_ids"][0]
    _text(action.get("action_id"), "prospective action_id")
    _text(resolved_run, "prospective run_id")
    _text(resolved_attempt, "prospective attempt_id")
    return {"action_id": action["action_id"], "run_id": resolved_run,
            "attempt_id": resolved_attempt}


def _artifact_pointer(bundle_root: Path, pointer: dict, label: str, *, required=True) -> Path | None:
    if pointer is None:
        if required:
            raise ValueError(f"Missing resume artifact: {label}")
        return None
    if type(pointer) is not dict or set(pointer) != {"path", "sha256", "restore_path", "required"}:
        raise ValueError(f"Invalid resume artifact pointer: {label}")
    path = _relative(pointer["path"], label + ".path")
    restore = _relative(pointer["restore_path"], label + ".restore_path")
    _digest(pointer["sha256"], label + ".sha256")
    if (type(pointer["required"]) is not bool
            or (required and pointer["required"] is not True)
            or (not required and pointer["required"] is not False)):
        raise ValueError(f"Invalid required flag: {label}")
    resolved = _path(bundle_root / path, label, regular=True)
    if sha256_file(resolved) != pointer["sha256"]:
        raise ValueError(f"Resume artifact hash mismatch: {label}")
    return resolved


def _pointer(bundle_root: Path, source: Path, restore_path: str, *, required: bool) -> dict:
    _relative(restore_path, "restore path")
    return {"path": "files/" + source.name, "sha256": sha256_file(source),
            "restore_path": restore_path, "required": bool(required)}


def _safe_torch_tree(value: Any, label: str = "checkpoint") -> None:
    """Reject objects outside the weights-only CPU inspection domain."""
    import numpy as np
    import torch

    safe_scalars = (str, int, float, bool, bytes, type(None), np.generic, np.dtype)
    if torch.is_tensor(value):
        if value.device.type != "cpu":
            raise ValueError(f"{label} contains a non-CPU tensor")
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise ValueError(f"{label} contains a non-finite tensor")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, (str, int, float, bool, type(None))):
                raise ValueError(f"{label} contains an unsafe mapping key")
            _safe_torch_tree(item, label + "." + str(key))
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _safe_torch_tree(item, label)
        return
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise ValueError(f"{label} contains an object NumPy array")
        if np.issubdtype(value.dtype, np.floating) and not bool(np.isfinite(value).all()):
            raise ValueError(f"{label} contains a non-finite NumPy array")
        return
    if isinstance(value, safe_scalars):
        if isinstance(value, (float, np.floating)) and not bool(np.isfinite(value)):
            raise ValueError(f"{label} contains a non-finite scalar")
        return
    raise ValueError(f"{label} contains an unsupported object: {type(value).__name__}")


def safe_cpu_torch_load(path: Path) -> dict:
    """Load a trusted checkpoint with weights-only semantics and known NumPy RNG types."""
    import numpy as np
    import torch

    safe = []
    try:
        from numpy._core.multiarray import _reconstruct
    except ImportError:  # pragma: no cover - old NumPy spelling
        from numpy.core.multiarray import _reconstruct
    safe.extend([_reconstruct, np.ndarray, np.dtype])
    # GPTrans stores a NumPy uint32 RNG array.  NumPy 2 exposes concrete dtype
    # classes that PyTorch's weights-only unpickler requires explicitly.
    dtype_module = getattr(np, "dtypes", None)
    if dtype_module is not None:
        for name in dir(dtype_module):
            candidate = getattr(dtype_module, name, None)
            if isinstance(candidate, type) and name.endswith("DType"):
                safe.append(candidate)
    try:
        context = torch.serialization.safe_globals(safe)
    except AttributeError:  # pragma: no cover - PyTorch before 2.6
        context = None
    try:
        if context is None:
            for item in safe:
                torch.serialization.add_safe_globals([item])
            payload = torch_load_compat(path, map_location="cpu", weights_only=True)
        else:
            with context:
                payload = torch_load_compat(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError("Resume checkpoint is not a safe CPU tensor artifact") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("Resume checkpoint must contain a mapping")
    _safe_torch_tree(payload)
    return dict(payload)


def _sidecar_paths(source: Path) -> dict[str, list[Path]]:
    groups = {"runtime": [], "provenance": [], "context": [], "cost_segments": []}
    for item in sorted(source.iterdir(), key=lambda p: p.name.casefold()):
        if not item.is_file() or item.suffix.casefold() != ".json":
            continue
        lower = item.name.casefold()
        if lower in {"runtime_manifest.json", "runtime_certificate.json", "preflight.json",
                     "architecture_preflight.json", "diagnostic_preflight.json"}:
            groups["runtime"].append(item)
        elif "provenance" in lower:
            groups["provenance"].append(item)
        elif "context" in lower:
            groups["context"].append(item)
        elif ("cost" in lower or "allocation" in lower
              or lower == "allocation_ledger.json"):
            groups["cost_segments"].append(item)
    return groups


def _find_required(source: Path) -> tuple[Path, Path, Path | None, Path | None]:
    checkpoint = source / "last_checkpoint.pt"
    if not checkpoint.is_file():
        raise ValueError("Incomplete output has no last_checkpoint.pt")
    historical = source / "trace.json"
    canonical = source / "canonical_trace.json"
    # Historical GPTrans outputs retain ``trace.json`` beside a normalized
    # canonical trace.  Prefer the trainer-native trace whenever both exist;
    # K1 only writes the canonical name.
    trace = historical if historical.is_file() else canonical if canonical.is_file() else None
    if trace is None:
        raise ValueError("Incomplete output has no retained trace")
    selected = source / "selected_model.pt"
    if not selected.is_file():
        selected = source / "best_model.pt"
    predictions = source / "development_predictions.pt"
    return checkpoint, trace, selected if selected.is_file() else None, predictions if predictions.is_file() else None


def _native_cursor(checkpoint: dict, trace: dict) -> dict:
    """Extract only acknowledged counters, preserving each owner's epoch base."""
    cursor = dict(checkpoint.get("cursor")) if isinstance(checkpoint.get("cursor"), dict) else {
        "epoch": checkpoint.get("epoch"), "next_batch": 0,
    }
    if type(trace.get("rows")) is list and trace["rows"]:
        rows = trace["rows"]
        last = rows[-1]
        cursor.setdefault("optimizer_step", last.get("cumulative_optimizer_steps",
                                                       len(rows) * last.get("optimizer_steps", 0)))
        cursor.setdefault("sample_presentations", last.get("cumulative_sample_presentations",
                                                            len(rows) * last.get("sample_presentations", 0)))
    elif type(trace.get("observations")) is list:
        rows = [row for row in trace["observations"] if row.get("event") == "observation"]
        if rows:
            cursor.setdefault("optimizer_step", rows[-1].get("optimizer_step"))
            cursor.setdefault("sample_presentations", rows[-1].get("sample_presentations"))
    for key in ("optimizer_step", "sample_presentations"):
        if key not in cursor and key in checkpoint:
            cursor[key] = checkpoint[key]
    return cursor


def _reject_completed(source: Path) -> None:
    if (source / "output_manifest.json").is_file():
        raise ValueError("Completed family output cannot be exported as a resume bundle")
    completion = source / "completion_manifest.json"
    if completion.is_file() and _strict_json(completion).get("complete") is True:
        raise ValueError("Completed GPTrans output cannot be exported as a resume bundle")
    partial = source / "partial_manifest.json"
    if partial.is_file() and _strict_json(partial).get("complete") is True:
        raise ValueError("Completed partial output cannot be exported as a resume bundle")
    frozen = source / "frozen_reference.json"
    if frozen.is_file() and (source / "best_model.pt").is_file() and (source / "development_predictions.pt").is_file():
        raise ValueError("Completed output cannot be exported as a resume bundle")


def create_resume_bundle(source_dir: Path | str, context: Any, trajectory: Any,
                         output: Path | str, *, run_id: str | None = None,
                         action_id: str | None = None,
                         attempt_id: str | None = None) -> dict:
    """Freeze an incomplete native output into a portable per-arm bundle.

    The destination may be retried only with byte-identical content.  Source
    output is read but never modified.  The caller must supply the original
    active prospective trajectory; this function never creates a replacement
    prospective action or attempt.
    JSON bytes preserve an already verified snapshot without rereading a path.
    """
    source = _path(source_dir, "source output")
    if not source.is_dir():
        raise ValueError("source output must be a directory")
    context_value = _context_dict(context)
    bundle = _path(output, "resume bundle")
    if bundle == source or bundle.resolve().is_relative_to(source.resolve()) or source.resolve().is_relative_to(bundle.resolve()):
        raise ValueError("Resume bundle must be separate from source output")
    _reject_completed(source)
    checkpoint, trace, selected, predictions = _find_required(source)
    for item in (checkpoint, trace, selected, predictions):
        if item is not None:
            _path(item, "source artifact", regular=True)
    checkpoint_payload = safe_cpu_torch_load(checkpoint)
    trace_payload = _strict_json(trace)
    if (selected is not None and predictions is None) or (selected is None and predictions is not None):
        raise ValueError("Selected model and development predictions must be retained together")
    trajectory_value, trajectory_bytes, trajectory_source = _load_trajectory(trajectory)
    attempt = _prospective_attempt(trajectory_value, context_value, run_id=run_id,
                                   action_id=action_id, attempt_id=attempt_id)
    if trajectory_value["trajectory_id"] is None:
        raise ValueError("Missing trajectory identity")
    # A raw GPTrans checkpoint uses zero-based ``epoch`` while FamilyOutput
    # checkpoints use a one-based cursor.  Owners re-check the exact scheme;
    # this transport stores the observed raw identity without inventing work.
    cursor = _native_cursor(checkpoint_payload, trace_payload)
    sidecars = _sidecar_paths(source)
    artifact_sources: dict[str, tuple[Path, str, bool]] = {
        "checkpoint": (checkpoint, checkpoint.name, True),
        "trace": (trace, trace.name, True),
        "trajectory": (source / ".resume_trajectory.json", "trajectory.json", True),
    }
    if selected is not None:
        artifact_sources["selected_model"] = (selected, selected.name, False)
        artifact_sources["predictions"] = (predictions, predictions.name, False)
    all_sources = [item[0] for item in artifact_sources.values() if item[0].is_file()]
    # The synthetic trajectory source is encoded below and never read from the
    # source directory; avoid making a hidden source file there.
    source_bytes = {item: item.read_bytes() for item in all_sources if item.name != ".resume_trajectory.json"}
    pointers: dict[str, dict | None] = {}
    for role, (item, restore_name, required) in artifact_sources.items():
        if item.name == ".resume_trajectory.json":
            digest = hashlib.sha256(trajectory_bytes).hexdigest()
            pointers[role] = {"path": "files/trajectory.json", "sha256": digest,
                              "restore_path": "trajectory.json", "required": True}
        else:
            pointers[role] = {"path": "files/" + item.name, "sha256": sha256_file(item),
                              "restore_path": restore_name, "required": required}
    for role in _OPTIONAL_ARTIFACTS:
        pointers.setdefault(role, None)
    sidecar_pointers = {}
    for group, paths in sidecars.items():
        sidecar_pointers[group] = []
        for item in paths:
            sidecar_pointers[group].append(_pointer(bundle, item, item.name, required=True))
            source_bytes[item] = item.read_bytes()
    manifest = {
        "format": RESUME_BUNDLE_FORMAT,
        "status": "incomplete",
        "source": {"commit": context_value["source_commit"],
                    "archive_sha256": context_value["source_archive_sha256"],
                    "package_identity": context_value["package_identity"]},
        "spec": {"identity": context_value["spec_identity"]},
        "arm": {"id": context_value["arm_id"], "identity": context_value["arm_identity"],
                "family": {"name": context_value["family_name"], "version": context_value["family_version"]},
                "recipe_sha256": context_value["training_recipe_sha256"]},
        "trajectory": {"id": trajectory_value["trajectory_id"],
                       "sha256": hashlib.sha256(trajectory_bytes).hexdigest(),
                       "source_path": trajectory_source},
        "run": {"logical_run_id": context_value["logical_run_id"], **attempt},
        "cursor": cursor,
        "artifacts": pointers,
        "runtime": sidecar_pointers["runtime"],
        "provenance": sidecar_pointers["provenance"],
        "context": sidecar_pointers["context"],
        "cost_segments": sidecar_pointers["cost_segments"],
    }
    manifest_bytes = _json_bytes(manifest)
    # Validate every destination before publishing any byte, so one conflict
    # cannot leave a partially overwritten recovery package.
    destinations = [(bundle / RESUME_BUNDLE_FILE, manifest_bytes)]
    for role, pointer in pointers.items():
        if pointer is None:
            continue
        payload = trajectory_bytes if role == "trajectory" else source_bytes[next(
            item for item in source_bytes if item.name == Path(pointer["restore_path"]).name)]
        destinations.append((bundle / pointer["path"], payload))
    for group in sidecar_pointers.values():
        for pointer in group:
            payload = source_bytes[next(item for item in source_bytes if item.name == Path(pointer["restore_path"]).name)]
            destinations.append((bundle / pointer["path"], payload))
    seen = set()
    for path, payload in destinations:
        if path.as_posix().casefold() in seen:
            raise ValueError("Resume bundle artifact paths alias each other")
        seen.add(path.as_posix().casefold())
        if path.exists():
            if not path.is_file() or path.read_bytes() != payload:
                raise ValueError("Resume bundle publication conflict; refusing overwrite")
    for path, payload in destinations:
        publish_immutable_bytes(path, payload)
    return manifest


def _bundle_root(path: Path | str) -> tuple[Path, Path, dict]:
    candidate = _path(path, "resume bundle")
    root = candidate if candidate.is_dir() else candidate.parent
    manifest_path = candidate / RESUME_BUNDLE_FILE if candidate.is_dir() else candidate
    if not manifest_path.is_file():
        raise ValueError("Resume bundle manifest is missing")
    manifest = _strict_json(manifest_path)
    return root, manifest_path, manifest


def _validate_manifest(root: Path, manifest: dict) -> dict[str, Any]:
    if set(manifest) != _ROOT_KEYS or manifest.get("format") != RESUME_BUNDLE_FORMAT:
        raise ValueError("Unsupported resume bundle manifest")
    if manifest.get("status") != "incomplete":
        raise ValueError("Completed resume bundles are not restorable")
    source = manifest["source"]
    if type(source) is not dict or set(source) != {"commit", "archive_sha256", "package_identity"}:
        raise ValueError("Invalid resume source identity")
    _commit(source["commit"], "source.commit")
    _digest(source["archive_sha256"], "source.archive_sha256")
    _digest(source["package_identity"], "source.package_identity")
    spec = manifest["spec"]
    if type(spec) is not dict or set(spec) != {"identity"}:
        raise ValueError("Invalid resume Spec identity")
    _digest(spec["identity"], "spec.identity")
    arm = manifest["arm"]
    if type(arm) is not dict or set(arm) != {"id", "identity", "family", "recipe_sha256"}:
        raise ValueError("Invalid resume arm identity")
    _text(arm["id"], "arm.id")
    _digest(arm["identity"], "arm.identity")
    _digest(arm["recipe_sha256"], "arm.recipe_sha256")
    if type(arm["family"]) is not dict or set(arm["family"]) != {"name", "version"}:
        raise ValueError("Invalid resume family")
    _text(arm["family"]["name"], "arm.family.name")
    _text(arm["family"]["version"], "arm.family.version")
    trajectory = manifest["trajectory"]
    if type(trajectory) is not dict or set(trajectory) != {"id", "sha256", "source_path"}:
        raise ValueError("Invalid resume trajectory identity")
    _text(trajectory["id"], "trajectory.id")
    _digest(trajectory["sha256"], "trajectory.sha256")
    if trajectory["source_path"] is not None:
        _text(trajectory["source_path"], "trajectory.source_path")
    run = manifest["run"]
    if type(run) is not dict or set(run) != {"logical_run_id", "action_id", "run_id", "attempt_id"}:
        raise ValueError("Invalid resume prospective attempt")
    for key in run:
        _text(run[key], "run." + key)
    if type(manifest["cursor"]) is not dict:
        raise ValueError("Invalid resume cursor")
    artifacts = manifest["artifacts"]
    if type(artifacts) is not dict or set(artifacts) != set(_REQUIRED_ARTIFACTS) | set(_OPTIONAL_ARTIFACTS):
        raise ValueError("Resume bundle lacks explicit artifact roles")
    pointers: dict[str, Path | None] = {}
    for role in _REQUIRED_ARTIFACTS:
        pointers[role] = _artifact_pointer(root, artifacts[role], role, required=True)
    for role in _OPTIONAL_ARTIFACTS:
        pointers[role] = _artifact_pointer(root, artifacts[role], role, required=False)
    if (pointers["selected_model"] is None) != (pointers["predictions"] is None):
        raise ValueError("Selected model and predictions must be retained together")
    checkpoint_payload = safe_cpu_torch_load(pointers["checkpoint"])
    trace_payload = _strict_json(pointers["trace"])
    expected_cursor = _native_cursor(checkpoint_payload, trace_payload)
    if manifest["cursor"] != expected_cursor:
        raise ValueError("Resume manifest cursor/checkpoint/trace mismatch")
    all_bundle_paths = {pointers[role].as_posix().casefold() for role in pointers if pointers[role] is not None}
    all_restore = {manifest["artifacts"][role]["restore_path"].casefold()
                   for role in pointers if pointers[role] is not None}
    sidecars: dict[str, list[Path]] = {}
    for group in ("runtime", "provenance", "context", "cost_segments"):
        entries = manifest[group]
        if type(entries) is not list:
            raise ValueError(f"Resume {group} must be a list")
        sidecars[group] = []
        for index, entry in enumerate(entries):
            item = _artifact_pointer(root, entry, f"{group}[{index}]", required=True)
            assert item is not None
            sidecars[group].append(item)
            bundle_key = item.as_posix().casefold()
            restore_key = entry["restore_path"].casefold()
            if bundle_key in all_bundle_paths or restore_key in all_restore:
                raise ValueError("Resume sidecar aliases a primary artifact")
            all_bundle_paths.add(bundle_key)
            all_restore.add(restore_key)
    trajectory_file = pointers["trajectory"]
    assert trajectory_file is not None
    trajectory_value = _strict_json(trajectory_file)
    if trajectory_value.get("trajectory_id") != trajectory["id"]:
        raise ValueError("Resume trajectory identity mismatch")
    attempt = _prospective_attempt(trajectory_value, {
        "source_commit": source["commit"], "arm_identity": arm["identity"]},
        run_id=run["run_id"], action_id=run["action_id"], attempt_id=run["attempt_id"])
    if attempt != {key: run[key] for key in ("action_id", "run_id", "attempt_id")}:
        raise ValueError("Resume prospective attempt mismatch")
    return {"manifest": manifest, "root": root, "artifacts": pointers,
            "sidecars": sidecars, "trajectory": trajectory_value}


def _same_context(expected: dict, observed: dict) -> None:
    if type(observed) is not dict:
        raise ValueError("Resume context sidecar is not an object")
    if observed != expected:
        raise ValueError("Resume context identity mismatch")


def validate_resume_bundle(path: Path | str, spec=None, *, arm_id: str | None = None,
                           context: Any | None = None, trajectory: Any | None = None) -> dict:
    """Validate transport and dispatch the family-specific resume gate."""
    root, manifest_path, manifest = _bundle_root(path)
    checked = _validate_manifest(root, manifest)
    context_value = _context_dict(context) if context is not None else None
    arm_name = arm_id or manifest["arm"]["id"]
    if arm_name != manifest["arm"]["id"]:
        raise ValueError("Resume arm identity mismatch")
    if context_value is not None:
        expected = {"source_commit": context_value["source_commit"],
                    "source_archive_sha256": context_value["source_archive_sha256"],
                    "package_identity": context_value["package_identity"],
                    "spec_identity": context_value["spec_identity"],
                    "arm_id": context_value["arm_id"],
                    "arm_identity": context_value["arm_identity"],
                    "training_recipe_sha256": context_value["training_recipe_sha256"]}
        observed = {"source_commit": manifest["source"]["commit"],
                    "source_archive_sha256": manifest["source"]["archive_sha256"],
                    "package_identity": manifest["source"]["package_identity"],
                    "spec_identity": manifest["spec"]["identity"],
                    "arm_id": manifest["arm"]["id"],
                    "arm_identity": manifest["arm"]["identity"],
                    "training_recipe_sha256": manifest["arm"]["recipe_sha256"]}
        if observed != expected:
            raise ValueError("Resume source/Spec/arm/recipe identity mismatch")
    if trajectory is not None:
        value, raw, _ = _load_trajectory(trajectory)
        if hashlib.sha256(raw).hexdigest() != manifest["trajectory"]["sha256"]:
            raise ValueError("Resume prospective trajectory bytes changed")
        if value != checked["trajectory"]:
            raise ValueError("Resume prospective trajectory changed")
    # Context sidecars are checked generically where their format is the
    # RunContext object; owners validate family-specific provenance fields.
    if context_value is not None:
        for sidecar in checked["sidecars"]["context"]:
            payload = _strict_json(sidecar)
            if "context" in payload and isinstance(payload["context"], dict):
                _same_context(context_value, payload["context"])
            elif set(payload) == set(context_value):
                _same_context(context_value, payload)
        for sidecar in checked["sidecars"]["cost_segments"]:
            payload = _strict_json(sidecar)
            if ("spec_identity" in payload
                    and payload["spec_identity"] != context_value["spec_identity"]):
                raise ValueError("Resume prior cost segment/Spec identity mismatch")
    if spec is None:
        return {"status": "TRANSPORT_VERIFIED", "manifest": manifest,
                "manifest_path": manifest_path,
                "root": root, "artifacts": checked["artifacts"],
                "sidecars": checked["sidecars"], "trajectory": checked["trajectory"]}
    from .experiment_execution import training_adapter
    declaration = spec.to_dict()
    arm = next((item for item in declaration.get("arms", []) if item.get("arm_id") == arm_name), None)
    if arm is None:
        raise ValueError("Resume arm is absent from Spec")
    if spec.identity != manifest["spec"]["identity"]:
        raise ValueError("Resume Spec identity mismatch")
    from .screen_policy import canonical_fingerprint
    expected_arm_identity = canonical_fingerprint(arm)
    if expected_arm_identity != manifest["arm"]["identity"]:
        raise ValueError("Resume Spec arm identity mismatch")
    if arm["training"]["recipe"]["sha256"] != manifest["arm"]["recipe_sha256"]:
        raise ValueError("Resume Spec recipe identity mismatch")
    expected_family = (arm["family"]["name"], arm["family"]["version"])
    if (manifest["arm"]["family"]["name"], manifest["arm"]["family"]["version"]) != expected_family:
        raise ValueError("Resume family identity mismatch")
    prospective = declaration.get("prospective", {}).get("arms", [])
    binding = next((item for item in prospective if item.get("arm_id") == arm_name), None)
    if binding is None or binding.get("trajectory_id") != manifest["trajectory"]["id"]:
        raise ValueError("Resume prospective trajectory is not bound to this Spec arm")
    if context_value is not None and context_value["arm_identity"] != manifest["arm"]["identity"]:
        raise ValueError("Resume context arm identity mismatch")
    adapter = training_adapter(arm)
    from .experiment_retention import validate_allocation_ledger
    for path in checked["sidecars"]["cost_segments"]:
        segment = _strict_json(path)
        if segment.get("format") == "molgap-allocation-ledger-v1":
            validate_allocation_ledger(segment, spec)
    owner = __import__(adapter.module, fromlist=["validate_screen_resume"])
    hook = getattr(owner, "validate_screen_resume", None)
    if not callable(hook):
        raise ValueError(f"Family owner {adapter.module} has no validate_screen_resume hook")
    owner_report = hook(spec=spec, arm_id=arm_name, manifest=manifest,
                        bundle_root=root, artifacts=checked["artifacts"],
                        sidecars=checked["sidecars"], trajectory=checked["trajectory"],
                        context=context_value)
    if type(owner_report) is not dict:
        raise ValueError("Family resume validator must return a mapping")
    return {"status": "RESUME_VERIFIED", "manifest": manifest,
            "manifest_path": manifest_path, "root": root,
            "artifacts": checked["artifacts"], "sidecars": checked["sidecars"],
            "trajectory": checked["trajectory"], "owner": owner_report}


def restore_resume_bundle(path: Path | str, output: Path | str, spec=None, *,
                          arm_id: str | None = None, context: Any | None = None,
                          trajectory: Any | None = None) -> dict:
    """Restore a verified bundle into a fresh output namespace atomically."""
    checked = validate_resume_bundle(path, spec, arm_id=arm_id, context=context,
                                     trajectory=trajectory)
    destination = _path(output, "resume output")
    bundle_root = checked["root"]
    if destination == bundle_root or destination.resolve().is_relative_to(bundle_root.resolve()):
        raise ValueError("Restored output must be outside the resume bundle")
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise ValueError("Resume restore requires a fresh output directory")
    # ``checked.artifacts`` stores resolved source Paths; use the manifest
    # pointers for exact restore destinations rather than source basenames.
    copies: list[tuple[Path, Path, str]] = []
    for role, source in checked["artifacts"].items():
        if role == "trajectory" or source is None:
            continue
        pointer = checked["manifest"]["artifacts"][role]
        copies.append((source, destination / pointer["restore_path"], pointer["sha256"]))
    for group, sources in checked["sidecars"].items():
        entries = checked["manifest"][group]
        for source, pointer in zip(sources, entries):
            copies.append((source, destination / pointer["restore_path"], pointer["sha256"]))
    targets = set()
    for source, target, _ in copies:
        _safe_local(target.absolute())
        key = target.as_posix().casefold()
        if key in targets:
            raise ValueError("Resume restore destinations alias each other")
        targets.add(key)
        if target.exists():
            raise ValueError("Resume restore conflict; refusing overwrite")
    for source, target, digest in copies:
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name("." + target.name + ".resume.tmp")
        if temporary.exists():
            raise ValueError("Resume restore temporary conflict")
        try:
            shutil.copyfile(source, temporary)
            if sha256_file(temporary) != digest:
                raise ValueError("Resume bundle bytes changed during restoration")
            try:
                os.link(temporary, target)
            except FileExistsError as exc:
                raise ValueError("Concurrent resume restore conflict; refusing overwrite") from exc
        finally:
            temporary.unlink(missing_ok=True)
    return {"status": "RESTORED", "output": destination, "manifest": checked["manifest"],
            "owner": checked.get("owner", {})}


__all__ = ["RESUME_BUNDLE_FORMAT", "create_resume_bundle", "validate_resume_bundle",
           "restore_resume_bundle", "safe_cpu_torch_load"]
