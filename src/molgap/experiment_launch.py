"""Local immutable launch observations; no platform execution or scientific authority."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile
from datetime import datetime

from .experiment_package import verify_experiment_source_package
from .experiment_spec import ExperimentSpec
from .screen_policy import canonical_fingerprint

__all__ = ["build_launch_receipt", "reconcile_platform_response",
           "read_launch_receipt", "write_launch_receipt", "canonical_json"]

RECEIPT_FORMAT = "molgap-experiment-launch-receipt"
RESPONSE_FORMAT = "molgap-platform-response"
VERSION = 1
_OUTCOMES = {"accepted", "queued", "pending", "rejected_409",
             "rejected_permission", "client_unknown_after_submit", "existing_found", "unknown"}
_REASONS = {"not_reported", "not_observed", "not_submitted", "response_lost",
            "adapter_unimplemented"}
_LIMITATIONS = [
    "Local receipt/reconcile core only; SUBMIT_UNIMPLEMENTED.",
    "Envelope observations are caller supplied, not independently authenticated.",
    "No runtime qualification, scientific acceptance or execution authority.",
]


def canonical_json(value: dict) -> str:
    """Canonical wire encoding (no trailing newline)."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def _parse(raw: str) -> dict:
    if type(raw) is not str:
        raise TypeError("Expected canonical JSON string, not executable objects")
    value = json.loads(raw, object_pairs_hook=_unique)
    if type(value) is not dict or canonical_json(value) != raw:
        raise ValueError("Expected canonical JSON object")
    return value


def _fields(value, names):
    if type(value) is not dict or set(value) != set(names.split()):
        raise ValueError("Missing or unknown schema fields")


def _opaque(value):
    if type(value) is not str or not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_./:-]{0,511}", value):
        raise ValueError("Expected opaque identity/reference without credentials or query")


def _digest(value):
    if type(value) is not str or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("Expected lowercase SHA256")


def _unknown(reason="not_reported"):
    return {"value": None, "missing_reason": reason}


def _fact(value, validator):
    _fields(value, "value missing_reason")
    if value["value"] is None:
        if type(value["missing_reason"]) is not str or value["missing_reason"] not in _REASONS:
            raise ValueError("Unknown value requires explicit missing_reason")
    else:
        if value["missing_reason"] is not None:
            raise ValueError("Known value cannot have missing_reason")
        validator(value["value"])


def _timestamp(value):
    if type(value) is not str or not re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d{1,6})?\+00:00", value):
        raise ValueError("Timestamp must be explicit UTC ISO8601")
    datetime.fromisoformat(value)


def _relative(value):
    if type(value) is not str or not value:
        raise ValueError("Expected relative monitor path")
    for part in value.split("/"):
        if (not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}", part)
                or part.endswith(".") or part.split(".")[0].upper() in {
                    "CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(10)),
                    *(f"LPT{i}" for i in range(10))}):
            raise ValueError("Unsafe relative monitor path")


def _paths(value):
    if type(value) is not list or not value:
        raise ValueError("Known monitor paths must be nonempty")
    for item in value:
        _relative(item)
    if len({p.casefold() for p in value}) != len(value):
        raise ValueError("Duplicate monitor paths")


def _binding(spec, package_dir, expected_package_identity):
    if type(spec) is not ExperimentSpec:
        raise TypeError("Expected exactly ExperimentSpec")
    if set(vars(spec)) != {"_canonical_json"} or type(spec._canonical_json) is not str:
        raise ValueError("Invalid ExperimentSpec snapshot")
    rebuilt = ExperimentSpec.from_json(spec._canonical_json)
    if rebuilt.to_json() != spec.to_json() or rebuilt.identity != spec.identity:
        raise ValueError("Noncanonical ExperimentSpec")
    _digest(expected_package_identity)
    _safe_local(package_dir)
    manifest = verify_experiment_source_package(package_dir)
    declaration = rebuilt.to_dict()
    arms = [{"arm_id": arm["arm_id"], "arm_identity": canonical_fingerprint(arm)}
            for arm in declaration["arms"]]
    if (manifest["package_identity"] != expected_package_identity
            or manifest["spec_identity"] != rebuilt.identity
            or manifest["spec_sha256"] != hashlib.sha256(rebuilt.to_json().encode()).hexdigest()
            or manifest["experiment_id"] != declaration["experiment_id"]
            or manifest["logical_run_id"] != declaration["logical_run_id"]
            or manifest["arms"] != [{"arm_id": a["arm_id"], "identity": a["arm_identity"]} for a in arms]):
        raise ValueError("Pinned source package/spec/arm binding mismatch")
    return {
        "spec_identity": rebuilt.identity, "spec_sha256": manifest["spec_sha256"],
        "experiment_id": declaration["experiment_id"],
        "logical_run_id": declaration["logical_run_id"],
        "package_identity": manifest["package_identity"],
        "source_commit": manifest["source_commit"],
        "source_archive_sha256": manifest["archive_sha256"], "arms": arms,
        "requested_logical_platform": declaration["platform"]["name"],
    }


def _validate_binding(value):
    _fields(value, "spec_identity spec_sha256 experiment_id logical_run_id package_identity source_commit source_archive_sha256 arms requested_logical_platform")
    for key in ("spec_identity", "spec_sha256", "package_identity", "source_archive_sha256"):
        _digest(value[key])
    for key in ("experiment_id", "logical_run_id"):
        _opaque(value[key])
    if type(value["source_commit"]) is not str or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", value["source_commit"]):
        raise ValueError("Invalid source commit")
    if type(value["requested_logical_platform"]) is not str or value["requested_logical_platform"] not in {"local", "kaggle", "scnet", "ims"}:
        raise ValueError("Invalid logical platform")
    if type(value["arms"]) is not list or not value["arms"]:
        raise ValueError("Expected ordered arms")
    seen = set()
    for arm in value["arms"]:
        _fields(arm, "arm_id arm_identity")
        _opaque(arm["arm_id"])
        _digest(arm["arm_identity"])
        if arm["arm_id"] in seen:
            raise ValueError("Duplicate arm")
        seen.add(arm["arm_id"])


def _runs(value, arms):
    if type(value) is not list or not value:
        raise ValueError("Known physical mapping must be nonempty")
    seen_runs, seen_arms = set(), set()
    for run in value:
        _fields(run, "run_identity canonical_reference platform_version arm_ids")
        _opaque(run["run_identity"])
        _opaque(run["canonical_reference"])
        _fact(run["platform_version"], _opaque)
        key = (run["canonical_reference"], run["run_identity"])
        if key in seen_runs:
            raise ValueError("Duplicate physical run")
        seen_runs.add(key)
        ids = run["arm_ids"]
        if type(ids) is not list or not ids or any(type(a) is not str for a in ids):
            raise ValueError("Expected explicit arm mapping")
        if len(set(ids)) != len(ids) or set(ids) & seen_arms or not set(ids) <= set(arms):
            raise ValueError("Duplicate or unknown physical arm mapping")
        if ids != [a for a in arms if a in ids]:
            raise ValueError("Physical arm mapping must preserve spec order")
        seen_arms.update(ids)
    if seen_arms != set(arms):
        raise ValueError("Partial physical mapping must remain unknown")


def _response(raw):
    value = _parse(raw)
    _fields(value, "format version mode outcome conflict_kind binding canonical_platform_reference platform_version physical_runs timestamp monitor_paths")
    if value["format"] != RESPONSE_FORMAT or type(value["version"]) is not int or value["version"] != VERSION:
        raise ValueError("Unsupported response format/version")
    if type(value["mode"]) is not str or value["mode"] not in {"observed", "dry_run"}:
        raise ValueError("Unknown response mode")
    if type(value["outcome"]) is not str or value["outcome"] not in _OUTCOMES:
        raise ValueError("Unknown response outcome")
    if value["outcome"] == "rejected_409":
        if type(value["conflict_kind"]) is not str or value["conflict_kind"] not in {"name_conflict", "existing_version", "permission", "other"}:
            raise ValueError("409 requires explicit classification")
    elif value["conflict_kind"] is not None:
        raise ValueError("conflict_kind only applies to 409")
    _validate_binding(value["binding"])
    _fact(value["canonical_platform_reference"], _opaque)
    _fact(value["platform_version"], _opaque)
    _fact(value["physical_runs"], lambda runs: _runs(runs, [a["arm_id"] for a in value["binding"]["arms"]]))
    _fact(value["timestamp"], _timestamp)
    _fact(value["monitor_paths"], _paths)
    if value["outcome"] == "existing_found" and value["canonical_platform_reference"]["value"] is None:
        raise ValueError("existing_found requires an observed canonical reference")
    return value


def build_launch_receipt(spec: ExperimentSpec, package_dir: Path, *,
                         expected_package_identity: str, response_json: str | None = None) -> dict:
    """Verify a pinned package and deterministically describe a local observation.

    No response means NOT_SUBMITTED. Supplied observed responses are assertions
    by the caller, not proof that this library contacted a platform.
    """
    binding = _binding(spec, package_dir, expected_package_identity)
    response = None if response_json is None else _response(response_json)
    state, result, action = "NOT_SUBMITTED", "SUBMIT_UNIMPLEMENTED", "IMPLEMENT_VERIFIED_ADAPTER"
    if response is not None:
        outcome = response["outcome"]
        if response["binding"] != binding:
            state, result, action = "REJECTED", "REJECTED_CONFLICT", "RESOLVE_IDENTITY_CONFLICT"
        elif response["mode"] == "dry_run":
            result, action = "DRY_RUN_ONLY", "IMPLEMENT_VERIFIED_ADAPTER"
        elif outcome == "accepted":
            state, result, action = "ACCEPTED", "ACCEPTED", "INSPECT_DURABLE_PLATFORM_STATE"
        elif outcome in {"queued", "pending", "client_unknown_after_submit"}:
            state, result, action = "PENDING_RECONCILIATION", outcome.upper(), "RECONCILE_BEFORE_ANY_RETRY"
        elif outcome == "existing_found":
            state, result, action = "RECONCILED_EXISTING", "RECONCILED_EXISTING", "INSPECT_DURABLE_PLATFORM_STATE"
        elif outcome == "unknown":
            state, result, action = "UNKNOWN", "UNKNOWN", "RECONCILE_BEFORE_ANY_RETRY"
        elif outcome == "rejected_409":
            state, result, action = "REJECTED", "REJECTED_409_" + response["conflict_kind"].upper(), "RESOLVE_WITHOUT_AUTOMATIC_RETRY"
        else:
            state, result, action = "REJECTED", "REJECTED_PERMISSION", "RESOLVE_PERMISSION_WITHOUT_RETRY"
    # Dry-run/conflicting observations never become effective platform identity.
    effective = response if response and response["mode"] == "observed" and response["binding"] == binding else None
    receipt = {
        "format": RECEIPT_FORMAT, "version": VERSION, "binding": binding,
        "launch_identity": canonical_fingerprint({key: binding[key] for key in (
            "experiment_id", "logical_run_id", "requested_logical_platform")}),
        "submission_state": state, "reconciliation_result": result,
        "submitter_status": "SUBMIT_UNIMPLEMENTED", "next_action": action,
        "response": response, "cost": _unknown("not_observed"),
        "limitations": list(_LIMITATIONS),
    }
    for key in ("canonical_platform_reference", "platform_version", "physical_runs", "timestamp", "monitor_paths"):
        receipt[key] = effective[key] if effective else _unknown("not_submitted" if state == "NOT_SUBMITTED" else "not_observed")
    receipt["receipt_identity"] = canonical_fingerprint(receipt)
    return json.loads(canonical_json(receipt))


def reconcile_platform_response(spec: ExperimentSpec, package_dir: Path, response_json: str,
                                *, expected_package_identity: str) -> dict:
    """Reconcile one explicit canonical observation locally, never retry submission."""
    if response_json is None:
        raise TypeError("Reconciliation requires an explicit response")
    return build_launch_receipt(spec, package_dir, expected_package_identity=expected_package_identity,
                                response_json=response_json)


def _validated_receipt(raw, spec, package_dir, expected_package_identity):
    value = _parse(raw)
    if "response" not in value:
        raise ValueError("Missing receipt response")
    rebuilt = build_launch_receipt(spec, package_dir, expected_package_identity=expected_package_identity,
                                  response_json=None if value["response"] is None else canonical_json(value["response"]))
    if canonical_json(rebuilt) != raw:
        raise ValueError("Receipt schema, identity or derived state mismatch")
    return rebuilt


def _safe_local(path):
    if type(path) is not type(Path()) or not path.is_absolute() or ".." in path.parts:
        raise ValueError("Expected absolute concrete local Path without traversal")
    if str(path).startswith(("\\\\", "//")):
        raise ValueError("Network paths are forbidden")
    for part in (path, *path.parents):
        if part != Path(part.anchor) and (
                ":" in part.name or part.name.endswith((".", " "))
                or any(ord(c) < 32 for c in part.name)
                or part.name.split(".")[0].upper() in {
                    "CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(10)),
                    *(f"LPT{i}" for i in range(10))}):
            raise ValueError("Unsafe local path component")
        try:
            info = part.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError("Symlink/junction/reparse paths are forbidden")


def read_launch_receipt(path: Path, spec: ExperimentSpec, package_dir: Path, *, expected_package_identity: str) -> dict:
    """Read canonical bytes and reverify package, schema, binding and derived state."""
    _safe_local(path)
    if not stat.S_ISREG(path.stat().st_mode):
        raise ValueError("Receipt must be a regular file")
    return _validated_receipt(path.read_bytes().decode("utf-8"), spec, package_dir, expected_package_identity)


def write_launch_receipt(receipt_json: str, output_dir: Path, spec: ExperimentSpec,
                         package_dir: Path, *, expected_package_identity: str) -> Path:
    """Atomically create <launch_identity>.json; identical retry is a no-op.

    A changed observation requires a separate snapshot directory. A conflicting
    retry in the same directory always fails; even an existing empty file is not
    overwritten. Hard-link publication is atomic and cannot replace a winner.
    """
    receipt = _validated_receipt(receipt_json, spec, package_dir, expected_package_identity)
    _safe_local(output_dir)
    package = Path(package_dir).resolve()
    if (not output_dir.is_dir() or output_dir == Path(output_dir.anchor)
            or output_dir.resolve().is_relative_to(package)
            or any(p.casefold() in {"research_memory", "ready_for_desktop"} for p in output_dir.parts)):
        raise ValueError("Expected existing dedicated receipt directory outside package/evidence indexes")
    path = output_dir / (receipt["launch_identity"] + ".json")
    _safe_local(path)
    payload = receipt_json.encode("utf-8")

    def identical():
        _safe_local(path)
        if not stat.S_ISREG(path.stat().st_mode) or path.read_bytes() != payload:
            raise ValueError("Receipt retry conflict; refusing overwrite")

    if path.exists():
        identical()
        return path
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=output_dir, prefix=".launch-", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        _safe_local(output_dir)
        try:
            os.link(temporary, path)
        except FileExistsError:
            identical()
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path
