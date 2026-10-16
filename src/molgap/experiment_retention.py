"""Compact retention binding for platform execution observations.

Family output manifests describe scientific artifacts.  This module owns the
separate, platform-facing retention record for allocation ledgers and the
small execution report written by the orchestration runtime.  It deliberately
does not discover or attach arbitrary files from an output directory.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta
import hashlib
import json
import math
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
from typing import Any

from .experiment_launch import _safe_local, publish_immutable_bytes
from .experiment_spec import ExperimentSpec


RETENTION_FORMAT = "molgap-execution-retention-v1"
RETENTION_FILENAME = "execution_retention.json"
LEDGER_FORMAT = "molgap-allocation-ledger-v1"
LEDGER_FILENAME = "allocation_ledger.json"
EXECUTION_REPORT_FILENAME = "execution_report.json"

_RETENTION_FIELDS = {
    "format", "status", "spec_identity", "package_identity", "source_commit",
    "source_archive_sha256", "allocation_ledger", "execution_report",
}
_ALLOCATION_FIELDS = {"root", "arms"}
_FILE_REF_FIELDS = {"path", "sha256"}
_LEDGER_FIELDS = {
    "format", "spec_identity", "started_at", "observed_at", "status",
    "allocation_device_count", "wall_seconds", "allocated_device_seconds",
    "unassigned_device_seconds", "devices", "scope",
    "provisioning_before_python_seconds", "queue_seconds", "prior_segments",
    "prior_unobserved_intervals",
}
_DEVICE_FIELDS = {"device", "hardware", "arm_id", "allocated_seconds"}
_MEASUREMENT_FIELDS = {"value", "status"}
_LEDGER_STATUSES = {"running", "complete", "failed"}
_MEASUREMENT_MISSING = {"value": None, "status": "measurement_missing"}
_MAX_PRIOR_DEPTH = 8
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(value: Any, label: str) -> str:
    if type(value) is not str or _HEX64.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA256")
    return value


def _source_commit(value: Any) -> str:
    if type(value) is not str or _COMMIT.fullmatch(value) is None:
        raise ValueError("source_commit must be a full lowercase Git identity")
    return value


def _finite(value: Any, label: str, *, nonnegative: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be finite numeric")
    if nonnegative and value < 0:
        raise ValueError(f"{label} must be non-negative")
    return float(value)


def _relative(value: Any, label: str) -> str:
    if (type(value) is not str or not value or "\\" in value or ":" in value
            or value.startswith("/") or PurePosixPath(value).is_absolute()
            or PureWindowsPath(value).drive):
        raise ValueError(f"{label} must be a relative POSIX path")
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"{label} must be a normalized relative POSIX path")
    if any(part.endswith((".", " ")) for part in parts):
        raise ValueError(f"{label} contains an unsafe path component")
    return value


def _load_json(path: Path, label: str) -> dict:
    raw = path.read_bytes()

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} contains duplicate JSON keys")
            result[key] = value
        return result

    try:
        value = json.loads(raw, object_pairs_hook=unique,
                           parse_constant=lambda _: (_ for _ in ()).throw(
                               ValueError(f"{label} contains nonfinite JSON")))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid JSON") from exc
    if type(value) is not dict:
        raise ValueError(f"{label} must be a JSON object")
    return value


def _regular(path: Path, label: str) -> Path:
    path = Path(path).absolute()
    _safe_local(path)
    if not path.is_file():
        raise ValueError(f"Missing {label}: {path.name}")
    return path


def _path_under(root: Path, path: Path, label: str) -> str:
    root = Path(root).absolute().resolve()
    path = Path(path).absolute()
    try:
        relative = path.resolve().relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} is outside the execution output root") from exc
    return _relative(relative.as_posix(), label)


def _spec_arms(spec: ExperimentSpec) -> tuple[str, ...]:
    if type(spec) is not ExperimentSpec:
        raise TypeError("Expected exactly ExperimentSpec")
    arms = spec.to_dict().get("arms")
    if type(arms) is not list or not arms:
        raise ValueError("Execution retention requires Spec arms")
    ids = tuple(arm.get("arm_id") for arm in arms)
    if any(type(value) is not str or not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("Execution retention requires unique Spec arm identities")
    return ids


def _platform_allocation(spec: ExperimentSpec, arm_ids: Sequence[str]) -> tuple[int, str]:
    platform = spec.to_dict().get("platform")
    if type(platform) is not dict:
        raise ValueError("Execution retention requires a Spec platform declaration")
    count, accelerator = platform.get("device_count"), platform.get("accelerator")
    if type(count) is not int or count <= 0 or type(accelerator) is not str or not accelerator:
        raise ValueError("Execution retention requires a valid Spec platform allocation")
    if count < len(arm_ids):
        raise ValueError("Spec platform device_count cannot cover every execution arm")
    return count, accelerator


def _validate_measurement(value: Any, label: str) -> None:
    if type(value) is not dict or set(value) != _MEASUREMENT_FIELDS:
        raise ValueError(f"{label} must contain exactly value and status")
    if value != _MEASUREMENT_MISSING:
        raise ValueError(f"{label} must retain the unobserved measurement explicitly")


def _validate_prior_segment(value: Any, spec_identity: str,
                            arm_ids: Sequence[str], label: str, depth: int) -> bool:
    """Validate a retained prior segment and return whether its interval is unknown."""
    if depth > _MAX_PRIOR_DEPTH:
        raise ValueError(f"{label} exceeds the bounded prior segment depth")
    if type(value) is not dict or value.get("format") != LEDGER_FORMAT:
        raise ValueError(f"{label} is not an allocation ledger segment")
    if value.get("spec_identity") != spec_identity:
        raise ValueError(f"{label} crosses the execution Spec identity")
    if value.get("status") not in _LEDGER_STATUSES:
        raise ValueError(f"{label} has an invalid status")
    if set(value) == {"format", "spec_identity", "status"}:
        # A compact historical marker carries no physical measurements. Keep
        # it, but force the parent ledger to retain the unknown interval bit.
        return True
    if set(value) != _LEDGER_FIELDS:
        raise ValueError(f"{label} has unknown or missing fields")
    if depth == _MAX_PRIOR_DEPTH:
        raise ValueError(f"{label} exceeds the bounded prior segment depth")
    # Historical segments may have had a different physical allocation, so
    # only their ledger self-consistency and recursive identity are checked.
    _validate_ledger(value, spec_identity, arm_ids, label, prior_depth=depth)
    return bool(value.get("status") == "running" or value.get("prior_unobserved_intervals"))


def _timestamp(value: Any, label: str) -> datetime:
    if type(value) is not str:
        raise ValueError(f"{label} must be an ISO timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{label} must be an ISO timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ValueError(f"{label} must be an explicit UTC timestamp")
    return parsed


def _hardware_matches(observed: str, expected: str) -> bool:
    observed_key = re.sub(r"[^a-z0-9]", "", observed.casefold())
    expected_key = re.sub(r"[^a-z0-9]", "", expected.casefold())
    return expected_key in observed_key or observed_key in expected_key


def _validate_ledger(
    ledger: dict,
    spec_identity: str,
    arm_ids: Sequence[str],
    label: str,
    *,
    prior_depth: int = 0,
    expected_device_count: int | None = None,
    expected_accelerator: str | None = None,
    require_all_arms: bool = False,
) -> dict:
    if type(ledger) is not dict or set(ledger) != _LEDGER_FIELDS:
        raise ValueError(f"{label} has unknown or missing fields")
    if ledger["format"] != LEDGER_FORMAT or ledger["spec_identity"] != spec_identity:
        raise ValueError(f"{label} format or Spec identity mismatch")
    if ledger["status"] not in _LEDGER_STATUSES:
        raise ValueError(f"{label} has an invalid status")
    started_at = _timestamp(ledger["started_at"], f"{label}.started_at")
    observed_at = _timestamp(ledger["observed_at"], f"{label}.observed_at")
    if observed_at < started_at:
        raise ValueError(f"{label}.observed_at precedes started_at")
    count = ledger["allocation_device_count"]
    if type(count) is not int or count <= 0:
        raise ValueError(f"{label}.allocation_device_count must be positive")
    if expected_device_count is not None and count != expected_device_count:
        raise ValueError(f"{label}.allocation_device_count differs from Spec platform.device_count")
    wall = _finite(ledger["wall_seconds"], f"{label}.wall_seconds")
    allocated = _finite(ledger["allocated_device_seconds"], f"{label}.allocated_device_seconds")
    unassigned = _finite(ledger["unassigned_device_seconds"], f"{label}.unassigned_device_seconds")
    devices = ledger["devices"]
    if type(devices) is not list or len(devices) != count:
        raise ValueError(f"{label}.devices must cover the declared allocation")
    seen_devices, seen_arms = set(), set()
    arm_set = set(arm_ids)
    for index, device in enumerate(devices):
        if type(device) is not dict or set(device) != _DEVICE_FIELDS:
            raise ValueError(f"{label}.devices[{index}] has unknown or missing fields")
        number = device["device"]
        if type(number) is not int or number < 0 or number >= count or number in seen_devices:
            raise ValueError(f"{label}.devices has an invalid or duplicate device")
        seen_devices.add(number)
        if type(device["hardware"]) is not str or not device["hardware"]:
            raise ValueError(f"{label}.devices[{index}].hardware is missing")
        if (expected_accelerator is not None
                and not _hardware_matches(device["hardware"], expected_accelerator)):
            raise ValueError(f"{label}.devices[{index}].hardware differs from Spec platform accelerator")
        arm_id = device["arm_id"]
        if arm_id is not None:
            if type(arm_id) is not str or arm_id not in arm_set or arm_id in seen_arms:
                raise ValueError(f"{label}.devices has an invalid or duplicate arm assignment")
            seen_arms.add(arm_id)
        seconds = _finite(device["allocated_seconds"], f"{label}.devices[{index}].allocated_seconds")
        if not math.isclose(seconds, wall, rel_tol=1e-9, abs_tol=1e-6):
            raise ValueError(f"{label}.devices[{index}] disagrees with wall_seconds")
    if seen_devices != set(range(count)):
        raise ValueError(f"{label}.devices must enumerate every allocated device")
    if require_all_arms and seen_arms != arm_set:
        raise ValueError(f"{label}.devices must assign every Spec arm exactly once")
    if not math.isclose(allocated, count * wall, rel_tol=1e-9, abs_tol=1e-6):
        raise ValueError(f"{label}.allocated_device_seconds disagrees with allocation")
    idle = count - len(seen_arms)
    if not math.isclose(unassigned, idle * wall, rel_tol=1e-9, abs_tol=1e-6):
        raise ValueError(f"{label}.unassigned_device_seconds disagrees with idle allocation")
    if ledger["scope"] != "Python bootstrap/runtime observation window; includes idle allocated devices":
        raise ValueError(f"{label}.scope is not the reviewed allocation scope")
    _validate_measurement(ledger["provisioning_before_python_seconds"],
                          f"{label}.provisioning_before_python_seconds")
    _validate_measurement(ledger["queue_seconds"], f"{label}.queue_seconds")
    prior = ledger["prior_segments"]
    if type(prior) is not list:
        raise ValueError(f"{label}.prior_segments must be an array")
    unknown_prior = any(_validate_prior_segment(item, spec_identity, arm_ids,
                                                f"{label}.prior_segments[{i}]",
                                                prior_depth + 1)
                        for i, item in enumerate(prior))
    if type(ledger["prior_unobserved_intervals"]) is not bool:
        raise ValueError(f"{label}.prior_unobserved_intervals must be boolean")
    if unknown_prior and ledger["prior_unobserved_intervals"] is not True:
        raise ValueError(f"{label} cannot claim prior physical intervals were observed")
    return ledger


def _arm_roots(output_root: Path, arm_ids: Sequence[str], arm_roots: Mapping | Sequence) -> dict[str, Path]:
    if isinstance(arm_roots, Mapping):
        if set(arm_roots) != set(arm_ids):
            raise ValueError("arm_roots must bind every Spec arm exactly once")
        values = {arm_id: Path(arm_roots[arm_id]) for arm_id in arm_ids}
    elif isinstance(arm_roots, Sequence) and not isinstance(arm_roots, (str, bytes)):
        values = {}
        for path_value in arm_roots:
            path = Path(path_value)
            arm_id = path.name
            if arm_id in values or arm_id not in arm_ids:
                raise ValueError("arm_roots contains an unknown or duplicate arm root")
            values[arm_id] = path
        if set(values) != set(arm_ids):
            raise ValueError("arm_roots must bind every Spec arm exactly once")
    else:
        raise ValueError("arm_roots must be a mapping or sequence of paths")
    root = Path(output_root).absolute().resolve()
    for arm_id, path in values.items():
        path = Path(path)
        if not path.is_absolute():
            path = root / path
        path = path.absolute()
        _safe_local(path)
        if path.resolve().parent != root:
            raise ValueError(f"arm root for {arm_id} must be directly under output_root")
        values[arm_id] = path
    return values


def _file_ref(root: Path, path: Path, label: str) -> dict[str, str]:
    path = _regular(path, label)
    return {"path": _path_under(root, path, label), "sha256": _file_sha(path)}


def _manifest_bytes(value: dict) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")


def _validate_manifest_shape(manifest: dict, spec: ExperimentSpec, *,
                            package_identity: str, source_commit: str,
                            source_archive_sha256: str) -> tuple[str, ...]:
    arm_ids = _spec_arms(spec)
    if type(manifest) is not dict or set(manifest) != _RETENTION_FIELDS:
        raise ValueError("Execution retention manifest has unknown or missing fields")
    if (manifest["format"] != RETENTION_FORMAT or manifest["status"] != "SEALED"
            or manifest["spec_identity"] != spec.identity
            or manifest["package_identity"] != package_identity
            or manifest["source_commit"] != source_commit
            or manifest["source_archive_sha256"] != source_archive_sha256):
        raise ValueError("Execution retention identity mismatch")
    allocation = manifest["allocation_ledger"]
    if type(allocation) is not dict or set(allocation) != _ALLOCATION_FIELDS:
        raise ValueError("Execution retention allocation binding is incomplete")
    root_ref = allocation["root"]
    if type(root_ref) is not dict or set(root_ref) != _FILE_REF_FIELDS or root_ref["path"] != LEDGER_FILENAME:
        raise ValueError("Execution retention root ledger binding is invalid")
    _digest(root_ref["sha256"], "allocation_ledger.root.sha256")
    arms = allocation["arms"]
    if type(arms) is not dict or set(arms) != set(arm_ids):
        raise ValueError("Execution retention arm ledger bindings are incomplete")
    for arm_id in arm_ids:
        item = arms[arm_id]
        expected_path = f"{arm_id}/{LEDGER_FILENAME}"
        if type(item) is not dict or set(item) != _FILE_REF_FIELDS or item["path"] != expected_path:
            raise ValueError(f"Execution retention ledger binding is invalid for {arm_id}")
        _digest(item["sha256"], f"allocation_ledger.arms[{arm_id}].sha256")
    report = manifest["execution_report"]
    if report is not None:
        if type(report) is not dict or set(report) != _FILE_REF_FIELDS or report["path"] != EXECUTION_REPORT_FILENAME:
            raise ValueError("Execution retention report binding is invalid")
        _digest(report["sha256"], "execution_report.sha256")
    return arm_ids


def validate_execution_retention_manifest(
    manifest: dict,
    spec: ExperimentSpec,
    *,
    package_identity: str,
    source_commit: str,
    source_archive_sha256: str,
) -> dict:
    """Validate the pinned retention envelope before retrieving its files.

    This shape-only gate checks the exact identity and fixed allowlist paths;
    :func:`validate_execution_retention` remains the authority for the bytes
    and ledger contents after retrieval.
    """
    package_identity = _digest(package_identity, "package_identity")
    source_commit = _source_commit(source_commit)
    source_archive_sha256 = _digest(source_archive_sha256, "source_archive_sha256")
    _validate_manifest_shape(
        manifest,
        spec,
        package_identity=package_identity,
        source_commit=source_commit,
        source_archive_sha256=source_archive_sha256,
    )
    return manifest


def _validate_files(output_root: Path, manifest: dict, spec: ExperimentSpec,
                    *, package_identity: str, source_commit: str,
                    source_archive_sha256: str) -> dict:
    arm_ids = _validate_manifest_shape(manifest, spec,
        package_identity=package_identity, source_commit=source_commit,
        source_archive_sha256=source_archive_sha256)
    expected_device_count, expected_accelerator = _platform_allocation(spec, arm_ids)
    root = Path(output_root).absolute()
    root_ledger_path = root / LEDGER_FILENAME
    root_ref = manifest["allocation_ledger"]["root"]
    root_ledger_path = _regular(root_ledger_path, "root allocation ledger")
    if _file_sha(root_ledger_path) != root_ref["sha256"]:
        raise ValueError("Root allocation ledger changed after sealing")
    root_ledger = _validate_ledger(_load_json(root_ledger_path, "root allocation ledger"),
                                   spec.identity, arm_ids, "root allocation ledger",
                                   expected_device_count=expected_device_count,
                                   expected_accelerator=expected_accelerator,
                                   require_all_arms=True)
    arm_ledgers = {}
    for arm_id in arm_ids:
        path = root / arm_id / LEDGER_FILENAME
        path = _regular(path, f"{arm_id} allocation ledger")
        ref = manifest["allocation_ledger"]["arms"][arm_id]
        if _file_sha(path) != ref["sha256"]:
            raise ValueError(f"{arm_id} allocation ledger changed after sealing")
        ledger = _validate_ledger(_load_json(path, f"{arm_id} allocation ledger"),
                                  spec.identity, arm_ids, f"{arm_id} allocation ledger")
        if ledger != root_ledger:
            raise ValueError(f"{arm_id} allocation ledger differs from root allocation")
        arm_ledgers[arm_id] = ledger
    report = manifest["execution_report"]
    report_value = None
    if report is not None:
        report_path = _regular(root / EXECUTION_REPORT_FILENAME, "execution report")
        if _file_sha(report_path) != report["sha256"]:
            raise ValueError("Execution report changed after sealing")
        report_value = _load_json(report_path, "execution report")
    return {"root": root_ledger, "arms": arm_ledgers, "execution_report": report_value}


def seal_execution_retention(
    output_root: Path,
    spec: ExperimentSpec,
    *,
    package_identity: str,
    source_commit: str,
    source_archive_sha256: str,
    arm_roots: Mapping | Sequence,
) -> dict:
    """Bind the compact allocation ledgers for one execution output.

    The source/package identities are release facts supplied by the caller;
    this function only seals files already retained by the runtime.  It never
    searches the output tree for additional artifacts.
    """
    arm_ids = _spec_arms(spec)
    expected_device_count, expected_accelerator = _platform_allocation(spec, arm_ids)
    package_identity = _digest(package_identity, "package_identity")
    source_archive_sha256 = _digest(source_archive_sha256, "source_archive_sha256")
    source_commit = _source_commit(source_commit)
    root = Path(output_root).absolute()
    _safe_local(root)
    if not root.is_dir():
        raise ValueError("Execution output root is missing")
    roots = _arm_roots(root, arm_ids, arm_roots)
    root_ledger = _validate_ledger(_load_json(_regular(root / LEDGER_FILENAME, "root allocation ledger"),
                                              "root allocation ledger"),
                                   spec.identity, arm_ids, "root allocation ledger",
                                   expected_device_count=expected_device_count,
                                   expected_accelerator=expected_accelerator,
                                   require_all_arms=True)
    refs = {"root": _file_ref(root, root / LEDGER_FILENAME, "root allocation ledger"), "arms": {}}
    for arm_id in arm_ids:
        path = roots[arm_id] / LEDGER_FILENAME
        ledger = _validate_ledger(_load_json(_regular(path, f"{arm_id} allocation ledger"),
                                             f"{arm_id} allocation ledger"),
                                  spec.identity, arm_ids, f"{arm_id} allocation ledger")
        if ledger != root_ledger:
            raise ValueError(f"{arm_id} allocation ledger differs from root allocation")
        ref = _file_ref(root, path, f"{arm_id} allocation ledger")
        if ref["path"] != f"{arm_id}/{LEDGER_FILENAME}":
            raise ValueError(f"{arm_id} allocation ledger path is not directly retained")
        refs["arms"][arm_id] = ref
    report_ref = None
    report_path = root / EXECUTION_REPORT_FILENAME
    if report_path.exists():
        _load_json(_regular(report_path, "execution report"), "execution report")
        report_ref = _file_ref(root, report_path, "execution report")
    manifest = {
        "format": RETENTION_FORMAT, "status": "SEALED", "spec_identity": spec.identity,
        "package_identity": package_identity, "source_commit": source_commit,
        "source_archive_sha256": source_archive_sha256,
        "allocation_ledger": refs, "execution_report": report_ref,
    }
    path = root / RETENTION_FILENAME
    publish_immutable_bytes(path, _manifest_bytes(manifest))
    checked = _validate_files(root, manifest, spec,
        package_identity=package_identity, source_commit=source_commit,
        source_archive_sha256=source_archive_sha256)
    return {"status": "EXECUTION_RETENTION_SEALED", "manifest": manifest,
            "manifest_path": path, "ledgers": checked}


def validate_execution_retention(
    output_root: Path,
    spec: ExperimentSpec,
    *,
    package_identity: str,
    source_commit: str,
    source_archive_sha256: str,
) -> dict:
    """Verify the sealed allowlist and every retained allocation ledger."""
    package_identity = _digest(package_identity, "package_identity")
    source_archive_sha256 = _digest(source_archive_sha256, "source_archive_sha256")
    source_commit = _source_commit(source_commit)
    root = Path(output_root).absolute()
    _safe_local(root)
    if not root.is_dir():
        raise ValueError("Execution output root is missing")
    path = _regular(root / RETENTION_FILENAME, "execution retention manifest")
    raw = path.read_bytes()
    manifest = _load_json(path, "execution retention manifest")
    if _manifest_bytes(manifest) != raw:
        raise ValueError("Execution retention manifest must use canonical JSON bytes")
    checked = _validate_files(root, manifest, spec,
        package_identity=package_identity, source_commit=source_commit,
        source_archive_sha256=source_archive_sha256)
    return {"status": "EXECUTION_RETENTION_VERIFIED", "manifest": manifest,
            "manifest_path": path, "ledgers": checked}


def validate_allocation_ledger(ledger: dict, spec: ExperimentSpec) -> dict:
    """Check a retained allocation segment before it enters a recovery launch."""
    arms = _spec_arms(spec)
    count, accelerator = _platform_allocation(spec, arms)
    return _validate_ledger(ledger, spec.identity, arms, "retained allocation ledger",
        expected_device_count=count, expected_accelerator=accelerator, require_all_arms=True)


__all__ = ["LEDGER_FILENAME", "RETENTION_FILENAME", "RETENTION_FORMAT",
           "seal_execution_retention", "validate_execution_retention",
           "validate_execution_retention_manifest", "validate_allocation_ledger"]
