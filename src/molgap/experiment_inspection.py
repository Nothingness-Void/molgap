"""Immutable, digest-bound output inspection snapshots.

The family output inspector reads retained tensors and checkpoints once.  The
resulting snapshot carries only canonical metadata and byte digests, so later
descriptor construction and terminal closure can re-check the files without
deserializing the tensors a second time.

Snapshots are deliberately created by :mod:`experiment_family_workflow`'s
owner API.  A caller cannot mark an arbitrary report as accepted; every use
revalidates the snapshot's context, expectations, manifest and artifact bytes.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


SNAPSHOT_FORMAT = "molgap-family-inspection-snapshot-v1"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_TOKEN = object()


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)


def _digest_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _parse_canonical(payload: str, *, label: str) -> dict:
    if type(payload) is not str:
        raise ValueError(f"Invalid inspection snapshot {label}")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key in inspection snapshot {label}: {key}")
            result[key] = value
        return result

    value = json.loads(
        payload,
        object_pairs_hook=unique,
        parse_constant=lambda _: (_ for _ in ()).throw(
            ValueError(f"Nonfinite JSON in inspection snapshot {label}")),
    )
    if type(value) is not dict or _canonical(value) != payload:
        raise ValueError(f"Inspection snapshot {label} is not canonical JSON")
    return value


def _safe_digest(value, label: str) -> None:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(f"Invalid {label} SHA256")


def _concrete_path(value, label: str) -> Path:
    if type(value) is not type(Path()) or not value.is_absolute() or ".." in value.parts:
        raise ValueError(f"Invalid inspection snapshot {label} path")
    return value


def _path_key(value: Path) -> str:
    # ``absolute`` preserves the caller's path spelling while avoiding a
    # second filesystem lookup.  Windows paths are case-insensitive.
    text = str(value.absolute())
    return text.casefold() if value.drive else text


class InspectionSnapshot:
    """Read-only metadata returned by the owner output-inspection API.

    Construction is intentionally private.  The snapshot stores canonical
    JSON rather than mutable dictionaries and seals its digest on creation;
    callers receive fresh dictionaries from the accessors.
    """

    __slots__ = ("_payload", "_digest", "_sealed")

    def __init__(self, *, _owner_token=None, **_ignored):
        if _owner_token is not _TOKEN:
            raise TypeError("InspectionSnapshot must be produced by inspect_output_snapshot")
        payload = _ignored
        encoded = _canonical(payload)
        object.__setattr__(self, "_payload", encoded)
        object.__setattr__(self, "_digest", _digest_bytes(encoded.encode("utf-8")))
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name, value):
        if getattr(self, "_sealed", False):
            raise AttributeError("InspectionSnapshot is immutable")
        object.__setattr__(self, name, value)

    @classmethod
    def _from_owner(cls, *, output_dir: Path, manifest_sha256: str,
                    context: dict, expected: dict, report: dict,
                    manifest_context: dict,
                    artifact_bindings: dict):
        """Create a snapshot after the real inspector has accepted an output."""
        _safe_digest(manifest_sha256, "output manifest")
        root = _concrete_path(output_dir, "output directory")
        if (type(context) is not dict or type(expected) is not dict
                or type(report) is not dict or type(manifest_context) is not dict):
            raise TypeError("Inspection snapshot metadata must be dictionaries")
        if report.get("status") != "MECHANICALLY_VERIFIED" or report.get("blockers") != []:
            raise ValueError("Only a mechanically verified output can produce an inspection snapshot")
        if report.get("context") != context:
            raise ValueError("Inspection report/context mismatch")
        if type(artifact_bindings) is not dict or not artifact_bindings:
            raise ValueError("Inspection snapshot requires produced artifact bindings")
        artifacts = []
        for role in sorted(artifact_bindings):
            binding = artifact_bindings[role]
            if type(binding) is not dict or set(binding) != {"path", "sha256"}:
                raise ValueError("Invalid produced artifact binding")
            path, digest = binding["path"], binding["sha256"]
            if type(path) is not str or not path or "\\" in path or path.startswith("/") or ":" in path:
                raise ValueError("Invalid produced artifact path")
            if any(part in {"", ".", ".."} for part in path.split("/")):
                raise ValueError("Invalid produced artifact path")
            _safe_digest(digest, f"artifact {role}")
            artifacts.append({"role": role, "path": path, "sha256": digest})
        payload = {
            "format": SNAPSHOT_FORMAT,
            "output_dir": str(root),
            "manifest_sha256": manifest_sha256,
            "manifest_context": manifest_context,
            "context": context,
            "expected": expected,
            "artifacts": artifacts,
            "report": report,
        }
        return cls(_owner_token=_TOKEN, **payload)

    @property
    def digest(self) -> str:
        return self._digest

    @property
    def output_dir(self) -> Path:
        return Path(self._payload_value()["output_dir"])

    @property
    def manifest_sha256(self) -> str:
        return self._payload_value()["manifest_sha256"]

    def _payload_value(self) -> dict:
        value = _parse_canonical(self._payload, label="payload")
        expected = _digest_bytes(self._payload.encode("utf-8"))
        if expected != self._digest:
            raise ValueError("Inspection snapshot metadata digest mismatch")
        if value.get("format") != SNAPSHOT_FORMAT:
            raise ValueError("Unsupported inspection snapshot format")
        return value

    @property
    def report(self) -> dict:
        """Return a fresh report copy for descriptor translation."""
        return self._payload_value()["report"]

    def context(self) -> dict:
        return self._payload_value()["context"]

    def expected(self) -> dict:
        return self._payload_value()["expected"]

    def artifact_bindings(self) -> dict:
        return {item["role"]: {"path": item["path"], "sha256": item["sha256"]}
                for item in self._payload_value()["artifacts"]}

    def metadata(self) -> dict:
        """Return immutable snapshot metadata as a new JSON-compatible object."""
        return self._payload_value()

    def __repr__(self) -> str:  # pragma: no cover - convenience for diagnostics
        return f"InspectionSnapshot(output_dir={self.output_dir!s}, digest={self.digest})"


def validate_snapshot(snapshot: InspectionSnapshot, *, output_dir: Path,
                      context: dict, expected: dict, file_digest,
                      load_json, safe_local) -> dict:
    """Validate a snapshot against current output bytes and return its report.

    ``file_digest``, ``load_json`` and ``safe_local`` are supplied by the owner
    module so the snapshot layer stays independent from the family workflow's
    schema helpers.  This function only reads metadata and raw bytes; it never
    deserializes a tensor or checkpoint.
    """
    if type(snapshot) is not InspectionSnapshot:
        raise TypeError("Expected an InspectionSnapshot produced by the owner API")
    root = _concrete_path(Path(output_dir).absolute(), "output directory")
    safe_local(root)
    metadata = snapshot._payload_value()
    if _path_key(root) != _path_key(Path(metadata["output_dir"])):
        raise ValueError("Inspection snapshot output directory mismatch")
    if metadata["context"] != context:
        raise ValueError("Inspection snapshot/context mismatch")
    if metadata["expected"] != expected:
        raise ValueError("Inspection snapshot/expected mismatch")
    manifest_path = root / "output_manifest.json"
    safe_local(manifest_path)
    current_manifest_sha256 = file_digest(manifest_path)
    if current_manifest_sha256 != metadata["manifest_sha256"]:
        raise ValueError("Inspection snapshot is stale: output manifest changed")
    manifest = load_json(manifest_path)
    if type(manifest) is not dict or manifest.get("context") != metadata["manifest_context"]:
        raise ValueError("Inspection snapshot manifest context changed")
    expected_artifacts = snapshot.artifact_bindings()
    actual_artifacts = manifest.get("artifacts")
    if type(actual_artifacts) is not dict:
        raise ValueError("Inspection snapshot manifest artifacts are missing")
    if actual_artifacts != expected_artifacts:
        raise ValueError("Inspection snapshot artifact bindings changed")
    for role, binding in expected_artifacts.items():
        path = root / binding["path"]
        safe_local(path)
        if not path.is_file():
            raise ValueError(f"Inspection snapshot artifact is missing: {role}")
        current = file_digest(path)
        if current != binding["sha256"]:
            raise ValueError(f"Inspection snapshot is stale: artifact hash mismatch: {role}")
    report = snapshot.report
    if report.get("status") != "MECHANICALLY_VERIFIED" or report.get("blockers") != []:
        raise ValueError("Inspection snapshot does not contain a mechanically verified report")
    if report.get("context") != context:
        raise ValueError("Inspection snapshot report/context mismatch")
    observed = report.get("observed")
    if type(observed) is not dict or observed.get("artifacts") != expected_artifacts:
        raise ValueError("Inspection snapshot report artifact bindings changed")
    return report


__all__ = ["InspectionSnapshot", "SNAPSHOT_FORMAT", "validate_snapshot"]
