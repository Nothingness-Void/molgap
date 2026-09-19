"""Machine-neutral repository pointer and evidence binding helpers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .comparison_readiness import (
    validate_reference_bundle,
    validate_target_transform_asset,
)


ALLOWED_REMOTE_SCHEMES = frozenset(
    {"external", "git", "https", "http", "ims", "kaggle", "scnet"}
)

REFERENCE_BUNDLE_POINTER_FIELDS = (
    "contract_ref",
    "runtime_certificate_ref",
    "row_manifest_ref",
    "target_manifest_ref",
    "trace_manifest_ref",
    "role_history_ref",
    "target_transform_asset_ref",
    "cost_records_ref",
    "acceptance_ref",
    "decision_ref",
)


def load_json_object(path: Path) -> dict[str, Any]:
    """Load one JSON object without accepting arrays or scalar payloads."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON record must be an object: {path}")
    return value


def resolve_repo_pointer(repo_root: Path, pointer: str) -> Path | None:
    """Resolve a safe repository pointer; recognized remote schemes stay unresolved."""

    root = Path(repo_root).resolve()
    parsed = urlparse(pointer)
    if parsed.scheme in ALLOWED_REMOTE_SCHEMES:
        return None
    if parsed.scheme == "repo":
        pointer = f"{parsed.netloc}{parsed.path}".lstrip("/")
    elif parsed.scheme:
        raise ValueError(f"unsupported pointer scheme: {pointer}")
    path = (root / pointer).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"repository pointer escapes root: {pointer}") from exc
    if not path.is_file():
        raise ValueError(f"required repository pointer is missing: {pointer}")
    return path


def verify_bound_artifact(root: Path, pointer: str, expected_sha256: str) -> None:
    """Verify strict evidence through a repository-local immutable artifact."""

    path = resolve_repo_pointer(root, pointer)
    if path is None:
        raise ValueError(
            "strict observed evidence must bind a repository-local retrievable "
            f"artifact or manifest: {pointer}"
        )
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected_sha256:
        raise ValueError(
            f"bound artifact SHA mismatch for {pointer}: "
            f"expected {expected_sha256}, found {actual}"
        )


def validate_release_reference_bundle_evidence(
    reference_bundle: Mapping[str, Any],
    *,
    repo_root: str | Path,
    reference_bundle_path: str | Path,
) -> dict[str, Any]:
    """Bind a compute release to one real, locally verifiable reference bundle.

    Remote evidence is valid historical provenance, but is not independently
    release-qualified.  A release must therefore point to a repository-local
    qualified artifact or manifest for every bundle pointer.
    """

    root = Path(repo_root).resolve()
    supplied_path = Path(reference_bundle_path)
    bundle_path = (
        supplied_path.resolve()
        if supplied_path.is_absolute()
        else (root / supplied_path).resolve()
    )
    try:
        relative_bundle_path = bundle_path.relative_to(root)
    except ValueError as exc:
        raise ValueError("reference bundle path escapes repository root") from exc
    if (
        bundle_path.name != "reference_bundle.json"
        or not relative_bundle_path.parts
        or relative_bundle_path.parts[0] != "experiments"
    ):
        raise ValueError(
            "release reference bundle must be an experiments/**/reference_bundle.json"
        )
    if not bundle_path.is_file():
        raise ValueError(f"release reference bundle is missing: {relative_bundle_path}")

    persisted_bundle = load_json_object(bundle_path)
    supplied_bundle = dict(reference_bundle)
    if persisted_bundle != supplied_bundle:
        raise ValueError("supplied reference bundle does not match repository artifact")
    validated_bundle = validate_reference_bundle(persisted_bundle)

    resolved: dict[str, Path] = {}
    for field in REFERENCE_BUNDLE_POINTER_FIELDS:
        pointer = validated_bundle[field]
        path = resolve_repo_pointer(root, pointer)
        if path is None:
            raise ValueError(
                f"{field} is remote and has no repository-local release qualification: "
                f"{pointer}"
            )
        resolved[field] = path

    target_transform = validate_target_transform_asset(
        load_json_object(resolved["target_transform_asset_ref"])
    )
    identity = validated_bundle["comparison_identity"]
    if identity["target_transform_identity"] != target_transform["asset_id"]:
        raise ValueError("reference target-transform identity does not match asset_id")
    if identity["target_transform_asset_sha256"] != target_transform["asset_sha256"]:
        raise ValueError("reference target-transform SHA does not match validated asset")
    if identity["target_identity"] != target_transform["target_identity"]:
        raise ValueError("reference target identity does not match transform asset")
    return validated_bundle
