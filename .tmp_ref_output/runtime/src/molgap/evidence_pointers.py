"""Machine-neutral repository pointer and evidence binding helpers."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


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
