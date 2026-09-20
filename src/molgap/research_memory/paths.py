"""Confine RML filesystem access while preserving metadata URI semantics."""

from __future__ import annotations

import hashlib
from pathlib import Path, PureWindowsPath
from urllib.parse import urlparse

from molgap.evidence_pointers import ALLOWED_REMOTE_SCHEMES


def repo_local_path(repo_root: str | Path, value: str | Path) -> Path:
    """Reject traversal before normalization and escapes through existing links."""
    root = Path(repo_root).resolve()
    raw = str(value)
    if not raw or "://" in raw:
        raise ValueError("a local filesystem path is required, not a metadata URI")
    if ".." in Path(raw).parts or ".." in PureWindowsPath(raw).parts:
        raise ValueError(f"repository path traversal is forbidden: {value}")
    path = (root / value).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"filesystem path escapes repository: {value}") from exc
    return path


def resolve_repo_pointer(repo_root: str | Path, pointer: str) -> Path | None:
    """Remote metadata stays unresolved; local pointers use the shared boundary."""
    parsed = urlparse(pointer)
    if parsed.scheme in ALLOWED_REMOTE_SCHEMES:
        return None
    if parsed.scheme == "repo":
        pointer = f"{parsed.netloc}{parsed.path}".lstrip("/")
    elif parsed.scheme and not PureWindowsPath(pointer).drive:
        raise ValueError(f"unsupported pointer scheme: {pointer}")
    path = repo_local_path(repo_root, pointer)
    if not path.is_file():
        raise ValueError(f"required repository pointer is missing: {pointer}")
    return path


def verify_bound_artifact(root: Path, pointer: str, expected_sha256: str) -> None:
    path = resolve_repo_pointer(root, pointer)
    if path is None:
        raise ValueError(f"bound artifact must be repository-local: {pointer}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected_sha256:
        raise ValueError(f"bound artifact SHA mismatch: {pointer}")
