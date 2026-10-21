"""Confine RML filesystem access while preserving metadata URI semantics."""

from __future__ import annotations

import hashlib
from pathlib import Path, PureWindowsPath
from urllib.parse import quote, unquote, urlparse

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


def bound_artifact_path(root: str | Path, pointer: str | Path) -> Path:
    """Resolve retained bytes, keeping metadata confined to ``repo_local_path``.

    Existing external locators may name an absolute file on this machine. They
    confer neither portability nor availability elsewhere; every consumer still
    verifies the frozen digest. No network retrieval or fallback is performed.
    """
    raw = str(pointer)
    if not raw.startswith("external://"):
        return repo_local_path(root, pointer)
    parsed = urlparse(raw)
    if parsed.netloc or parsed.query or parsed.fragment:
        raise ValueError("external artifact requires an absolute local path without a host")
    value = unquote(parsed.path)
    if len(value) >= 4 and value[0] == "/" and PureWindowsPath(value[1:]).drive:
        value = value[1:]
    path = Path(value)
    if not path.is_absolute() or path == Path(path.anchor):
        raise ValueError("external artifact requires an absolute file path")
    from molgap.common.io import _safe_local
    _safe_local(path)
    return path.resolve()


def artifact_locator(root: str | Path, path: str | Path) -> str:
    """Use a portable repo pointer when possible, otherwise explicit local custody."""
    path = Path(path).absolute()
    from molgap.common.io import _safe_local
    _safe_local(path)
    try:
        return path.relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        locator = "external:///" + quote(path.as_posix().lstrip("/"), safe="/:")
        bound_artifact_path(root, locator)
        return locator


def verify_bound_artifact(root: Path, pointer: str, expected_sha256: str) -> None:
    path = (bound_artifact_path(root, pointer) if pointer.startswith("external://")
            else resolve_repo_pointer(root, pointer))
    if path is None:
        raise ValueError(f"bound artifact must be repository-local: {pointer}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected_sha256:
        raise ValueError(f"bound artifact SHA mismatch: {pointer}")
