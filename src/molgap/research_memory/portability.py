"""Check that the local RML evidence closure exists in the committed tree."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path, PureWindowsPath
from tempfile import TemporaryDirectory
from typing import Any, Iterable
from urllib.parse import urlparse

from molgap.evidence_pointers import ALLOWED_REMOTE_SCHEMES, load_json_object

from .paths import repo_local_path, resolve_repo_pointer


def require_byte_preserving_git_paths(root: Path, paths: Iterable[str | Path]) -> None:
    """Require committed attributes that preserve hash-bound bytes verbatim.

    The check is deliberately limited to the paths about to be published.  It
    does not migrate existing evidence and it does not inspect model payloads.
    The empty-tree probe keeps an uncommitted ``.gitattributes`` or an
    ``info/attributes`` rule from being mistaken for a committed contract.
    """
    names: set[str] = set()
    for pointer in paths:
        raw = str(pointer)
        if raw.startswith("external://"):
            continue
        if raw.startswith("repo://"):
            parsed = urlparse(raw)
            raw = f"{parsed.netloc}{parsed.path}".lstrip("/")
        names.add(repo_local_path(root, raw).relative_to(root).as_posix())
    if not names:
        return

    attributes = ("text", "filter", "ident", "working-tree-encoding")
    request = b"".join(name.encode("utf-8") + b"\0" for name in sorted(names))
    # Keep the empty-tree probe outside the repository: qualification is a read-only
    # publication check and must not add a loose object during validate-only planning.
    git_objects = subprocess.run(
        ["git", "rev-parse", "--git-path", "objects"],
        cwd=root,
        capture_output=True,
    )
    if git_objects.returncode:
        raise ValueError("Cannot qualify Git byte preservation outside an initialized repository")
    object_path = Path(git_objects.stdout.decode().strip())
    alternate = str((root / object_path).resolve() if not object_path.is_absolute()
                    else object_path.resolve())
    with TemporaryDirectory(prefix="molgap-git-attrs-") as object_dir:
        env = {**os.environ, "GIT_ATTR_NOSYSTEM": "1",
               "GIT_OBJECT_DIRECTORY": object_dir,
               "GIT_ALTERNATE_OBJECT_DIRECTORIES": alternate}
        empty = subprocess.run(
            ["git", "hash-object", "-w", "-t", "tree", "--stdin"],
            cwd=root,
            input=b"",
            capture_output=True,
            env=env,
        )
        if empty.returncode:
            raise ValueError("Cannot qualify Git byte preservation outside an initialized repository")
        empty_tree = empty.stdout.decode("ascii").strip()
        for source in (empty_tree, "HEAD", None):
            options = ["-c", "core.attributesFile=" + os.devnull] if source else []
            result = subprocess.run(
                [
                    "git",
                    *options,
                    "check-attr",
                    *( ["--source=" + source] if source else [] ),
                    "-z",
                    "--stdin",
                    *attributes,
                ],
                cwd=root,
                input=request,
                capture_output=True,
                env=env,
            )
            if result.returncode:
                raise ValueError(
                    "Cannot qualify Git byte preservation; commit repository attributes first"
                )
            fields = result.stdout.split(b"\0")
            if fields[-1:] != [b""] or len(fields) != len(names) * len(attributes) * 3 + 1:
                raise ValueError("Malformed Git attribute qualification response")
            for index in range(0, len(fields) - 1, 3):
                name, attr, value = (
                    item.decode("utf-8") for item in fields[index : index + 3]
                )
                if name not in names or attr not in attributes:
                    raise ValueError("Unexpected Git attribute qualification response")
                if source == empty_tree:
                    if value != "unspecified":
                        raise ValueError(
                            f"Git info/attributes masks hash-bound path: {name}; "
                            "move its transform rules into committed .gitattributes"
                        )
                    continue
                if value != "unset" and (attr == "text" or value != "unspecified"):
                    raise ValueError(
                        f"Git may transform hash-bound bytes: {name} ({attr}={value}); "
                        "commit -text -filter -ident -working-tree-encoding for this path "
                        "before publication"
                    )


# These identifiers and auxiliary logs are not repository file pointers.
_NON_FILE_REFERENCE_FIELDS = {
    "expected_native_cost_ref",
    "kernel_ref",
    "raw_log_ref",
}


def _pointer_values(value: Any, parent_key: str | None = None) -> Iterable[str]:
    if isinstance(value, dict):
        for key, child in value.items():
            is_record_pointer = (
                key.endswith("_ref") and key not in _NON_FILE_REFERENCE_FIELDS
            )
            is_authority_pointer = parent_key == "authority" and key == "pointers"
            if is_record_pointer or is_authority_pointer:
                if isinstance(child, str):
                    yield child
                elif isinstance(child, list):
                    yield from (item for item in child if isinstance(item, str))
            elif key.endswith("_refs") and isinstance(child, list):
                yield from (item for item in child if isinstance(item, str))
            yield from _pointer_values(child, key)
    elif isinstance(value, list):
        for child in value:
            yield from _pointer_values(child, parent_key)


def _git_head_paths(root: Path) -> set[str]:
    result = subprocess.run(
        ["git", "-C", str(root), "ls-tree", "-r", "--name-only", "-z", "HEAD"],
        check=False,
        capture_output=True,
    )
    if result.returncode:
        message = result.stderr.decode(errors="replace").strip()
        raise ValueError(f"cannot inspect committed Git tree at {root}: {message}")
    return {
        os.fsdecode(item).replace("\\", "/")
        for item in result.stdout.split(b"\0")
        if item
    }


def uncommitted_closure_paths(
    repo_root: str | Path, records: dict[str, list[tuple[Path, dict[str, Any]]]]
) -> list[str]:
    """Return canonical RML inputs and local pointers absent from Git HEAD.

    Trace migration sidecars are included because replay consumes them by
    convention rather than through a field in the original trace manifest.
    """
    root = Path(repo_root).resolve()
    required = _evidence_closure_paths(root, records)
    head_paths = _git_head_paths(root)
    return sorted(
        _relative(root, path)
        for path in required
        if _relative(root, path) not in head_paths
    )


def _evidence_closure_paths(
    root: Path, records: dict[str, list[tuple[Path, dict[str, Any]]]]
) -> set[Path]:
    """Collect canonical records, their direct file refs, and replay sidecars."""
    required: set[Path] = set()
    pointer_documents: list[dict[str, Any]] = []
    trace_sidecars: set[Path] = set()

    for kind, entries in records.items():
        for path, record in entries:
            source = Path(path).resolve()
            required.add(source)
            pointer_documents.append(record)
            if kind == "traces":
                sidecar = source.parent / "trace_migration.json"
                if sidecar.is_file():
                    trace_sidecars.add(sidecar.resolve())

    for sidecar in trace_sidecars:
        required.add(sidecar)
        pointer_documents.append(load_json_object(sidecar))

    for document in pointer_documents:
        for pointer in _pointer_values(document):
            resolved = resolve_repo_pointer(root, pointer)
            if resolved is not None:
                required.add(resolved)
    return required


def _relative(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError(f"RML source escapes repository root: {path}") from exc


def _git_changed_paths(root: Path) -> set[str]:
    result = subprocess.run(
        ["git", "-C", str(root), "diff", "--name-only", "-z", "HEAD", "--"],
        check=False,
        capture_output=True,
    )
    if result.returncode:
        message = result.stderr.decode(errors="replace").strip()
        raise ValueError(f"cannot inspect working-tree changes at {root}: {message}")
    return {
        os.fsdecode(item).replace("\\", "/")
        for item in result.stdout.split(b"\0")
        if item
    }


def _runtime_paths(root: Path, head_paths: set[str]) -> set[Path]:
    required = {
        root / relative
        for relative in (
            "pyproject.toml",
            "src/molgap/__init__.py",
            "src/molgap/constants.py",
            "src/molgap/compat/__init__.py",
            "src/molgap/compat/legacy_constants.py",
            "src/molgap/comparison_readiness.py",
            "src/molgap/evidence_pointers.py",
            "src/molgap/v5_common.py",
        )
    }
    runtime_prefixes = (
        ("src/molgap/research_memory/", ".py"),
        ("research_memory/schemas/", ".json"),
        ("research_memory/policies/", ".json"),
    )
    candidates = set(head_paths)
    for directory, suffix in (
        (root / "src/molgap/research_memory", ".py"),
        (root / "research_memory/schemas", ".json"),
        (root / "research_memory/policies", ".json"),
    ):
        if directory.exists():
            candidates.update(
                path.relative_to(root).as_posix()
                for path in directory.rglob(f"*{suffix}")
                if path.is_file()
            )
    for relative in candidates:
        if any(
            relative.startswith(prefix) and relative.endswith(suffix)
            for prefix, suffix in runtime_prefixes
        ):
            required.add(root / relative)

    from .compiler import DERIVED_FILENAMES

    required.update(root / "research_memory/derived" / name for name in DERIVED_FILENAMES)
    return required


def committed_head_differences(
    repo_root: str | Path, records: dict[str, list[tuple[Path, dict[str, Any]]]]
) -> dict[str, list[str]]:
    """Report missing or modified RML inputs/runtime/outputs relative to HEAD."""
    root = Path(repo_root).resolve()
    evidence_paths = _evidence_closure_paths(root, records)
    head_paths = _git_head_paths(root)
    required = evidence_paths | _runtime_paths(root, head_paths)
    relative_paths = {_relative(root, path) for path in required}
    missing = sorted(relative_paths - head_paths)
    changed = sorted(_git_changed_paths(root) & relative_paths & head_paths)
    return {"missing_from_head": missing, "changed_from_head": changed}


def missing_locally_claimed_artifacts(
    repo_root: str | Path, records: dict[str, list[tuple[Path, dict[str, Any]]]]
) -> list[str]:
    """Find local files that V5 evidence explicitly claims are locally retained."""
    root = Path(repo_root).resolve()
    missing = []
    for evidence_path, evidence in records.get("evidence", []):
        for artifact in evidence.get("artifacts", []):
            if "local" not in str(artifact.get("availability", "")).lower():
                continue
            locator = artifact.get("locator")
            if not isinstance(locator, str) or not locator.strip():
                continue
            parsed = urlparse(locator)
            if parsed.scheme in ALLOWED_REMOTE_SCHEMES:
                continue
            if parsed.scheme == "repo":
                locator = f"{parsed.netloc}{parsed.path}".lstrip("/")
            elif parsed.scheme and not PureWindowsPath(locator).drive:
                raise ValueError(f"unsupported artifact locator scheme: {locator}")
            target = repo_local_path(root, locator)
            if not target.is_file():
                missing.append(f"{evidence_path}: {locator}")
    return sorted(missing)
