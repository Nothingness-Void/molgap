"""Verify the neutral source payload and expanded source before any install."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tarfile


SOURCE_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def find_one(pattern: str) -> Path:
    matches = sorted(
        path
        for path in Path("/kaggle/input").rglob(pattern)
        if path.is_file()
    )
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {pattern}, found {matches}")
    return matches[0]


def _inventory() -> dict[str, str]:
    try:
        payload = json.loads(find_one("SOURCE_FILES.json").read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Source identity mismatch: invalid inventory ({error})")
    if not isinstance(payload, dict) or not isinstance(payload.get("files"), list):
        raise RuntimeError("Source identity mismatch: inventory shape changed")
    result = {}
    for item in payload["files"]:
        if not isinstance(item, dict):
            raise RuntimeError("Source identity mismatch: invalid inventory entry")
        raw_path = item.get("path")
        digest = item.get("sha256")
        if not isinstance(raw_path, str) or not isinstance(digest, str):
            raise RuntimeError("Source identity mismatch: invalid inventory entry")
        relative = PurePosixPath(raw_path)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not relative.parts
            or relative.parts[0] != "src"
            or not SHA256_RE.fullmatch(digest)
            or raw_path in result
        ):
            raise RuntimeError(f"Source identity mismatch: invalid path {raw_path!r}")
        result[raw_path] = digest
    if not result:
        raise RuntimeError("Source identity mismatch: empty source inventory")
    return result


def _archive_inventory(archive: Path, expected: dict[str, str]) -> None:
    """Check archive members as well as the mounted expanded tree."""
    observed = {}
    try:
        handle = tarfile.open(archive, mode="r:*")
    except (OSError, tarfile.TarError) as error:
        raise RuntimeError(f"Source identity mismatch: unreadable payload ({error})")
    with handle:
        for member in handle.getmembers():
            if member.isdir():
                continue
            if not member.isfile():
                raise RuntimeError(
                    f"Source identity mismatch: non-file archive member {member.name}"
                )
            relative = PurePosixPath(member.name)
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or not relative.parts
                or relative.parts[0] != "src"
                or member.name in observed
            ):
                raise RuntimeError(
                    f"Source identity mismatch: invalid archive member {member.name}"
                )
            stream = handle.extractfile(member)
            if stream is None:
                raise RuntimeError(
                    f"Source identity mismatch: unreadable archive member {member.name}"
                )
            digest = hashlib.sha256(stream.read()).hexdigest()
            observed[member.name] = digest
    if observed != expected:
        raise RuntimeError(
            "Source identity mismatch: archive members differ from SOURCE_FILES.json"
        )


def _validate_expanded(root: Path, expected: dict[str, str]) -> None:
    if not root.is_dir():
        raise RuntimeError(f"Source identity mismatch: missing expanded src at {root}")
    observed = {}
    for path in root.rglob("*"):
        if path.is_symlink():
            raise RuntimeError(f"Source identity mismatch: symlink in src: {path}")
        if not path.is_file():
            continue
        relative = PurePosixPath("src") / path.relative_to(root).as_posix()
        name = relative.as_posix()
        observed[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    if set(observed) != set(expected):
        raise RuntimeError(
            "Source identity mismatch: expanded src inventory has missing or extra files"
        )
    for name, digest in expected.items():
        if observed[name] != digest:
            raise RuntimeError(f"Source identity mismatch: mounted source differs: {name}")


def _expanded_root(expected: dict[str, str], archive: Path) -> Path:
    modules = sorted(
        path
        for path in Path("/kaggle/input").rglob(
            "src/molgap/k1_edge_kaggle_runtime.py"
        )
        if path.is_file()
    )
    roots = sorted({path.parents[1] for path in modules}, key=str)
    if len(roots) > 1:
        raise RuntimeError(f"Source identity mismatch: ambiguous expanded src {roots}")
    if roots:
        _validate_expanded(roots[0], expected)
        return roots[0]

    # The neutral suffix avoids archive auto-expansion/name heuristics.  This
    # fallback is still checked against the exact archive and file inventory.
    expanded = Path("/kaggle/working") / (
        "_k1_edge_source_" + hashlib.sha256(archive.read_bytes()).hexdigest()[:12]
    )
    if not (expanded / "src").is_dir():
        try:
            shutil.unpack_archive(archive, expanded, format="gztar")
        except (OSError, shutil.ReadError) as error:
            raise RuntimeError(
                f"Source identity mismatch: cannot expand neutral payload ({error})"
            )
    root = expanded / "src"
    _validate_expanded(root, expected)
    return root


def _verified_source() -> tuple[Path, str, str]:
    archive = find_one("source_payload.bin")
    sidecar = find_one("SOURCE_ARCHIVE_SHA256.txt").read_text().strip()
    commit = find_one("SOURCE_COMMIT.txt").read_text().strip()
    if not SHA256_RE.fullmatch(sidecar) or not SOURCE_COMMIT_RE.fullmatch(commit):
        raise RuntimeError("Source identity mismatch: invalid source sidecars")
    if hashlib.sha256(archive.read_bytes()).hexdigest() != sidecar:
        raise RuntimeError("Source identity mismatch: source_payload.bin hash changed")
    expected = _inventory()
    _archive_inventory(archive, expected)
    root = _expanded_root(expected, archive)
    return root, commit, sidecar


def main() -> None:
    # This entire verification path is stdlib-only and intentionally precedes
    # dependency installation and all molgap imports.
    root, commit, archive_sha = _verified_source()
    sys.path.insert(0, str(root))
    from molgap.k1_edge_kaggle_runtime import main as run

    run(
        {
            "python_root": str(root),
            "source_commit": commit,
            "source_archive_sha256": archive_sha,
        }
    )


if __name__ == "__main__":
    main()
