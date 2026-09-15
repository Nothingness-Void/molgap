"""Verify immutable source, then run the approved GPSPP-local T4 pair."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tarfile


MODES = (
    "neural_atom_k1_gpspp_sender",
    "neural_atom_k1_gpspp_bidirectional",
)
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
SHA_RE = re.compile(r"^[0-9a-f]{64}$")


def find_one(pattern: str) -> Path:
    matches = sorted(
        path for path in Path("/kaggle/input").rglob(pattern) if path.is_file()
    )
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {pattern}, found {matches}")
    return matches[0]


def inventory() -> dict[str, str]:
    payload = json.loads(find_one("SOURCE_FILES.json").read_text())
    result = {}
    for item in payload.get("files", []):
        path = PurePosixPath(item.get("path", ""))
        digest = item.get("sha256", "")
        if (
            path.is_absolute()
            or ".." in path.parts
            or not path.parts
            or path.parts[0] != "src"
            or not SHA_RE.fullmatch(digest)
            or path.as_posix() in result
        ):
            raise RuntimeError("Source identity mismatch: invalid inventory")
        result[path.as_posix()] = digest
    if not result:
        raise RuntimeError("Source identity mismatch: empty inventory")
    return result


def verify_archive(archive: Path, expected: dict[str, str]) -> None:
    observed = {}
    with tarfile.open(archive, "r:*") as handle:
        for member in handle.getmembers():
            if member.isdir():
                continue
            path = PurePosixPath(member.name)
            if (
                not member.isfile()
                or path.is_absolute()
                or ".." in path.parts
                or not path.parts
                or path.parts[0] != "src"
                or member.name in observed
            ):
                raise RuntimeError("Source identity mismatch: invalid archive")
            stream = handle.extractfile(member)
            if stream is None:
                raise RuntimeError("Source identity mismatch: unreadable member")
            observed[member.name] = hashlib.sha256(stream.read()).hexdigest()
    if observed != expected:
        raise RuntimeError("Source identity mismatch: archive inventory changed")


def verify_tree(root: Path, expected: dict[str, str]) -> None:
    observed = {}
    for path in root.rglob("*"):
        if path.is_symlink():
            raise RuntimeError("Source identity mismatch: symlink")
        if path.is_file():
            relative = "src/" + path.relative_to(root).as_posix()
            observed[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    if observed != expected:
        raise RuntimeError("Source identity mismatch: expanded tree changed")


def verified_source() -> tuple[Path, str, str]:
    archive = find_one("source_payload.bin")
    expected_sha = find_one("SOURCE_ARCHIVE_SHA256.txt").read_text().strip()
    commit = find_one("SOURCE_COMMIT.txt").read_text().strip()
    if not SHA_RE.fullmatch(expected_sha) or not COMMIT_RE.fullmatch(commit):
        raise RuntimeError("Source identity mismatch: invalid sidecars")
    if hashlib.sha256(archive.read_bytes()).hexdigest() != expected_sha:
        raise RuntimeError("Source identity mismatch: payload hash changed")
    expected = inventory()
    verify_archive(archive, expected)
    # Kaggle may auto-expand both source.tar.gz and its generated src.zip,
    # yielding several equivalent mounted roots. Never choose among them:
    # expand the already hash-verified neutral payload into one private root.
    expanded = Path("/kaggle/working") / (
        "_k1_gpspp_source_" + expected_sha[:12]
    )
    if not (expanded / "src").is_dir():
        shutil.unpack_archive(archive, expanded, format="gztar")
    root = expanded / "src"
    verify_tree(root, expected)
    return root, commit, expected_sha


def main() -> None:
    # No project import occurs before source bytes and inventory are verified.
    root, commit, archive_sha = verified_source()
    sys.path.insert(0, str(root))
    os.environ["MOLGAP_SCREEN_MODES"] = json.dumps(MODES)
    os.environ["MOLGAP_K1_OUTPUT_ROOT"] = (
        "/kaggle/working/pcqm_k1_gpspp_local"
    )
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
