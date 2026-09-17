"""Reproducible minimal source bundles for V4 remote runs."""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import subprocess
import tarfile
from pathlib import Path, PurePosixPath, PureWindowsPath

from .training_reproducibility import sha256_file


TEXT_SUFFIXES = {".py", ".md", ".json", ".toml", ".slurm", ".sh", ".txt"}


def _payload(path: Path) -> bytes:
    payload = path.read_bytes()
    if path.suffix.lower() in TEXT_SUFFIXES:
        payload = payload.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return payload


def _tracked_paths(repo_root: Path) -> set[str]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    return {item.decode("utf-8") for item in result.stdout.split(b"\0") if item}


def _assert_clean_paths(repo_root: Path, names: list[str]) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, check=True,
        capture_output=True, text=True,
    )
    commit = result.stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--", *names],
        cwd=repo_root, check=True, capture_output=True, text=True,
    )
    if status.stdout.strip():
        raise RuntimeError("V4 bundle source files must be committed before packaging")
    return commit


def build_v4_source_bundle(
    *,
    repo_root: Path,
    relative_paths,
    output_dir: Path,
    source_commit: str,
    archive_name: str = "source.tar.gz",
) -> dict:
    """Build a small LF-normalized archive from an explicit tracked allowlist."""
    repo_root = repo_root.resolve()
    output_dir = output_dir.resolve()
    tracked = _tracked_paths(repo_root)
    names = sorted({PurePosixPath(str(path).replace("\\", "/")).as_posix() for path in relative_paths})
    if not names:
        raise ValueError("A V4 source bundle cannot be empty")
    if any(
        name.startswith("../")
        or name.startswith("/")
        or PureWindowsPath(name).drive
        or ".." in PurePosixPath(name).parts
        for name in names
    ):
        raise ValueError("V4 source paths must stay inside the repository")
    missing = [name for name in names if name not in tracked]
    if missing:
        raise RuntimeError(f"V4 source bundle includes untracked paths: {missing}")
    observed_commit = _assert_clean_paths(repo_root, names)
    if source_commit != observed_commit:
        raise RuntimeError(
            f"Requested source commit {source_commit} is not checked-out HEAD {observed_commit}"
        )

    entries = []
    payloads = {}
    for name in names:
        path = repo_root / Path(name)
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.is_symlink():
            raise RuntimeError(f"V4 bundles do not accept symlinked source: {name}")
        payload = _payload(path)
        payloads[name] = payload
        entries.append({"path": name, "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)})

    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / archive_name
    temporary = archive_path.with_name(f".{archive_path.name}.tmp")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as archive:
                for name in names:
                    payload = payloads[name]
                    info = tarfile.TarInfo(name)
                    info.size = len(payload)
                    info.mode = 0o644
                    info.mtime = 0
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    archive.addfile(info, io.BytesIO(payload))
    os.replace(temporary, archive_path)
    archive_sha256 = sha256_file(archive_path)
    inventory = {
        "format": "molgap-v4-source-inventory-v1",
        "source_commit": source_commit,
        "files": entries,
    }
    for name, contents in (
        ("SOURCE_COMMIT.txt", source_commit + "\n"),
        ("SOURCE_ARCHIVE_SHA256.txt", archive_sha256 + "\n"),
        ("SOURCE_FILES.json", json.dumps(inventory, indent=2) + "\n"),
    ):
        target = output_dir / name
        sidecar_tmp = target.with_name(f".{target.name}.tmp")
        with sidecar_tmp.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(contents)
        os.replace(sidecar_tmp, target)
    return {
        "archive": str(archive_path),
        "archive_sha256": archive_sha256,
        "source_commit": source_commit,
        "file_count": len(entries),
        "payload_bytes": sum(item["bytes"] for item in entries),
    }
