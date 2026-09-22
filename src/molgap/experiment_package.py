"""Immutable source receipts bound to ExperimentSpec, with no execution authority."""
from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
import tarfile
import tempfile
from pathlib import Path, PurePosixPath, PureWindowsPath

from .experiment_spec import ExperimentSpec
from .screen_policy import canonical_fingerprint
from .v4_bundle import TEXT_SUFFIXES, _payload, build_v4_source_bundle


PACKAGE_FORMAT = "molgap-experiment-source-package-v1"
PACKAGE_STATUS = "source_packaged"
SIDECARS = frozenset({
    "source.tar.gz", "SOURCE_COMMIT.txt", "SOURCE_ARCHIVE_SHA256.txt",
    "SOURCE_FILES.json", "experiment_spec.json", "package_manifest.json",
})
_BLOCKED_PARTS = frozenset({
    ".git", ".ssh", ".aws", ".azure", "credential", "credentials", "secret",
    "secrets", "key", "keys", "password", "passwords", "token", "tokens",
    "data", "dataset", "datasets", "checkpoint",
    "checkpoints", "_records", "records", "output", "outputs", "result",
    "results", "artifacts", "receipt", "receipts",
})
_BLOCKED_SUFFIXES = frozenset({
    ".pem", ".key", ".p12", ".pfx", ".pt", ".pth", ".ckpt", ".pkl",
    ".pickle", ".npy", ".npz", ".parquet", ".csv", ".tsv", ".sdf",
    ".h5", ".hdf5", ".sqlite", ".db", ".zip", ".gz", ".tar",
})


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_bytes(value: dict) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _load(payload: bytes):
    return json.loads(payload, object_pairs_hook=_unique)


def _name(value) -> str:
    # Reject aliases instead of silently changing the caller's allowlist.
    if type(value) is not str or not value or "\\" in value or ":" in value:
        raise ValueError("Source paths must be nonempty POSIX relative strings")
    path = PurePosixPath(value)
    if (path.is_absolute() or PureWindowsPath(value).drive
            or any(part in {"", ".", ".."} for part in value.split("/"))
            or any(ord(c) < 32 for c in value)
            or any(part.endswith((".", " ")) for part in path.parts)):
        raise ValueError(f"Unsafe source path: {value}")
    lowered = [part.lower() for part in path.parts]
    tokens = {token for part in lowered for token in re.split(r"[._-]+", part)}
    if (set(lowered) & _BLOCKED_PARTS or tokens & _BLOCKED_PARTS
            or any(part == ".env" or part.startswith(".env.") for part in lowered)
            or path.suffix.lower() in _BLOCKED_SUFFIXES
            or path.name.lower() in {name.lower() for name in SIDECARS}
            or path.name.lower().startswith(("id_rsa", "id_ed25519"))):
        raise ValueError(f"Source allowlist includes reserved or sensitive path: {value}")
    return value


def _allowlist(relative_paths) -> list[str]:
    if isinstance(relative_paths, (str, bytes)):
        raise ValueError("Expected an explicit collection of relative paths")
    names = [_name(value) for value in relative_paths]
    if not names or len(names) != len(set(names)):
        raise ValueError("Source allowlist must be nonempty and unique")
    if len({name.casefold() for name in names}) != len(names):
        raise ValueError("Case-colliding source names are not portable")
    return names


def _no_links(path: Path) -> None:
    for part in (path, *path.parents):
        if part.is_symlink() or (hasattr(part, "is_junction") and part.is_junction()):
            raise ValueError(f"Symlink/junction path is forbidden: {part}")


def _source(root: Path, name: str) -> Path:
    path = root / name
    _no_links(path)
    if not path.resolve().is_relative_to(root) or not path.is_file():
        raise ValueError(f"Source must be a regular repository file: {name}")
    if not stat.S_ISREG(path.stat().st_mode):
        raise ValueError(f"Nonregular source: {name}")
    return path


def _git(root: Path, *args: str) -> bytes:
    return subprocess.run(["git", "--literal-pathspecs", *args], cwd=root,
                          check=True, capture_output=True).stdout


def _spec(spec: ExperimentSpec) -> ExperimentSpec:
    if type(spec) is not ExperimentSpec:
        raise TypeError("Expected exactly ExperimentSpec")
    rebuilt = ExperimentSpec.from_json(spec.to_json())
    if rebuilt.to_json() != spec.to_json() or rebuilt.identity != spec.identity:
        raise ValueError("ExperimentSpec canonical snapshot/identity mismatch")
    return rebuilt


def _manifest(spec, names, commit, archive_hash, inventory_hash):
    declaration = spec.to_dict()
    stable = {
        "format": PACKAGE_FORMAT, "package_status": PACKAGE_STATUS,
        "spec_identity": spec.identity,
        "spec_sha256": _sha(spec.to_json().encode("utf-8")),
        "experiment_id": declaration["experiment_id"],
        "logical_run_id": declaration["logical_run_id"],
        "arms": [{"arm_id": arm["arm_id"], "identity": canonical_fingerprint(arm)}
                 for arm in declaration["arms"]],
        "source_commit": commit, "archive_sha256": archive_hash,
        "inventory_sha256": inventory_hash, "relative_allowlist": names,
    }
    # Only package_identity is excluded; every other field is stable and bound.
    return {**stable, "package_identity": canonical_fingerprint(stable)}


def _atomic(path: Path, payload: bytes) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_experiment_source_package(
    spec: ExperimentSpec, repo_root: Path, relative_paths, output_dir: Path,
) -> dict:
    """Publish a new source-only package, manifest last; never overwrite a package."""
    spec = _spec(spec)
    names = _allowlist(relative_paths)
    root = Path(repo_root).absolute()
    output = Path(output_dir).absolute()
    _no_links(root)
    _no_links(output)
    root, output = root.resolve(), output.resolve()
    for name in names:
        source = _source(root, name)
        if source == output or source.is_relative_to(output) or output.is_relative_to(source):
            raise ValueError("Package output overlaps source allowlist")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Package output must be new or empty")
    commit = _git(root, "rev-parse", "HEAD").decode().strip()
    output.mkdir(parents=True, exist_ok=True)
    # Exclusive reservation also prevents two cooperating builders sharing an empty dir.
    lock = output / ".package-building"
    with lock.open("xb"):
        pass
    try:
        with tempfile.TemporaryDirectory(prefix=".source-stage-", dir=output) as stage_name:
            stage = Path(stage_name)
            bundle = build_v4_source_bundle(repo_root=root, relative_paths=names,
                                           output_dir=stage, source_commit=commit)
            manifest = _manifest(spec, names, commit, bundle["archive_sha256"],
                                 _sha((stage / "SOURCE_FILES.json").read_bytes()))
            _atomic(stage / "experiment_spec.json", spec.to_json().encode("utf-8"))
            _atomic(stage / "package_manifest.json", _json_bytes(manifest))
            # Recheck commit blobs too: Git clean filters must not conceal changed bytes.
            verify_experiment_source_package(stage, repo_root=root)
            for name in sorted(SIDECARS - {"package_manifest.json"}):
                os.replace(stage / name, output / name)
        _atomic(output / "package_manifest.json", _json_bytes(manifest))
        return manifest
    finally:
        lock.unlink(missing_ok=True)


def verify_experiment_source_package(package_dir: Path, repo_root: Path | None = None) -> dict:
    """Verify source integrity only, optionally against commit blobs and tracked payloads."""
    directory = Path(package_dir).absolute()
    _no_links(directory)
    if not directory.is_dir() or {p.name for p in directory.iterdir()} != SIDECARS:
        raise ValueError("Package must contain exactly the six declared files")
    for name in SIDECARS:
        _source(directory, name)
    raw = {name: (directory / name).read_bytes() for name in SIDECARS}
    manifest = _load(raw["package_manifest.json"])
    if type(manifest) is not dict or raw["package_manifest.json"] != _json_bytes(manifest):
        raise ValueError("Manifest must be a canonical JSON object")
    spec = _spec(ExperimentSpec.from_json(raw["experiment_spec.json"].decode("utf-8")))
    if raw["experiment_spec.json"] != spec.to_json().encode("utf-8"):
        raise ValueError("Spec sidecar must be the canonical snapshot")
    names = _allowlist(manifest.get("relative_allowlist", []))
    commit = raw["SOURCE_COMMIT.txt"].decode("ascii").removesuffix("\n")
    if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", commit):
        raise ValueError("Invalid source commit")
    archive_hash = _sha(raw["source.tar.gz"])
    expected = _manifest(spec, names, commit, archive_hash, _sha(raw["SOURCE_FILES.json"]))
    if manifest != expected:
        raise ValueError("Package manifest identity or binding mismatch")
    if (raw["SOURCE_COMMIT.txt"] != (commit + "\n").encode()
            or raw["SOURCE_ARCHIVE_SHA256.txt"] != (archive_hash + "\n").encode()):
        raise ValueError("Source sidecar hash/commit mismatch")
    inventory = _load(raw["SOURCE_FILES.json"])
    if (type(inventory) is not dict or set(inventory) != {"format", "source_commit", "files"}
            or inventory["format"] != "molgap-v4-source-inventory-v1"
            or inventory["source_commit"] != commit or type(inventory["files"]) is not list):
        raise ValueError("Invalid V4 inventory")
    entries = {}
    for entry in inventory["files"]:
        if type(entry) is not dict or set(entry) != {"path", "sha256", "bytes"}:
            raise ValueError("Invalid inventory entry")
        name = _name(entry["path"])
        if (name in entries or type(entry["bytes"]) is not int or entry["bytes"] < 0
                or type(entry["sha256"]) is not str
                or not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"])):
            raise ValueError("Invalid inventory payload declaration")
        entries[name] = entry
    if list(entries) != sorted(names):
        raise ValueError("Inventory and explicit allowlist differ")
    seen = set()
    # Never extract untrusted tar members onto the filesystem.
    with tarfile.open(directory / "source.tar.gz", mode="r:gz") as archive:
        for member in archive:
            name = _name(member.name)
            if not member.isfile() or member.linkname or name in seen or name not in entries:
                raise ValueError("Unsafe, duplicate, or unexpected archive member")
            entry = entries[name]
            if member.size != entry["bytes"]:
                raise ValueError("Archive member size mismatch")
            stream = archive.extractfile(member)
            if stream is None or _sha(stream.read()) != entry["sha256"]:
                raise ValueError("Archive member payload mismatch")
            seen.add(name)
    if seen != set(entries):
        raise ValueError("Archive member set differs from inventory")
    if repo_root is not None:
        root = Path(repo_root).absolute()
        _no_links(root)
        root = root.resolve()
        tracked = set(_git(root, "ls-files", "-z").decode("utf-8").split("\0"))
        for name, entry in entries.items():
            if name not in tracked:
                raise ValueError(f"Source no longer tracked: {name}")
            path = _source(root, name)
            tree = _git(root, "ls-tree", "-z", commit, "--", name).split(b"\0")
            expected_tree = [row for row in tree if row]
            if len(expected_tree) != 1 or expected_tree[0].split(b" ", 1)[0] not in {b"100644", b"100755"}:
                raise ValueError(f"Commit source is not a regular tracked file: {name}")
            payload = _git(root, "show", f"{commit}:{name}")
            if path.suffix.lower() in TEXT_SUFFIXES:
                payload = payload.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
            if any(len(data) != entry["bytes"] or _sha(data) != entry["sha256"]
                   for data in (payload, _payload(path))):
                raise ValueError(f"Repository source payload mismatch: {name}")
    return manifest


def inspect_experiment_source_package(package_dir: Path, repo_root: Path | None = None) -> dict:
    """Inspection is fail-closed verification, never a readiness decision."""
    return verify_experiment_source_package(package_dir, repo_root)
