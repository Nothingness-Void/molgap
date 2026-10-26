"""Immutable source receipts bound to ExperimentSpec, with no execution authority."""
from __future__ import annotations

from contextlib import nullcontext
import hashlib
import io
import json
import os
import re
import stat
import subprocess
import tarfile
import tempfile
from uuid import uuid4
from pathlib import Path, PurePosixPath, PureWindowsPath

from .experiment_spec import ExperimentSpec
from .screen_policy import canonical_fingerprint
from .v4_bundle import TEXT_SUFFIXES, _payload, build_v4_source_bundle, committed_payloads


PACKAGE_FORMAT = "molgap-experiment-source-package-v1"
GENERATED_PACKAGE_FORMAT = "molgap-experiment-source-package-v2"
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
    # A Python module under the reviewed package is source, even when its
    # responsibility is data/checkpoint IO. Directories and credential names
    # retain their existing confinement; artifact payload suffixes stay blocked.
    if ((value.startswith(("src/molgap/", "tests/")) or
         value == "platforms/kaggle/retrieve_family_outputs.py") and path.suffix == ".py"
            and path.name == path.name.lower()):
        io_tokens = {"data", "dataset", "datasets", "checkpoint", "checkpoints",
                     "records", "output", "outputs", "result", "results", "artifacts"}
        # This reviewed compatibility alias contains adapter code, not account
        # material. The explicit test distribution needs the host import too.
        if value == "src/molgap/kaggle_credentials.py":
            io_tokens.add("credentials")
        # Token model source is not an authentication payload. Credential-like
        # token names and directories remain blocked, even under the source root.
        leaf_tokens = set(re.split(r"[._-]+", lowered[-1]))
        if ("token" in leaf_tokens and path.stem not in {"token","tokens"} and
                not leaf_tokens & {"auth","access","refresh","bearer","api","private"}):
            io_tokens.add("token")
        lowered = [part for part in lowered if part not in io_tokens]
        directory_tokens = {token for part in lowered[:-1] for token in re.split(r"[._-]+", part)}
        filename_tokens = set(re.split(r"[._-]+", lowered[-1])) - io_tokens
        tokens = directory_tokens | filename_tokens
    # Versioned RML policy/schema JSON describes checkpoint semantics, not weights.
    if (len(path.parts) == 3 and path.parts[:2] in {
            ("research_memory", "policies"), ("research_memory", "schemas")}
            and path.suffix == ".json"):
        tokens.discard("checkpoint")
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


def read_packaged_text(package_dir: Path, relative_path: str, *, _verification=None) -> str:
    """Read a reviewed source member from an already verified immutable package."""
    name = _name(relative_path)
    if _verification is not None:
        # Re-read all six files before using the exact-content memo. A path or
        # unchanged stat record is never authority for cached archive members.
        _read_verified_package(package_dir, _verification=_verification)
        try:
            return _verification._members[name].decode("utf-8")
        except KeyError as exc:
            raise ValueError("Missing verified packaged text source") from exc
    with tarfile.open(Path(package_dir) / "source.tar.gz", "r:gz") as archive:
        members = [member for member in archive.getmembers() if member.name == name]
        if len(members) != 1 or not members[0].isfile() or members[0].linkname:
            raise ValueError("Expected one regular packaged text source")
        stream = archive.extractfile(members[0])
        if stream is None:
            raise ValueError("Missing packaged text source")
        with stream:
            return stream.read().decode("utf-8")


def initial_state_paths(initial_states: dict) -> dict[str, str]:
    """Bind byte-identical initializations to one deterministic staged file."""
    from .experiment_launch import _safe_local
    from .experiment_spec import _identifier
    from .training_reproducibility import sha256_file
    representatives, result = {}, {}
    for arm_id in sorted(initial_states):
        _identifier(arm_id, "arm_id")
        path = Path(initial_states[arm_id]).absolute()
        _safe_local(path)
        digest = sha256_file(path)
        result[arm_id] = representatives.setdefault(digest, f"initial_states/{arm_id}.pt")
    return result


def stage_source_inputs(package_dir: Path, destination: Path, initial_states: dict, *,
                        initial_paths=None, _verification=None) -> Path:
    """Common immutable worker inputs; no platform metadata or submission."""
    from .experiment_launch import _safe_local, publish_immutable_bytes
    from .experiment_launch_config import _relative
    from .experiment_spec import _identifier
    package_dir, destination = Path(package_dir).absolute(), Path(destination).absolute()
    raw, _ = _read_verified_package(package_dir, _verification=_verification)
    _safe_local(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for name in sorted(SIDECARS):
        target = "source_payload.bin" if name == "source.tar.gz" else name
        publish_immutable_bytes(destination / target, raw[name])
    paths = initial_state_paths(initial_states) if initial_paths is None else initial_paths
    if set(paths) != set(initial_states):
        raise ValueError("Staged initialization paths must cover exactly the supplied arms")
    for arm_id, path in initial_states.items():
        _identifier(arm_id, "arm_id")
        path = Path(path).absolute()
        _safe_local(path)
        relative = _relative(paths[arm_id], f"initial_paths[{arm_id}]")
        if not relative.startswith("initial_states/"):
            raise ValueError("Staged initialization must stay in initial_states")
        # Publishing a shared path again also rejects conflicting arm bytes.
        publish_immutable_bytes(destination / relative, path.read_bytes())
    return destination


def _no_links(path: Path) -> None:
    from .common.io import _safe_local
    _safe_local(path)


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


def _generation_bindings(generated_hashes, configuration_source, names):
    from .common.spec import _digest
    if (type(generated_hashes) is not dict or not generated_hashes or len(generated_hashes)>128 or
            type(configuration_source) is not dict or set(configuration_source)!={"path","sha256","compiler"}
            or configuration_source["compiler"]!="molgap-scientific-experiment-v1"):
        raise ValueError("Explicit scientific configuration provenance is required")
    config_name=_name(configuration_source["path"])
    if config_name not in names or config_name in generated_hashes or not config_name.endswith(".json"):
        raise ValueError("Configuration provenance must name committed source JSON")
    _digest(configuration_source["sha256"],"configuration_source.sha256")
    for name,digest in generated_hashes.items():
        _name(name);_digest(digest,"generated record sha256")
        if name not in names or not name.startswith("experiments/") or "/generated/" not in name or not name.endswith(".json"):
            raise ValueError("Derived records cannot replace executable Python or arbitrary files")
    return {"generated_records":dict(sorted(generated_hashes.items())),"configuration_source":dict(configuration_source)}


def _manifest(spec, names, commit, archive_hash, inventory_hash, *, generated_hashes=None,configuration_source=None):
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
    if generated_hashes is not None or configuration_source is not None:
        stable.update(format=GENERATED_PACKAGE_FORMAT,
            **_generation_bindings(generated_hashes,configuration_source,names))
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
    spec: ExperimentSpec, repo_root: Path, relative_paths, output_dir: Path, *, source_cache=None,
    generated_records=None,configuration_source=None, _verification=None,
) -> dict:
    """Publish a new source-only package, manifest last; never overwrite a package."""
    spec = _spec(spec)
    names = _allowlist(relative_paths)
    generated = {} if generated_records is None else dict(generated_records)
    if generated or configuration_source is not None:
        _generation_bindings({n:_sha(b) for n,b in generated.items()},configuration_source,names)
        if any(type(b) is not bytes or len(b)>1024**2 or _json_bytes(_load(b))!=b for b in generated.values()):
            raise ValueError("Derived JSON must be bounded and canonical")
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
    if generated:
        if _sha(_payload(_source(root,configuration_source["path"]))) != configuration_source["sha256"]:
            raise ValueError("Scientific configuration provenance bytes changed")
        if any(_payload(_source(root,n)) != b for n,b in generated.items()):
            raise ValueError("Generated configuration record bytes changed")
    bundle_options={"generated_records":generated} if generated else {}
    output.mkdir(parents=True, exist_ok=True)
    # Exclusive reservation also prevents two cooperating builders sharing an empty dir.
    lock = output / ".package-building"
    with lock.open("xb"):
        pass
    try:
        cache_session = nullcontext(None)
        if source_cache is not None:
            from .local_storage import source_cache_session
            cache_session = source_cache_session(source_cache)
        with tempfile.TemporaryDirectory(prefix=".source-stage-", dir=output) as stage_name, cache_session as cache_root:
            stage = Path(stage_name)
            if cache_root is None:
                bundle = build_v4_source_bundle(repo_root=root, relative_paths=names,
                                               output_dir=stage, source_commit=commit,**bundle_options)
            else:
                from .common.io import copy_immutable_file, publish_immutable_bytes
                from .v4_bundle import _assert_clean_paths
                if _assert_clean_paths(root, [n for n in names if n not in generated]) != commit:
                    raise ValueError("Source HEAD changed before cache lookup")
                committed = {**committed_payloads(root, commit, [n for n in names if n not in generated]),**generated}
                if any(_payload(_source(root,n)) != committed[n] for n in names):
                    raise ValueError("Repository source payload mismatch before cache lookup")
                # Commit provenance belongs to each package, not to reusable tar bytes.
                key = canonical_fingerprint({"builder":"v4-source-content-v1",
                    "files":{n:_sha(committed[n]) for n in sorted(names)}})
                cached = cache_root / key
                _no_links(cached)
                cache_names={"source.tar.gz", "SOURCE_COMMIT.txt", "SOURCE_ARCHIVE_SHA256.txt", "SOURCE_FILES.json"}
                if cached.exists() and {p.name for p in cached.iterdir()} != cache_names:
                    # Only this locked namespace owns these reproducible objects.
                    # Retain them, then rebuild from the already verified commit.
                    retained=cached.with_name(".incomplete-"+key+"-"+uuid4().hex)
                    _no_links(retained)
                    if retained.parent.resolve() != cached.parent.resolve():
                        raise ValueError("Cache recovery escaped its configured root")
                    cached.rename(retained)
                if not cached.exists():
                    cached.parent.mkdir(parents=True, exist_ok=True)
                    with tempfile.TemporaryDirectory(dir=cached.parent, prefix=".partial-"+key+"-") as temporary_root:
                        temporary = Path(temporary_root)
                        built=build_v4_source_bundle(repo_root=root, relative_paths=names,
                            output_dir=temporary, source_commit=commit,**bundle_options)
                        from .v4_runtime import validate_standard_source_bundle
                        validate_standard_source_bundle(temporary/"source.tar.gz",built["archive_sha256"],commit)
                        temporary.rename(cached)  # All readers/publishers share the OS lock.
                for name in ("source.tar.gz", "SOURCE_COMMIT.txt", "SOURCE_ARCHIVE_SHA256.txt", "SOURCE_FILES.json"):
                    _source(cached, name)
                cached_commit = (cached / "SOURCE_COMMIT.txt").read_bytes()
                if not re.fullmatch(rb"(?:[0-9a-f]{40}|[0-9a-f]{64})\n", cached_commit):
                    raise ValueError("Invalid cached source commit")
                inventory = {"format":"molgap-v4-source-inventory-v1",
                    "source_commit":cached_commit.decode("ascii").removesuffix("\n"),
                    "files":[{"path":n,"sha256":_sha(committed[n]),"bytes":len(committed[n])}
                             for n in sorted(names)]}
                if (cached / "SOURCE_FILES.json").read_bytes() != (json.dumps(inventory, indent=2)+"\n").encode():
                    raise ValueError("Cached source inventory payload or commit mismatch")
                cached_hash = (cached / "SOURCE_ARCHIVE_SHA256.txt").read_bytes()
                if not re.fullmatch(rb"[0-9a-f]{64}\n", cached_hash):
                    raise ValueError("Invalid cached archive hash")
                archive_hash = cached_hash.decode("ascii").removesuffix("\n")
                copy_immutable_file(cached / "source.tar.gz",stage / "source.tar.gz",expected_sha256=archive_hash)
                # Rebind only after checking the cache's original metadata. The
                # ordinary package verifier below checks every archived payload.
                inventory["source_commit"] = commit
                for name, contents in (
                    ("SOURCE_COMMIT.txt", (commit+"\n").encode()),
                    ("SOURCE_ARCHIVE_SHA256.txt", cached_hash),
                    ("SOURCE_FILES.json", (json.dumps(inventory, indent=2)+"\n").encode()),
                ):
                    publish_immutable_bytes(stage / name, contents)
                try:
                    os.utime(cached, None)
                except OSError:
                    pass  # Access-time bookkeeping cannot invalidate the independent copy.
                bundle = {"archive_sha256":archive_hash}
            manifest = _manifest(spec, names, commit, bundle["archive_sha256"],
                _sha((stage / "SOURCE_FILES.json").read_bytes()),
                **({"generated_hashes":{n:_sha(b) for n,b in generated.items()},
                    "configuration_source":configuration_source} if generated else {}))
            _atomic(stage / "experiment_spec.json", spec.to_json().encode("utf-8"))
            _atomic(stage / "package_manifest.json", _json_bytes(manifest))
            # Recheck commit blobs too: Git clean filters must not conceal changed bytes.
            verify_experiment_source_package(stage, repo_root=root,
                **({"_verification": _verification} if _verification is not None else {}))
            if _git(root, "rev-parse", "HEAD").decode().strip() != commit:
                raise ValueError("Source HEAD changed before package publication")
            for name in sorted(SIDECARS - {"package_manifest.json"}):
                os.replace(stage / name, output / name)
        _atomic(output / "package_manifest.json", _json_bytes(manifest))
        return manifest
    finally:
        lock.unlink(missing_ok=True)


class _PackageVerification:
    """One synchronous preparation's exact-content validation memo.

    Paths and stat metadata never establish a hit. Repository comparisons are
    always fresh, and callers cannot mutate the cached canonical result.
    """

    def __init__(self):
        self._fingerprint = None
        self._manifest = None
        self._members = {}

    def verify(self, raw, repo_root=None):
        fingerprint = tuple((name, _sha(raw[name])) for name in sorted(SIDECARS))
        if repo_root is not None or fingerprint != self._fingerprint:
            members = {}
            manifest = _verify_package_snapshot(raw, repo_root, _members=members)
            self._fingerprint = fingerprint
            self._manifest = _json_bytes(manifest)
            self._members = members
        return _load(self._manifest)


def _read_verified_package(package_dir, repo_root=None, *, _verification=None):
    directory = Path(package_dir).absolute()
    _no_links(directory)
    if not directory.is_dir() or {p.name for p in directory.iterdir()} != SIDECARS:
        raise ValueError("Package must contain exactly the six declared files")
    for name in SIDECARS:
        _source(directory, name)
    raw = {name: (directory / name).read_bytes() for name in SIDECARS}
    if _verification is not None:
        if type(_verification) is not _PackageVerification:
            raise TypeError("Expected a scoped package verification memo")
        manifest = _verification.verify(raw, repo_root)
    else:
        manifest = _verify_package_snapshot(raw, repo_root)
    return raw, manifest


def verify_experiment_source_package(package_dir: Path, repo_root: Path | None = None, *,
                                     _verification=None) -> dict:
    """Verify source integrity only, optionally against commit blobs and tracked payloads."""
    return _read_verified_package(package_dir, repo_root, _verification=_verification)[1]


def _verify_package_snapshot(raw, repo_root=None, *, _members=None):
    """Validate the exact bytes read by the package owner, never reopen the archive."""
    manifest = _load(raw["package_manifest.json"])
    if type(manifest) is not dict or raw["package_manifest.json"] != _json_bytes(manifest):
        raise ValueError("Manifest must be a canonical JSON object")
    spec = ExperimentSpec.from_frozen_json(raw["experiment_spec.json"].decode("utf-8"))
    if raw["experiment_spec.json"] != spec.to_json().encode("utf-8"):
        raise ValueError("Spec sidecar must be the canonical snapshot")
    names = _allowlist(manifest.get("relative_allowlist", []))
    commit = raw["SOURCE_COMMIT.txt"].decode("ascii").removesuffix("\n")
    if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", commit):
        raise ValueError("Invalid source commit")
    archive_hash = _sha(raw["source.tar.gz"])
    generation=({"generated_hashes":manifest.get("generated_records"),"configuration_source":manifest.get("configuration_source")}
                if manifest.get("format")==GENERATED_PACKAGE_FORMAT else {})
    expected = _manifest(spec, names, commit, archive_hash, _sha(raw["SOURCE_FILES.json"]),**generation)
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
    config_name = manifest.get("configuration_source", {}).get("path")
    config_payload = None
    # Never extract untrusted tar members onto the filesystem.
    # Parse the exact snapshot whose hash was checked, without reopening the file.
    with tarfile.open(fileobj=io.BytesIO(raw["source.tar.gz"]), mode="r:gz") as archive:
        for member in archive:
            name = _name(member.name)
            if not member.isfile() or member.linkname or name in seen or name not in entries:
                raise ValueError("Unsafe, duplicate, or unexpected archive member")
            entry = entries[name]
            if member.size != entry["bytes"]:
                raise ValueError("Archive member size mismatch")
            stream = archive.extractfile(member)
            payload = None if stream is None else stream.read()
            if payload is None or _sha(payload) != entry["sha256"]:
                raise ValueError("Archive member payload mismatch")
            if name in manifest.get("generated_records", {}):
                if len(payload) > 1024**2 or _json_bytes(_load(payload)) != payload:
                    raise ValueError("Derived JSON must be bounded and canonical")
            if name == config_name:
                config_payload = payload
            if _members is not None:
                _members[name] = payload
            seen.add(name)
    if seen != set(entries):
        raise ValueError("Archive member set differs from inventory")
    generated=manifest.get("generated_records",{})
    if generated:
        if any(entries[n]["sha256"]!=digest or entries[n]["bytes"]>1024**2 for n,digest in generated.items()):
            raise ValueError("Derived configuration inventory differs from its provenance")
        config=manifest["configuration_source"]
        if entries[config["path"]]["sha256"] != config["sha256"]:
            raise ValueError("Configuration provenance differs from archived source")
        config_value = _load(config_payload.decode("utf-8"))
        if type(config_value) is not dict or config_value.get("format") != config["compiler"]:
            raise ValueError("Configuration provenance is not a scientific input")
        prefix = "experiments/" + spec.to_dict()["experiment_id"] + "/generated/" + canonical_fingerprint({
            "input": config["sha256"], "commit": commit}) + "/"
        if config_value.get("experiment_id") != spec.to_dict()["experiment_id"] or any(
                not n.startswith(prefix) for n in generated):
            raise ValueError("Derived records differ from their configuration/source namespace")
    if repo_root is not None:
        root = Path(repo_root).absolute()
        _no_links(root)
        root = root.resolve()
        tracked = set(_git(root, "ls-files", "-z").decode("utf-8").split("\0"))
        committed = committed_payloads(root, commit, [n for n in entries if n not in generated])
        for name, entry in entries.items():
            if name not in tracked and name not in generated:
                raise ValueError(f"Source no longer tracked: {name}")
            path = _source(root, name)
            payload = committed[name] if name not in generated else _payload(_source(root,name))
            if any(len(data) != entry["bytes"] or _sha(data) != entry["sha256"]
                   for data in (payload, _payload(path))):
                raise ValueError(f"Repository source payload mismatch: {name}")
    return manifest


def inspect_experiment_source_package(package_dir: Path, repo_root: Path | None = None) -> dict:
    """Inspection is fail-closed verification, never a readiness decision."""
    return verify_experiment_source_package(package_dir, repo_root)
