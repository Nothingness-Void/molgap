"""Synthetic source-only package tests. No model, dataset, or remote execution."""
import hashlib
import io
import json
import subprocess
import tarfile

import pytest

from molgap.experiment_package import (
    PACKAGE_FORMAT, build_experiment_source_package,
    inspect_experiment_source_package, verify_experiment_source_package,
)
from molgap.experiment_spec import ExperimentSpec
from molgap.screen_policy import canonical_fingerprint
from test_experiment_spec import payload  # Reuse the existing synthetic fixture.


def git(root, *args):
    return subprocess.run(["git", *args], cwd=root, check=True,
                          capture_output=True).stdout.decode().strip()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def dump(path, value):
    path.write_bytes(json.dumps(value, sort_keys=True, separators=(",", ":"),
                               ensure_ascii=True).encode())


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init")
    git(root, "config", "user.email", "synthetic@example.invalid")
    git(root, "config", "user.name", "Synthetic fixture")
    git(root, "config", "core.autocrlf", "false")
    (root / "src").mkdir()
    (root / "src" / "example.py").write_bytes(b"VALUE = 1\r\n")
    (root / "README.md").write_bytes(b"Synthetic source\n")
    git(root, "add", ".")
    git(root, "-c", "commit.gpgsign=false", "commit", "-m", "Synthetic source")
    return root


@pytest.fixture
def package(repo, payload, tmp_path):
    target = tmp_path / "package"
    build_experiment_source_package(ExperimentSpec(payload), repo,
                                    ["src/example.py", "README.md"], target)
    return target


def resign(package):
    """Update outer hashes so malformed inner structures reach safety checks."""
    manifest_path = package / "package_manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["archive_sha256"] = sha((package / "source.tar.gz").read_bytes())
    manifest["inventory_sha256"] = sha((package / "SOURCE_FILES.json").read_bytes())
    (package / "SOURCE_ARCHIVE_SHA256.txt").write_bytes(
        (manifest["archive_sha256"] + "\n").encode())
    manifest.pop("package_identity")
    manifest["package_identity"] = canonical_fingerprint(manifest)
    dump(manifest_path, manifest)


def test_binding_and_reproducibility(package, repo, payload, tmp_path):
    manifest = inspect_experiment_source_package(package, repo)
    other = tmp_path / "another-location"
    rebuilt = build_experiment_source_package(ExperimentSpec(payload), repo,
                                             ["src/example.py", "README.md"], other)
    assert manifest == rebuilt == verify_experiment_source_package(other)
    assert manifest["format"] == PACKAGE_FORMAT
    assert manifest["spec_identity"] == ExperimentSpec(payload).identity
    assert manifest["spec_sha256"] == sha(ExperimentSpec(payload).to_json().encode())
    assert manifest["relative_allowlist"] == ["src/example.py", "README.md"]
    assert manifest["arms"] == [
        {"arm_id": arm["arm_id"], "identity": canonical_fingerprint(arm)}
        for arm in payload["arms"]
    ]
    assert (package / "source.tar.gz").read_bytes() == (other / "source.tar.gz").read_bytes()
    with tarfile.open(package / "source.tar.gz") as archive:
        assert archive.getnames() == ["README.md", "src/example.py"]
        assert archive.extractfile("src/example.py").read() == b"VALUE = 1\n"
    identity_fields = {key: value for key, value in manifest.items() if key != "package_identity"}
    assert manifest["package_identity"] == canonical_fingerprint(identity_fields)


@pytest.mark.parametrize("sidecar", [
    "experiment_spec.json", "SOURCE_FILES.json", "source.tar.gz",
    "SOURCE_COMMIT.txt", "SOURCE_ARCHIVE_SHA256.txt", "package_manifest.json",
])
def test_any_sidecar_tampering_rejected(package, sidecar):
    path = package / sidecar
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises((ValueError, UnicodeError)):
        verify_experiment_source_package(package)


@pytest.mark.parametrize("change", ["spec", "arm", "unknown_arm", "manifest", "authority"])
def test_declaration_tampering_rejected(package, change):
    path = package / ("package_manifest.json" if change in {"manifest", "authority"}
                      else "experiment_spec.json")
    value = json.loads(path.read_bytes())
    if change == "spec":
        value["logical_run_id"] = "changed"
    elif change == "arm":
        value["arms"][0]["arm_id"] = "changed"
    elif change == "unknown_arm":
        value["arms"][0]["family"]["name"] = "unknown"
    elif change == "manifest":
        value["arms"][0]["identity"] = "0" * 64
    else:
        value["replay_ready"] = True
    dump(path, value)
    with pytest.raises(ValueError):
        verify_experiment_source_package(package)


def test_exact_spec_and_forged_snapshot_rejected(repo, payload, tmp_path):
    class Derived(ExperimentSpec):
        pass

    for spec in (payload, Derived(payload)):
        with pytest.raises(TypeError):
            build_experiment_source_package(spec, repo, ["README.md"], tmp_path / "bad")
    spec = ExperimentSpec(payload)
    object.__setattr__(spec, "_canonical_json", json.dumps(payload, indent=2))
    with pytest.raises(ValueError):
        build_experiment_source_package(spec, repo, ["README.md"], tmp_path / "bad")
    payload["arms"][0]["family"]["name"] = "unknown"
    object.__setattr__(spec, "_canonical_json", json.dumps(payload))
    with pytest.raises(ValueError):
        build_experiment_source_package(spec, repo, ["README.md"], tmp_path / "bad")


@pytest.mark.parametrize("names", [
    [], [""], ["../README.md"], ["src/../../README.md"], ["/README.md"],
    ["C:/README.md"], ["C:README.md"], ["src\\example.py"], ["src"],
    ["README.md", "README.md"], ["./README.md"], ["src//example.py"],
    ["credentials.json"], ["keys/private.pem"], ["datasets/sample.json"],
    ["checkpoints/model.pt"], ["platforms/_records/source.py"],
    ["results/report.md"], [".env"], ["source.tar.gz"],
    ["experiment_spec.json"], ["package_manifest.json"],
])
def test_invalid_paths_rejected(repo, payload, tmp_path, names):
    with pytest.raises((ValueError, RuntimeError)):
        build_experiment_source_package(ExperimentSpec(payload), repo, names, tmp_path / "bad")


@pytest.mark.parametrize("state", ["untracked", "dirty", "staged"])
def test_uncommitted_source_rejected(repo, payload, tmp_path, state):
    name = "new.py" if state == "untracked" else "src/example.py"
    (repo / name).write_bytes(b"changed\n")
    if state == "staged":
        git(repo, "add", name)
    with pytest.raises(RuntimeError):
        build_experiment_source_package(ExperimentSpec(payload), repo, [name], tmp_path / "bad")


def test_nonempty_and_overlapping_output_rejected(package, repo, payload):
    for output in (package, repo, repo / "src", repo / "README.md"):
        with pytest.raises(ValueError):
            build_experiment_source_package(ExperimentSpec(payload), repo,
                                            ["src/example.py", "README.md"], output)


def test_existing_empty_output_accepted(repo, payload, tmp_path):
    output = tmp_path / "empty"
    output.mkdir()
    build_experiment_source_package(ExperimentSpec(payload), repo, ["README.md"], output)
    assert verify_experiment_source_package(output)["package_status"] == "source_packaged"


def test_optional_repo_verification_does_not_require_head(package, repo):
    (repo / "unrelated.md").write_bytes(b"Unrelated commit\n")
    git(repo, "add", "unrelated.md")
    git(repo, "-c", "commit.gpgsign=false", "commit", "-m", "Unrelated change")
    manifest = verify_experiment_source_package(package, repo)
    assert manifest["source_commit"] != git(repo, "rev-parse", "HEAD")
    (repo / "README.md").write_bytes(b"changed\n")
    verify_experiment_source_package(package)
    with pytest.raises(ValueError, match="payload mismatch"):
        verify_experiment_source_package(package, repo)


def test_commit_blob_is_checked(package, repo):
    original = (repo / "README.md").read_bytes()
    (repo / "README.md").write_bytes(b"changed commit\n")
    git(repo, "add", "README.md")
    git(repo, "-c", "commit.gpgsign=false", "commit", "-m", "Changed source")
    commit = git(repo, "rev-parse", "HEAD")
    (repo / "README.md").write_bytes(original)
    (package / "SOURCE_COMMIT.txt").write_bytes((commit + "\n").encode())
    inventory = json.loads((package / "SOURCE_FILES.json").read_bytes())
    inventory["source_commit"] = commit
    dump(package / "SOURCE_FILES.json", inventory)
    manifest = json.loads((package / "package_manifest.json").read_bytes())
    manifest["source_commit"] = commit
    dump(package / "package_manifest.json", manifest)
    resign(package)
    with pytest.raises(ValueError, match="payload mismatch"):
        verify_experiment_source_package(package, repo)


@pytest.mark.parametrize("kind", ["traversal", "absolute", "symlink", "hardlink", "device",
                                  "directory", "missing", "extra", "duplicate", "payload"])
def test_archive_safety_even_with_recomputed_outer_hashes(package, kind):
    members = [("README.md", b"Synthetic source\n"), ("src/example.py", b"VALUE = 1\n")]
    with tarfile.open(package / "source.tar.gz", "w:gz") as archive:
        for name, data in members:
            if kind == "missing" and name == "README.md":
                continue
            info = tarfile.TarInfo(name)
            if name == "README.md":
                if kind == "traversal":
                    info.name = "../README.md"
                elif kind == "absolute":
                    info.name = "/README.md"
                elif kind in {"symlink", "hardlink", "device", "directory"}:
                    info.type = {"symlink": tarfile.SYMTYPE, "hardlink": tarfile.LNKTYPE,
                                 "device": tarfile.CHRTYPE, "directory": tarfile.DIRTYPE}[kind]
                    if kind in {"symlink", "hardlink"}:
                        info.linkname = "src/example.py"
                elif kind == "payload":
                    data = b"X" * len(data)
            info.size = len(data) if info.isfile() else 0
            archive.addfile(info, io.BytesIO(data) if info.isfile() else None)
        if kind in {"extra", "duplicate"}:
            archive.addfile(tarfile.TarInfo("extra.py" if kind == "extra" else "README.md"))
    resign(package)
    with pytest.raises(ValueError):
        verify_experiment_source_package(package)


def test_inventory_mismatch_with_recomputed_identity(package):
    path = package / "SOURCE_FILES.json"
    value = json.loads(path.read_bytes())
    value["files"].pop()
    dump(path, value)
    resign(package)
    with pytest.raises(ValueError, match="Inventory"):
        verify_experiment_source_package(package)


@pytest.mark.parametrize("change", ["duplicate", "hash", "size", "boolean_size"])
def test_inventory_entry_tampering_even_with_recomputed_identity(package, change):
    path = package / "SOURCE_FILES.json"
    value = json.loads(path.read_bytes())
    if change == "duplicate":
        value["files"].append(value["files"][0].copy())
    elif change == "hash":
        value["files"][0]["sha256"] = "0" * 64
    elif change == "size":
        value["files"][0]["bytes"] += 1
    else:
        value["files"][0]["bytes"] = True
    dump(path, value)
    resign(package)
    with pytest.raises(ValueError):
        verify_experiment_source_package(package)


def test_duplicate_manifest_keys_rejected(package):
    path = package / "package_manifest.json"
    original = path.read_bytes()
    path.write_bytes(b'{"format":"duplicate",' + original[1:])
    with pytest.raises(ValueError, match="Duplicate JSON"):
        verify_experiment_source_package(package)


def test_symlinks_rejected(repo, payload, tmp_path, package):
    link = repo / "alias.py"
    try:
        link.symlink_to(repo / "src" / "example.py")
    except OSError:
        pytest.skip("Host does not permit creating symlinks")
    with pytest.raises(ValueError, match="Symlink"):
        build_experiment_source_package(ExperimentSpec(payload), repo, ["alias.py"], tmp_path / "bad")
    directory_link = repo / "alias_dir"
    directory_link.symlink_to(repo / "src", target_is_directory=True)
    with pytest.raises(ValueError, match="Symlink"):
        build_experiment_source_package(ExperimentSpec(payload), repo,
                                        ["alias_dir/example.py"], tmp_path / "bad")
    sidecar = package / "SOURCE_COMMIT.txt"
    copy = tmp_path / "commit.txt"
    copy.write_bytes(sidecar.read_bytes())
    sidecar.unlink()
    sidecar.symlink_to(copy)
    with pytest.raises(ValueError, match="Symlink"):
        verify_experiment_source_package(package)


def test_failure_never_publishes_complete_manifest(repo, payload, tmp_path, monkeypatch):
    import molgap.experiment_package as implementation

    def fail(**kwargs):
        (kwargs["output_dir"] / "source.tar.gz").write_bytes(b"partial")
        raise OSError("Synthetic interruption")

    monkeypatch.setattr(implementation, "build_v4_source_bundle", fail)
    output = tmp_path / "interrupted"
    with pytest.raises(OSError, match="interruption"):
        build_experiment_source_package(ExperimentSpec(payload), repo, ["README.md"], output)
    assert not (output / "package_manifest.json").exists()


def test_no_runtime_or_evidence_authority(package):
    manifest = verify_experiment_source_package(package)
    assert manifest["package_status"] == "source_packaged"
    assert not ({"ready", "replay_ready", "training_success", "cost_measured", "rml_closed"}
                & manifest.keys())
