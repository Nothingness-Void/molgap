"""Boundary tests, not model tests. Synthetic inputs never produce real-shard success."""
import copy
import io
import json
import os
from pathlib import Path
import tarfile

import pytest

from molgap import experiment_preflight as pf
from molgap.experiment_spec import ExperimentSpec
from molgap.screen_policy import canonical_fingerprint
from test_experiment_spec import payload
from test_experiment_package import repo, package


def receipt(package):
    return json.loads((package / "package_manifest.json").read_bytes())


def run(package, payload, tmp_path, **kwargs):
    return pf.run_experiment_preflight(
        ExperimentSpec(payload), package, tmp_path / "output",
        expected_package_identity=receipt(package)["package_identity"], **kwargs)


@pytest.fixture
def manifest(payload, tmp_path):
    spec = ExperimentSpec(payload)
    arm = payload["arms"][0]
    value = {"format": pf.MANIFEST_FORMAT, "spec_identity": spec.identity, "arms": [{
        "arm_id": arm["arm_id"], "arm_identity": canonical_fingerprint(arm),
        "data": copy.deepcopy(arm["data"]), "kind": "real-packed-pcqm",
        "fixed_manifest": {"path": "fixed.json", "sha256": "1" * 64},
        "files": [{"path": f"{role}.pt", "sha256": "2" * 64, "bytes": 5, "role": role}
                  for role in ("train", "development")],
    }]}
    path = tmp_path / "manifest.json"
    path.write_bytes(pf._canonical(value))
    return value, path


def validate(payload, value, path):
    raw = pf._canonical(value)
    path.write_bytes(raw)
    return pf.validate_real_shard_manifest(ExperimentSpec(payload), path, pf._sha(raw))


def test_missing_real_and_k1_are_not_pass(package, payload, tmp_path):
    result = run(package, payload, tmp_path)
    assert [a["status"] for a in result["arms"]] == [
        "MISSING_REAL_SHARD", "UNSUPPORTED_FAMILY_PREFLIGHT"]
    assert result["status"] == "MIXED_NONPASS"
    for arm in result["arms"]:
        assert arm["package_verified"]
        assert arm["device"] is None
        assert all(arm[key] is False for key in pf.CHECKS[1:])
    assert result["arms"][0]["arm_identity"] != result["arms"][1]["arm_identity"]


@pytest.mark.parametrize("mutation", [
    lambda m: m.update(format="unknown"),
    lambda m: m.update(spec_identity="0" * 64),
    lambda m: m.update(ready_for_desktop=True),
    lambda m: m["arms"].append(copy.deepcopy(m["arms"][0])),
    lambda m: m["arms"][0].update(arm_identity="0" * 64),
    lambda m: m["arms"][0].update(kind="synthetic"),
    lambda m: m["arms"][0]["data"].update(target="experimental"),
    lambda m: m["arms"][0]["data"]["dataset"].update(sha256="0" * 64),
    lambda m: m["arms"][0]["data"]["split"].update(sha256="0" * 64),
    lambda m: m["arms"][0]["data"].update(feature_sha256="0" * 64),
    lambda m: m["arms"][0]["data"]["roles"][0].update(usage_sha256="0" * 64),
    lambda m: m["arms"][0]["files"][0].update(role="official_validation"),
    lambda m: m["arms"][0]["files"][0].update(role="test-dev"),
    lambda m: m["arms"][0]["files"][0].update(bytes=True),
    lambda m: m["arms"][0]["files"][0].update(sha256="not-a-digest"),
    lambda m: m["arms"][0]["files"].pop(),
    lambda m: m["arms"][0]["files"].append(copy.deepcopy(m["arms"][0]["files"][0])),
])
def test_strict_manifest_failures(payload, manifest, mutation):
    value, path = manifest
    mutation(value)
    with pytest.raises(ValueError):
        validate(payload, value, path)


@pytest.mark.parametrize("name", ["../escape", "/absolute", "C:/outside", "a\\b", "a/../b",
                                  "a//b", "./b", "NUL.pt", "x:stream", "x. "])
def test_manifest_paths(payload, manifest, name):
    value, path = manifest
    value["arms"][0]["files"][0]["path"] = name
    with pytest.raises(ValueError):
        validate(payload, value, path)


def test_manifest_declaration_is_not_real_verification(payload, manifest):
    value, path = manifest
    parsed = validate(payload, value, path)
    assert "status" not in parsed
    assert "shard_verified" not in parsed


@pytest.mark.parametrize("raw", [b'{"x":1,"x":1}', b'{ "x": 1 }', b'{"x":NaN}', b'{}\n'])
def test_noncanonical_rejected(payload, tmp_path, raw):
    path = tmp_path / "bad.json"
    path.write_bytes(raw)
    with pytest.raises(ValueError):
        pf.validate_real_shard_manifest(ExperimentSpec(payload), path, pf._sha(raw))


def test_manifest_tamper(package, payload, manifest, tmp_path):
    value, path = manifest
    digest = pf._file_sha(path)
    path.write_bytes(path.read_bytes() + b" ")
    result = run(package, payload, tmp_path, shard_manifest=path,
                 expected_shard_manifest_sha256=digest)
    assert result["status"] == "SHARD_MANIFEST_INVALID"
    assert not any(a["shard_verified"] for a in result["arms"])


@pytest.mark.parametrize("name", ["source.tar.gz", "experiment_spec.json", "SOURCE_FILES.json",
                                  "SOURCE_COMMIT.txt", "SOURCE_ARCHIVE_SHA256.txt"])
def test_package_tamper(package, payload, tmp_path, name):
    target = package / name
    target.write_bytes(target.read_bytes() + b"tamper")
    result = run(package, payload, tmp_path)
    assert result["status"] == "PACKAGE_INVALID"
    assert all(not a["package_verified"] for a in result["arms"])


def test_pinned_package_identity(package, payload, tmp_path):
    result = pf.run_experiment_preflight(ExperimentSpec(payload), package, tmp_path / "output",
                                          expected_package_identity="0" * 64)
    assert result["status"] == "PACKAGE_INVALID"


@pytest.mark.parametrize("mode", ["missing", "hash", "size", "escape"])
def test_shard_staging_rejects(manifest, tmp_path, mode):
    value, _ = manifest
    entry = value["arms"][0]
    root, target = tmp_path / "data", tmp_path / "stage"
    root.mkdir()
    target.mkdir()
    for item in [entry["fixed_manifest"], *entry["files"]]:
        (root / item["path"]).write_bytes(b"fake!")
        item["sha256"] = pf._sha(b"fake!")
    if mode == "missing":
        (root / "train.pt").unlink()
    elif mode == "hash":
        (root / "train.pt").write_bytes(b"wrong")
    elif mode == "size":
        entry["files"][0]["bytes"] += 1
    else:
        entry["files"][0]["path"] = "../outside.pt"
    with pytest.raises((ValueError, FileNotFoundError)):
        pf._stage_files(entry, root, target)


@pytest.mark.parametrize("kind", [tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.CHRTYPE,
                                  tarfile.BLKTYPE, tarfile.FIFOTYPE, tarfile.DIRTYPE, "duplicate", "escape"])
def test_extraction_rejects_unsafe_members(tmp_path, kind):
    package, root = tmp_path / "package", tmp_path / "source"
    package.mkdir()
    root.mkdir()
    with tarfile.open(package / "source.tar.gz", "w:gz") as archive:
        member = tarfile.TarInfo("../escape" if kind == "escape" else "src/file.py")
        if kind in {"duplicate", "escape"}:
            member.size = 1
            archive.addfile(member, io.BytesIO(b"x"))
            if kind == "duplicate":
                archive.addfile(member, io.BytesIO(b"x"))
        else:
            member.type = kind
            archive.addfile(member)
    with pytest.raises(ValueError):
        pf._unpack(package, root)
    assert not (tmp_path / "escape").exists()


def test_host_origin_pollution_rejected(tmp_path):
    # The pytest interpreter already imported the host package. It must not be
    # usable as a family execution interpreter, even with a valid root argument.
    with pytest.raises(ImportError, match="Host import pollution"):
        pf._origins(tmp_path)
    with pytest.raises(ImportError, match="Bootstrap already"):
        pf._worker({"source_root": str(tmp_path)})
    with pytest.raises(ImportError):
        pf._PackageOnly(tmp_path).find_spec("molgap")


def test_unpacked_origin_in_fresh_interpreter_rejects_fake_real_manifest(tmp_path, manifest, monkeypatch):
    root = tmp_path / "source"
    root.mkdir()
    archive_dir = tmp_path / "archive"
    archive_dir.mkdir()
    stub = (b'MANIFEST_SHA256 = "0" * 64\n'
            b'def forbidden(*a, **kw): raise AssertionError("synthetic loader must never run")\n'
            b'validate_fixed_assets = _load_datasets = _training_loader = _development_loader = forbidden\n')
    with tarfile.open(archive_dir / "source.tar.gz", "w:gz") as archive:
        for name, raw in {"src/molgap/__init__.py": b"", "src/molgap/pcqm_gptrans_v4.py": stub}.items():
            member = tarfile.TarInfo(name)
            member.size = len(raw)
            archive.addfile(member, io.BytesIO(raw))
    pf._unpack(archive_dir, root)
    host = tmp_path / "host/molgap"
    host.mkdir(parents=True)
    (host / "__init__.py").write_text('raise AssertionError("host contamination")\n', encoding="ascii")
    monkeypatch.setenv("PYTHONPATH", str(host.parent))
    data = tmp_path / "data"
    data.mkdir()
    (data / "fixed.json").write_bytes(b"synthetic, not real")
    stage = tmp_path / "worker"
    stage.mkdir()
    result = pf._launch(root, data, manifest[0]["arms"][0], 30, stage)
    assert result["status"] == "LOADER_FAILED"
    assert result["shard_verified"] is False
    assert result["loader_batch_built"] is False
    assert result["import_origins"]["molgap.pcqm_gptrans_v4"] == "src/molgap/pcqm_gptrans_v4.py"
    assert "frozen real PCQM" in result["error"]["message"]


def test_worker_missing_frozen_loader_is_unsupported(tmp_path, manifest):
    root = tmp_path / "source"
    module = root / "src/molgap"
    module.mkdir(parents=True)
    (module / "__init__.py").write_text("", encoding="ascii")
    (module / "pcqm_gptrans_v4.py").write_text("", encoding="ascii")
    stage = tmp_path / "worker"
    stage.mkdir()
    result = pf._launch(root, tmp_path / "missing-data", manifest[0]["arms"][0], 30, stage)
    assert result["status"] == "UNSUPPORTED_FAMILY_PREFLIGHT"
    assert result["loader_batch_built"] is False


def test_existing_output_rejected(package, payload, tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    with pytest.raises(ValueError, match="Output must be new"):
        run(package, payload, tmp_path)
    assert list(output.iterdir()) == []


def test_output_cannot_modify_package(package, payload):
    with pytest.raises(ValueError, match="overlaps"):
        pf.run_experiment_preflight(ExperimentSpec(payload), package, package / "output",
                                     expected_package_identity=receipt(package)["package_identity"])
    assert not (package / "output").exists()


def test_canonical_reports_no_authority(package, payload, tmp_path):
    result = run(package, payload, tmp_path)
    for path in (tmp_path / "output").rglob("*.json"):
        raw = path.read_bytes()
        assert raw == pf._canonical(json.loads(raw))
        def check(value):
            if isinstance(value, dict):
                assert not any(token in key.lower() for key in value for token in ("rml", "ready", "replay", "authority"))
                for item in value.values():
                    check(item)
            elif isinstance(value, list):
                for item in value:
                    check(item)
        check(json.loads(raw))
    assert json.loads((tmp_path / "output/preflight.json").read_bytes()) == result
    assert not list((tmp_path / "output").rglob(".preflight-*"))


def test_atomic_replace_failure_preserves_old_bytes(tmp_path, monkeypatch):
    target = tmp_path / "report.json"
    target.write_bytes(b"old")
    def fail(source, destination):
        assert Path(source).parent == target.parent
        assert pf._load(Path(source).read_bytes()) == {"new": True}
        raise OSError("replacement failed")
    monkeypatch.setattr(pf.os, "replace", fail)
    with pytest.raises(OSError):
        pf._atomic(target, {"new": True})
    assert target.read_bytes() == b"old"
    assert list(tmp_path.iterdir()) == [target]


@pytest.fixture
def real_inputs():
    """Only external, explicitly authorized real assets can reach loader success."""
    config = os.environ.get("MOLGAP_PREFLIGHT_REAL_INPUTS")
    if not config:
        pytest.skip("No explicitly authorized real package/shards; never substitute synthetic data")
    inputs = json.loads(Path(config).read_bytes())
    spec = ExperimentSpec.from_json(Path(inputs["spec"]).read_text(encoding="utf-8"))
    return inputs, spec


def test_real_dual_arm_independent_success_failure(real_inputs, tmp_path):
    inputs, spec = real_inputs
    arms = spec.to_dict()["arms"]
    assert len(arms) == 2 and all(a["family"]["name"] == "gptrans_t" for a in arms)
    manifest = pf.validate_real_shard_manifest(spec, inputs["shard_manifest"],
                                              inputs["expected_shard_manifest_sha256"])
    second = next(a for a in manifest["arms"] if a["arm_id"] == arms[1]["arm_id"])
    second["files"][0]["path"] = "deliberately-missing-real-shard.pt"
    assert not (Path(inputs["shard_root"]) / second["files"][0]["path"]).exists()
    manifest_path = tmp_path / "authorized-negative-case.json"
    manifest_path.write_bytes(pf._canonical(manifest))
    result = pf.run_experiment_preflight(
        spec, inputs["package_dir"], tmp_path / "output",
        expected_package_identity=inputs["expected_package_identity"],
        shard_manifest=manifest_path, shard_root=inputs["shard_root"],
        expected_shard_manifest_sha256=pf._file_sha(manifest_path), timeout_seconds=600)
    first, second = result["arms"]
    assert first["status"] == "LOADER_VERIFIED_ONLY"
    assert first["shard_verified"] and first["loader_batch_built"]
    assert second["status"] == "MISSING_REAL_SHARD"
    assert not second["shard_verified"] and not second["loader_batch_built"]
    assert result["status"] == "MIXED_NONPASS"
    for arm in result["arms"]:
        assert all(arm[key] is False for key in pf.CHECKS[3:])
