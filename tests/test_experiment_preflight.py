"""Boundary tests, not model tests. Synthetic inputs never produce real-shard success."""
import copy
import io
import json
import os
from pathlib import Path
import tarfile

import pytest

from molgap import experiment_preflight as pf
from molgap.experiment_package import build_experiment_source_package
from molgap.experiment_spec import ExperimentSpec
from molgap.screen_policy import canonical_fingerprint
from test_experiment_spec import addon, payload
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


def k1_authorization(payload):
    spec = ExperimentSpec(payload)
    arm = next(arm for arm in payload["arms"] if arm["family"] == pf.K1_FAMILY)
    return {"format": pf.MANIFEST_FORMAT, "spec_identity": spec.identity, "arms": [{
        "arm_id": arm["arm_id"], "arm_identity": canonical_fingerprint(arm),
        "data": copy.deepcopy(arm["data"]), "kind": "real-packed-pcqm",
        "fixed_manifest": {"path": "full.json", "sha256": "1" * 64},
        "files": [{"path": "store/topology/train.pt", "sha256": "2" * 64,
                   "bytes": 5, "rows": 50_000, "role": "train"}],
    }]}


def mocked_arm_reports(package, payload, manifest, tmp_path, monkeypatch, statuses, *, mode=pf.LOADER_MODE):
    # Mock only the isolated worker boundary; no shard bytes or model code are read.
    value, path = manifest
    spec = ExperimentSpec(payload)
    gp_template = value["arms"][0]
    value["spec_identity"] = spec.identity
    value["arms"] = []
    for arm in payload["arms"]:
        if arm["family"] == pf.K1_FAMILY:
            entry = k1_authorization(payload)["arms"][0]
        else:
            entry = copy.deepcopy(gp_template)
            entry.update(arm_id=arm["arm_id"], arm_identity=canonical_fingerprint(arm),
                         data=copy.deepcopy(arm["data"]))
        value["arms"].append(entry)
    path.write_bytes(pf._canonical(value))
    shard_root = tmp_path / "synthetic-shards"
    shard_root.mkdir()

    def unpack_stub(package_path, source_root):
        module = source_root / "src/molgap"
        module.mkdir(parents=True)
        for name in ("pcqm_gptrans_v4.py", "pcqm_k1_full_runner.py"):
            (module / name).write_text("# synthetic routing fixture\n", encoding="ascii")

    def launch_stub(source_root, data_root, entry, timeout_seconds, workspace, *, mode, arm,
                    phase="load"):
        if phase == "authorize":
            assert arm["family"] == pf.K1_FAMILY
            return {"status": "SELECTED_SHARDS_AUTHORIZED_ONLY"}
        status = statuses[arm["family"]["name"]]
        report = {"status": status}
        if status in {"LOADER_VERIFIED_ONLY", pf.K1_LOADER_STATUS,
                      "MODEL_SMOKE_VERIFIED_ONLY"}:
            report.update(shard_verified=True, loader_batch_built=True, device="cpu")
            if arm["family"] == pf.K1_FAMILY:
                report.update(verification_scope=pf.K1_SCOPE,
                              selected_real_shards=[
                                  {key: item[key] for key in ("path", "sha256", "rows", "role")}
                                  for item in entry["files"]],
                              batches={"train": {"graphs": 128, "rows": entry["files"][0]["rows"]}})
            else:
                report["batches"] = {"train": {"graphs": 128},
                                     "development": {"graphs": 128}}
            if status == "MODEL_SMOKE_VERIFIED_ONLY":
                report.update({key: True for key in pf.MODEL_CHECKS})
        else:
            report["error"] = {"type": "SyntheticFixture", "message": "Mocked nonpass"}
        return report

    monkeypatch.setattr(pf, "_unpack", unpack_stub)
    monkeypatch.setattr(pf, "_stage_files", lambda *args, **kwargs: None)
    monkeypatch.setattr(pf, "_launch", launch_stub)
    return run(package, payload, tmp_path, shard_manifest=path, shard_root=shard_root,
               expected_shard_manifest_sha256=pf._file_sha(path), mode=mode)


def test_missing_real_and_k1_are_not_pass(package, payload, tmp_path):
    result = run(package, payload, tmp_path)
    assert [a["status"] for a in result["arms"]] == [
        "MISSING_REAL_SHARD", "MISSING_REAL_SHARD"]
    assert result["status"] == "MISSING_REAL_SHARD"
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


@pytest.mark.parametrize("with_addon", [False, True])
def test_k1_manifest_declares_only_selected_train_shards(payload, tmp_path, with_addon):
    if with_addon:
        payload["arms"][1].update(addons=[addon("k1_pair_value")], addon_semantics="ordered")
    value = k1_authorization(payload)
    path = tmp_path / "k1-authorization.json"
    parsed = validate(payload, value, path)
    selected = parsed["arms"][0]["files"]
    assert selected[0]["role"] == "train" and selected[0]["rows"] == 50_000
    assert "status" not in parsed and "shard_verified" not in parsed


@pytest.mark.parametrize("change", [
    lambda entry: entry["files"][0].update(role="development"),
    lambda entry: entry["files"][0].update(role="official_validation"),
    lambda entry: entry["files"][0].update(rows=0),
    lambda entry: entry["files"][0].update(rows=127),
    lambda entry: entry["files"][0].update(rows=True),
    lambda entry: entry["files"][0].update(bytes=0),
    lambda entry: entry["files"][0].update(sha256="bad"),
    lambda entry: entry["files"][0].update(path="../train.pt"),
    lambda entry: entry["files"].append(dict(entry["files"][0])),
    lambda entry: entry["files"].append(dict(entry["files"][0], path="STORE/TOPOLOGY/TRAIN.PT")),
    lambda entry: entry["files"].clear(),
    lambda entry: entry["files"][0].pop("rows"),
])
def test_k1_manifest_rejects_bad_selected_declarations(payload, tmp_path, change):
    value = k1_authorization(payload)
    change(value["arms"][0])
    with pytest.raises(ValueError):
        validate(payload, value, tmp_path / "k1-authorization.json")


def test_k1_staging_rejects_hardlinked_shard(payload, tmp_path):
    entry = k1_authorization(payload)["arms"][0]
    root, target = tmp_path / "data", tmp_path / "stage"
    (root / "store/topology").mkdir(parents=True)
    target.mkdir()
    (root / "full.json").write_bytes(b"fixed")
    (root / "store/topology/train.pt").write_bytes(b"shard")
    entry["fixed_manifest"]["sha256"] = pf._sha(b"fixed")
    entry["files"][0].update(sha256=pf._sha(b"shard"), bytes=5)
    try:
        os.link(root / "store/topology/train.pt", root / "alias.pt")
    except OSError:
        pytest.skip("Host filesystem does not permit hardlink fixture")
    with pytest.raises(ValueError, match="Hardlinked"):
        pf._stage_files(entry, root, target, reject_hardlinks=True)


def test_k1_staging_rejects_symlinked_shard(payload, tmp_path):
    entry = k1_authorization(payload)["arms"][0]
    root, target = tmp_path / "data", tmp_path / "stage"
    (root / "store/topology").mkdir(parents=True)
    target.mkdir()
    (root / "full.json").write_bytes(b"fixed")
    original = root / "original.pt"
    original.write_bytes(b"shard")
    try:
        (root / "store/topology/train.pt").symlink_to(original)
    except OSError:
        pytest.skip("Host filesystem does not permit symlink fixture")
    entry["fixed_manifest"]["sha256"] = pf._sha(b"fixed")
    entry["files"][0].update(sha256=pf._sha(b"shard"), bytes=5)
    with pytest.raises(ValueError, match="Symlink"):
        pf._stage_files(entry, root, target, reject_hardlinks=True)


def test_k1_selection_matches_only_frozen_topology_records_with_synthetic_unit_contract(tmp_path, monkeypatch):
    # This unit fixture patches contract constants locally; it never publishes a preflight result.
    import molgap.pcqm_k1_full_runner as runner

    root = tmp_path / "data"
    (root / "store/topology").mkdir(parents=True)
    raw = b"synthetic-not-a-packed-graph"
    records = [{"role": "train", "file": f"store/topology/{index:02d}.pt",
                "rows": 128, "source_idx_min": index * 128,
                "source_idx_max": (index + 1) * 128 - 1,
                "sha256": pf._sha(raw), "bytes": len(raw)} for index in range(68)]
    selected = {"path": records[3]["file"], "sha256": records[3]["sha256"],
                "bytes": records[3]["bytes"], "rows": records[3]["rows"], "role": "train"}
    shard = root / selected["path"]
    shard.write_bytes(raw)
    rows = 68 * 128
    manifest = {"status": "complete", "identity": {
        "name": "ogb-train-full", "train_rows": rows, "development_rows": 0,
        "kaggle1": False, "scnet_compatible": False},
        "roles": {"train": {"source_idx_start": 0, "source_idx_stop": rows, "rows": rows},
                  "development": None},
        "graph_contract": {"feature_schema": "ogb", "node_feature_dim": 9,
                           "edge_feature_dim": 3, "rwse_dim": 16},
        "source": {"official_row_manifest_sha256": "a" * 64, "external_data_used": False},
        "official_validation_role_read": False, "test_dev_role_read": False,
        "test_challenge_role_read": False, "assets": {"topology": records}}
    fixed = root / "full.json"
    fixed.write_bytes(pf._canonical(manifest))
    with pytest.raises(RuntimeError, match="frozen K1 full PCQM manifest"):
        runner.validate_selected_preflight_shards(root, fixed, [selected], verify_content=True)
    with monkeypatch.context() as patch:
        patch.setattr(runner, "TRAIN_ROWS", rows)
        patch.setattr(runner, "OFFICIAL_ROW_MANIFEST_SHA256", "a" * 64)
        patch.setattr(runner, "FULL_TOPOLOGY_AGGREGATE_SHA256", runner._aggregate(records))
        patch.setattr(runner, "FULL_MANIFEST_CANONICAL_SHA256", canonical_fingerprint(manifest))
        matched, paths = runner.validate_selected_preflight_shards(
            root, fixed, [selected], verify_content=True)
        assert matched == [records[3]] and paths == [shard]
        not_staged = dict(selected, path=records[4]["file"])
        not_staged.update(sha256=records[4]["sha256"], bytes=records[4]["bytes"],
                          rows=records[4]["rows"])
        metadata, unstaged_paths = runner.validate_selected_preflight_shards(
            root, fixed, [not_staged], verify_content=False, metadata_only=True)
        assert metadata == [records[4]] and not unstaged_paths[0].exists()
        for key, bad in (("path", "store/topology/unknown.pt"), ("sha256", "0" * 64),
                         ("rows", 127), ("role", "development"), ("bytes", len(raw) + 1)):
            forged = dict(selected, **{key: bad})
            with pytest.raises(ValueError, match="differs from frozen train topology"):
                runner.validate_selected_preflight_shards(root, fixed, [forged], verify_content=True)
        with pytest.raises(ValueError, match="Duplicate selected"):
            runner.validate_selected_preflight_shards(root, fixed, [selected, selected], verify_content=True)
        shard.write_bytes(b"x" * len(raw))
        with pytest.raises(RuntimeError, match="hash changed"):
            runner.validate_selected_preflight_shards(root, fixed, [selected], verify_content=True)
        shard.write_bytes(b"modified")
        with pytest.raises(RuntimeError, match="byte count changed"):
            runner.validate_selected_preflight_shards(root, fixed, [selected], verify_content=True)


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


def test_k1_host_loader_import_is_rejected(tmp_path):
    host = tmp_path / "host/molgap"
    host.mkdir(parents=True)
    (host / "pcqm_k1_full_runner.py").write_text("raise AssertionError('host loader')\n", encoding="ascii")
    with pytest.raises(ImportError, match="Host/missing package import rejected"):
        pf._PackageOnly(tmp_path / "frozen").find_spec(
            "molgap.pcqm_k1_full_runner", [str(host)])


@pytest.mark.parametrize("mode", [pf.LOADER_MODE, pf.MODEL_MODE])
def test_unpacked_origin_in_fresh_interpreter_rejects_fake_real_manifest(tmp_path, manifest, monkeypatch, mode):
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
    result = pf._launch(root, data, manifest[0]["arms"][0], 30, stage, mode=mode)
    assert result["status"] == "LOADER_FAILED"
    assert result["shard_verified"] is False
    assert result["loader_batch_built"] is False
    assert result["import_origins"]["molgap.pcqm_gptrans_v4"] == "src/molgap/pcqm_gptrans_v4.py"
    assert "frozen real PCQM" in result["error"]["message"]
    assert not any(result[key] for key in pf.MODEL_CHECKS)


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


@pytest.mark.parametrize("with_addon", [False, True])
@pytest.mark.parametrize("authorized", [False, True])
def test_k1_baseline_and_pair_value_route_to_isolated_loader_only(
        repo, payload, tmp_path, monkeypatch, with_addon, authorized):
    payload["arms"] = [payload["arms"][1]]
    if with_addon:
        payload["arms"][0].update(addons=[addon("k1_pair_value")], addon_semantics="ordered")
    spec = ExperimentSpec(payload)
    source_package = tmp_path / "k1-package"
    build_experiment_source_package(spec, repo, ["src/example.py", "README.md"], source_package)
    value = k1_authorization(payload)
    authorization = tmp_path / "k1-authorization.json"
    authorization.write_bytes(pf._canonical(value))
    data = tmp_path / "data"
    data.mkdir()
    calls = []

    def unpack_stub(package, root):
        module = root / "src/molgap"
        module.mkdir(parents=True)
        (module / "pcqm_k1_full_runner.py").write_text("# routing-only fixture\n", encoding="ascii")

    def stage_stub(entry, root, destination, *, reject_hardlinks, items):
        calls.append(("stage", reject_hardlinks, tuple(item["path"] for item in items)))

    def launch_stub(source_root, data_root, entry, timeout_seconds, workspace, *, mode, arm,
                    phase="load"):
        calls.append(("launch", phase, mode, arm["family"],
                      tuple((item["name"], item["version"]) for item in arm["addons"])))
        if phase == "authorize":
            return ({"status": "SELECTED_SHARDS_AUTHORIZED_ONLY"} if authorized else
                    {"status": "LOADER_FAILED", "error": {
                        "type": "SyntheticFixture", "message": "Selected path is not authorized"}})
        return {"status": "UNSUPPORTED_FAMILY_PREFLIGHT",
                "error": {"type": "SyntheticFixture", "message": "No real data loaded"}}

    monkeypatch.setattr(pf, "_unpack", unpack_stub)
    monkeypatch.setattr(pf, "_stage_files", stage_stub)
    monkeypatch.setattr(pf, "_launch", launch_stub)
    result = pf.run_experiment_preflight(
        spec, source_package, tmp_path / "output",
        expected_package_identity=receipt(source_package)["package_identity"],
        shard_manifest=authorization, shard_root=data,
        expected_shard_manifest_sha256=pf._file_sha(authorization))
    expected = [
        ("stage", True, ("full.json",)),
        ("launch", "authorize", pf.LOADER_MODE, pf.K1_FAMILY,
         (("k1_pair_value", "1"),) if with_addon else ()),
    ]
    if authorized:
        expected.extend([
            ("stage", True, ("store/topology/train.pt",)),
            ("launch", "load", pf.LOADER_MODE, pf.K1_FAMILY,
             (("k1_pair_value", "1"),) if with_addon else ()),
        ])
    assert calls == expected
    assert result["status"] == ("UNSUPPORTED_FAMILY_PREFLIGHT" if authorized else "LOADER_FAILED")
    assert result["arms"][0]["verification_scope"] is None
    assert result["arms"][0]["selected_real_shards"] == []
    assert not result["arms"][0]["loader_batch_built"]


def test_k1_missing_fixed_manifest_is_nonpass_before_selected_shard(
        repo, payload, tmp_path, monkeypatch):
    payload["arms"] = [payload["arms"][1]]
    spec = ExperimentSpec(payload)
    source_package = tmp_path / "k1-package"
    build_experiment_source_package(spec, repo, ["src/example.py", "README.md"], source_package)
    value = k1_authorization(payload)
    authorization = tmp_path / "k1-authorization.json"
    authorization.write_bytes(pf._canonical(value))
    data = tmp_path / "data"
    data.mkdir()

    def unpack_stub(package, root):
        module = root / "src/molgap"
        module.mkdir(parents=True)
        (module / "pcqm_k1_full_runner.py").write_text("# no real loader\n", encoding="ascii")

    monkeypatch.setattr(pf, "_unpack", unpack_stub)
    result = pf.run_experiment_preflight(
        spec, source_package, tmp_path / "output",
        expected_package_identity=receipt(source_package)["package_identity"],
        shard_manifest=authorization, shard_root=data,
        expected_shard_manifest_sha256=pf._file_sha(authorization))
    assert result["status"] == "MISSING_REAL_SHARD"
    assert result["arms"][0]["selected_real_shards"] == []
    assert not result["arms"][0]["shard_verified"]


def test_missing_k1_loader_source_is_nonpass(tmp_path, payload):
    root = tmp_path / "source"
    module = root / "src/molgap"
    module.mkdir(parents=True)
    (module / "__init__.py").write_text("", encoding="ascii")
    stage = tmp_path / "worker"
    stage.mkdir()
    result = pf._launch(root, tmp_path / "data", k1_authorization(payload)["arms"][0],
                        30, stage, arm=payload["arms"][1])
    assert result["status"] == "LOADER_FAILED"
    assert result["loader_batch_built"] is False
    assert result["verification_scope"] is None
    assert result["selected_real_shards"] == []


@pytest.mark.parametrize("with_addon", [False, True])
def test_k1_worker_imports_only_frozen_loader_and_optional_addon_before_data(
        tmp_path, payload, with_addon):
    root = tmp_path / "source"
    module = root / "src/molgap"
    module.mkdir(parents=True)
    (module / "__init__.py").write_text("", encoding="ascii")
    (module / "pcqm_k1_full_runner.py").write_text(
        "def validate_selected_preflight_shards(*args, **kwargs): raise AssertionError('no data')\n"
        "def _load_selected_preflight_graphs(*args, **kwargs): raise AssertionError('no data')\n"
        "def _loader(*args, **kwargs): raise AssertionError('no data')\n", encoding="ascii")
    arm = copy.deepcopy(payload["arms"][1])
    if with_addon:
        addon_source = (
            "VALUE_DECOUPLED_MODE = 'neural_atom_k1_pair_token_value_decoupled'\n"
            "def make_encoder(*args): raise AssertionError('model construction forbidden')\n")
        (module / "k1_pair_token.py").write_bytes(addon_source.encode("ascii"))
        extension = addon("k1_pair_value")
        extension["source_sha256"] = pf._sha(addon_source.encode("ascii"))
        arm.update(addons=[extension], addon_semantics="ordered")
    stage = tmp_path / "worker"
    stage.mkdir()
    result = pf._launch(root, tmp_path / "absent-data",
                        k1_authorization(payload)["arms"][0], 30, stage,
                        arm=arm, phase="authorize")
    assert result["status"] == "LOADER_FAILED"
    assert result["import_origins"]["molgap.pcqm_k1_full_runner"] == (
        "src/molgap/pcqm_k1_full_runner.py")
    assert ("molgap.k1_pair_token" in result["import_origins"]) is with_addon
    assert not result["loader_batch_built"] and not any(result[key] for key in pf.MODEL_CHECKS)


def test_k1_worker_model_request_is_structured_unsupported(tmp_path, payload):
    root = tmp_path / "source"
    (root / "src/molgap").mkdir(parents=True)
    stage = tmp_path / "worker"
    stage.mkdir()
    result = pf._launch(root, tmp_path / "data", k1_authorization(payload)["arms"][0],
                        30, stage, mode=pf.MODEL_MODE, arm=payload["arms"][1])
    assert result["status"] == "UNSUPPORTED_MODEL_SMOKE"
    assert result["error"]["type"] == "UnsupportedMode"
    assert not result["shard_verified"] and not result["loader_batch_built"]
    assert not any(result[key] for key in pf.MODEL_CHECKS)


def test_smoke_dependency_failure_is_structured(tmp_path, manifest):
    root = tmp_path / "source"
    module = root / "src/molgap"
    module.mkdir(parents=True)
    (module / "__init__.py").write_text("", encoding="ascii")
    (module / "pcqm_gptrans_v4.py").write_text(
        'raise ModuleNotFoundError("required frozen dependency unavailable")\n', encoding="ascii")
    stage = tmp_path / "worker"
    stage.mkdir()
    result = pf._launch(root, tmp_path / "data", manifest[0]["arms"][0],
                        30, stage, mode=pf.MODEL_MODE)
    assert result["status"] == "LOADER_FAILED"
    assert result["error"]["type"] == "ModuleNotFoundError"
    assert result["device"] is None
    assert not any(result[key] for key in pf.MODEL_CHECKS)


def test_k1_selected_decoder_uses_training_loader_collate_without_full_role(tmp_path, monkeypatch):
    # Synthetic interface exercise only: no real manifest, report, or scientific evidence.
    import torch
    from torch_geometric.data import Data, InMemoryDataset
    import molgap.pcqm_k1_full_runner as runner

    graph = Data(x=torch.zeros((2, 9), dtype=torch.long),
                 edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
                 edge_attr=torch.zeros((2, 3), dtype=torch.long),
                 random_walk_pe=torch.zeros((2, 16)), y=torch.tensor([1.0]))
    packed = tmp_path / "synthetic-packed.pt"
    torch.save(InMemoryDataset.collate([graph for _ in range(runner.PHYSICAL_BATCH)]), packed)
    graphs, shards = runner._load_selected_preflight_graphs(
        [packed], [runner.PHYSICAL_BATCH])
    assert len(graphs) == len(shards[0]) == runner.PHYSICAL_BATCH
    monkeypatch.setattr(runner, "LOADER_WORKERS", 0)
    batch = next(iter(runner._loader(graphs, pass_index=0, start_batch=0)))
    assert batch.num_graphs == runner.PHYSICAL_BATCH
    assert batch.x.shape == (runner.PHYSICAL_BATCH * 2, 9)
    assert batch.random_walk_pe.shape == (runner.PHYSICAL_BATCH * 2, 16)
    with pytest.raises(RuntimeError, match="row count"):
        runner._load_selected_preflight_graphs([packed], [runner.PHYSICAL_BATCH + 1])


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
    assert json.loads((tmp_path / "output/preflight_summary.json").read_bytes()) == result
    assert not (tmp_path / "output/preflight.json").exists()
    for arm in result["arms"]:
        assert json.loads((tmp_path / "output/arms" / arm["arm_id"] / "preflight.json").read_bytes()) == arm
    assert not list((tmp_path / "output").rglob(".preflight-*"))


def test_cross_family_loader_summary_preserves_per_arm_status_and_evidence(
        package, payload, manifest, tmp_path, monkeypatch):
    result = mocked_arm_reports(
        package, payload, manifest, tmp_path, monkeypatch,
        {"gptrans_t": "LOADER_VERIFIED_ONLY", "neural_atom_k1": pf.K1_LOADER_STATUS})
    assert result["status"] == pf.ALL_ARMS_WITHIN_SCOPE_STATUS
    gptrans, k1 = result["arms"]
    assert [gptrans["status"], k1["status"]] == ["LOADER_VERIFIED_ONLY", pf.K1_LOADER_STATUS]
    assert gptrans["batches"] == {"train": {"graphs": 128}, "development": {"graphs": 128}}
    assert k1["verification_scope"] == pf.K1_SCOPE
    assert k1["selected_real_shards"] == [{
        "path": "store/topology/train.pt", "sha256": "2" * 64,
        "rows": 50_000, "role": "train"}]
    assert "full_role_content_unverified" in k1["missing_evidence"]
    for arm in result["arms"]:
        path = tmp_path / "output/arms" / arm["arm_id"] / "preflight.json"
        assert json.loads(path.read_bytes()) == arm


@pytest.mark.parametrize("failed_family", ["gptrans_t", "neural_atom_k1"])
@pytest.mark.parametrize("failure", ["MISSING_REAL_SHARD", "LOADER_FAILED",
                                     "UNSUPPORTED_FAMILY_PREFLIGHT", "PREFLIGHT_FAILED"])
def test_cross_family_loader_nonpass_never_aggregates(
        package, payload, manifest, tmp_path, monkeypatch, failed_family, failure):
    statuses = {"gptrans_t": "LOADER_VERIFIED_ONLY", "neural_atom_k1": pf.K1_LOADER_STATUS}
    statuses[failed_family] = failure
    result = mocked_arm_reports(package, payload, manifest, tmp_path, monkeypatch, statuses)
    assert result["status"] == "MIXED_NONPASS"
    assert [arm["status"] for arm in result["arms"]] == [
        statuses[arm["family"]["name"]] for arm in payload["arms"]]


def test_cross_family_success_statuses_cannot_be_exchanged(
        package, payload, manifest, tmp_path, monkeypatch):
    result = mocked_arm_reports(
        package, payload, manifest, tmp_path, monkeypatch,
        {"gptrans_t": pf.K1_LOADER_STATUS, "neural_atom_k1": "LOADER_VERIFIED_ONLY"})
    assert result["status"] == "MIXED_NONPASS"


@pytest.mark.parametrize("same_status", ["LOADER_VERIFIED_ONLY", pf.K1_LOADER_STATUS])
def test_cross_family_identical_success_label_is_not_valid_for_both_families(
        package, payload, manifest, tmp_path, monkeypatch, same_status):
    result = mocked_arm_reports(
        package, payload, manifest, tmp_path, monkeypatch,
        {"gptrans_t": same_status, "neural_atom_k1": same_status})
    assert result["status"] == "MIXED_NONPASS"
    assert [arm["status"] for arm in result["arms"]] == [same_status, same_status]


def test_summary_rejects_missing_arm_or_wrong_mode_success(payload):
    arms = payload["arms"]
    assert pf._summary_status(
        arms, [{"status": "LOADER_VERIFIED_ONLY"}], pf.LOADER_MODE,
    ) == "MIXED_NONPASS"
    assert pf._summary_status(
        arms, [{"status": "MODEL_SMOKE_VERIFIED_ONLY"} for _ in arms], pf.MODEL_MODE,
    ) == "MIXED_NONPASS"
    assert pf._summary_status(
        [arms[0]], [{"status": "LOADER_VERIFIED_ONLY"}], pf.MODEL_MODE,
    ) == "MIXED_NONPASS"
    assert pf._summary_status(
        arms, [{"status": pf.ALL_ARMS_WITHIN_SCOPE_STATUS} for _ in arms], pf.LOADER_MODE,
    ) == "MIXED_NONPASS"


def test_k1_model_smoke_unsupported_is_not_aggregated(
        package, payload, manifest, tmp_path, monkeypatch):
    result = mocked_arm_reports(
        package, payload, manifest, tmp_path, monkeypatch,
        {"gptrans_t": "MODEL_SMOKE_VERIFIED_ONLY"}, mode=pf.MODEL_MODE)
    assert result["status"] == "MIXED_NONPASS"
    assert [arm["status"] for arm in result["arms"]] == [
        "MODEL_SMOKE_VERIFIED_ONLY", "UNSUPPORTED_MODEL_SMOKE"]
    assert result["arms"][1]["error"]["type"] == "UnsupportedMode"


def test_same_status_gptrans_loader_summary_is_unchanged(
        repo, payload, manifest, tmp_path, monkeypatch):
    payload["arms"] = [payload["arms"][0], copy.deepcopy(payload["arms"][0])]
    payload["arms"][1]["arm_id"] = "gptrans-second"
    spec = ExperimentSpec(payload)
    source_package = tmp_path / "gptrans-package"
    build_experiment_source_package(spec, repo, ["src/example.py", "README.md"], source_package)
    result = mocked_arm_reports(source_package, payload, manifest, tmp_path, monkeypatch,
                                {"gptrans_t": "LOADER_VERIFIED_ONLY"})
    assert result["status"] == "LOADER_VERIFIED_ONLY"
    assert [arm["status"] for arm in result["arms"]] == ["LOADER_VERIFIED_ONLY"] * 2


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


@pytest.mark.parametrize("mode", [pf.LOADER_MODE, pf.MODEL_MODE])
def test_real_dual_arm_independent_success_failure(real_inputs, tmp_path, mode):
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
        expected_shard_manifest_sha256=pf._file_sha(manifest_path), timeout_seconds=600, mode=mode)
    first, second = result["arms"]
    assert first["status"] == ("LOADER_VERIFIED_ONLY" if mode == pf.LOADER_MODE else "MODEL_SMOKE_VERIFIED_ONLY")
    assert first["shard_verified"] and first["loader_batch_built"]
    assert second["status"] == "MISSING_REAL_SHARD"
    assert not second["shard_verified"] and not second["loader_batch_built"]
    assert result["status"] == "MIXED_NONPASS"
    assert all(second[key] is False for key in pf.MODEL_CHECKS)
    assert all(first[key] is (mode == pf.MODEL_MODE) for key in pf.MODEL_CHECKS)


def test_real_k1_selected_shards_are_scoped_if_explicitly_authorized(tmp_path):
    config = os.environ.get("MOLGAP_PREFLIGHT_K1_REAL_INPUTS")
    if not config:
        pytest.skip("No explicitly authorized K1 package and selected real train shards")
    inputs = json.loads(Path(config).read_bytes())
    spec = ExperimentSpec.from_json(Path(inputs["spec"]).read_text(encoding="utf-8"))
    arms = spec.to_dict()["arms"]
    assert arms and all(arm["family"] == pf.K1_FAMILY for arm in arms)
    manifest = pf.validate_real_shard_manifest(
        spec, inputs["shard_manifest"], inputs["expected_shard_manifest_sha256"])
    selected = {entry["arm_id"]: entry["files"] for entry in manifest["arms"]}
    assert set(selected) == {arm["arm_id"] for arm in arms}
    result = pf.run_experiment_preflight(
        spec, inputs["package_dir"], tmp_path / "output",
        expected_package_identity=inputs["expected_package_identity"],
        shard_manifest=inputs["shard_manifest"], shard_root=inputs["shard_root"],
        expected_shard_manifest_sha256=inputs["expected_shard_manifest_sha256"],
        timeout_seconds=600, mode=pf.LOADER_MODE)
    assert result["status"] == pf.K1_LOADER_STATUS
    for arm in result["arms"]:
        assert arm["status"] == pf.K1_LOADER_STATUS
        assert arm["verification_scope"] == pf.K1_SCOPE
        assert arm["selected_real_shards"] == [
            {key: item[key] for key in ("path", "sha256", "rows", "role")}
            for item in selected[arm["arm_id"]]]
        assert arm["batches"]["train"]["graphs"] == 128
        assert arm["batches"]["train"]["rows"] == sum(
            item["rows"] for item in selected[arm["arm_id"]])
        assert "full_role_content_unverified" in arm["missing_evidence"]
        assert "full_role_assembly_unverified" in arm["missing_evidence"]
        assert "model_numerics_unverified" in arm["missing_evidence"]
        assert not any(arm[key] for key in pf.MODEL_CHECKS)


def test_default_never_dispatches_model(package, payload, tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("default must never dispatch model smoke")
    monkeypatch.setattr(pf, "_model_smoke", forbidden)
    result = run(package, payload, tmp_path)
    assert result["mode"] == pf.LOADER_MODE
    for arm in result["arms"]:
        assert not arm["initial_state_checked"]
        assert arm["initial_state_sha256"] is None
        assert arm["normalized_gap_l1"] is None
        assert not any(arm[key] for key in pf.MODEL_CHECKS)


def test_explicit_smoke_missing_real_is_blocked(package, payload, tmp_path):
    result = run(package, payload, tmp_path, mode=pf.MODEL_MODE)
    assert result["mode"] == pf.MODEL_MODE
    assert result["status"] == "MIXED_NONPASS"
    assert [a["status"] for a in result["arms"]] == [
        "MISSING_REAL_SHARD", "UNSUPPORTED_MODEL_SMOKE"]
    assert result["arms"][1]["error"]["type"] == "UnsupportedMode"
    assert all(not any(a[k] for k in pf.MODEL_CHECKS) for a in result["arms"])


def test_unknown_mode_rejected_before_output(package, payload, tmp_path):
    with pytest.raises(ValueError, match="mode/version"):
        run(package, payload, tmp_path, mode="model-smoke-v99")
    assert not (tmp_path / "output").exists()


def test_smoke_dispatch_is_fixed(payload):
    arm = copy.deepcopy(payload["arms"][0])
    arm["initialization"]["kind"] = "random"
    arm["addons"] = []
    assert pf._smoke_variant(arm) == "reference"
    arm["addons"] = [{"name": "centered_logits", "version": "1"}]
    assert pf._smoke_variant(arm) == "centered_logits"
    arm["addons"][0]["name"] = "arbitrary.module"
    with pytest.raises(ValueError, match="dispatch"):
        pf._smoke_variant(arm)
    arm["initialization"]["kind"] = "frozen_state"
    with pytest.raises(ValueError, match="random initialization"):
        pf._smoke_variant(arm)


@pytest.mark.parametrize("failure", ["model", "optimizer", "scheduler", "rng", "write", "restore"])
def test_checkpoint_mismatch_or_failure_raises(tmp_path, failure):
    # Isolated serialization rejection test; never a real/model success claim.
    from types import SimpleNamespace

    class State:
        def __init__(self, value):
            self.value = value
        def state_dict(self):
            return copy.deepcopy(self.value)
        def load_state_dict(self, value, **kwargs):
            self.value = copy.deepcopy(value)
            if failure == "restore":
                self.value["broken"] = True

    model, optimizer, scheduler = State({"weight": 1}), State({"step": 1}), State({"epoch": 0})
    saved = {}
    def save(path, value):
        if failure == "write":
            raise OSError("diagnostic write failed")
        saved.update(copy.deepcopy(value))
    def load(*args, **kwargs):
        if failure in {"model", "optimizer", "scheduler", "rng"}:
            saved[failure]["corrupted"] = True
        return saved
    runtime = SimpleNamespace(
        atomic_torch_save=save, torch_load_compat=load,
        capture_rng_state=lambda **kwargs: {"python": (1, 2)},
        restore_rng_state=lambda *args, **kwargs: None)
    with pytest.raises((ValueError, OSError), match="checkpoint|write"):
        pf._diagnostic_roundtrip(runtime, model, optimizer, scheduler,
                                tmp_path / "diagnostic.pt", None)


def test_real_scheduler_step_api():
    from types import SimpleNamespace
    from molgap.pcqm_gptrans_v4 import FrozenEpochScheduler
    optimizer = SimpleNamespace(param_groups=[{"lr": -1}])
    scheduler = FrozenEpochScheduler(optimizer)
    rate = scheduler.step(0)
    assert scheduler.state_dict() == {"epoch": 0}
    assert optimizer.param_groups[0]["lr"] == rate == scheduler.learning_rate(0)
    assert not hasattr(scheduler, "set_epoch")


def test_exact_state_rejects_tensor_rng_changes():
    import numpy as np
    import torch
    value = {"torch": torch.tensor([1, 2], dtype=torch.uint8),
             "numpy": ("MT19937", np.array([1, 2], dtype=np.uint32))}
    altered = copy.deepcopy(value)
    altered["torch"][0] = 3
    assert not pf._exact_state(value, altered)
    altered = copy.deepcopy(value)
    altered["numpy"][1][0] = 3
    assert not pf._exact_state(value, altered)


def test_initial_hash_mismatch_blocks_before_forward(payload, tmp_path):
    from types import SimpleNamespace
    arm = copy.deepcopy(payload["arms"][0])
    arm["initialization"].update(kind="random", state_sha256="1" * 64)
    arm["addons"] = []
    model = SimpleNamespace(to=lambda device: model)
    runtime = SimpleNamespace(
        configure_fp32_determinism=lambda seed: None,
        _make_model=lambda **kwargs: model,
        model_state_sha256=lambda model: "0" * 64)
    result = {key: False for key in (*pf.MODEL_CHECKS, "initial_state_checked")}
    with pytest.raises(ValueError, match="initial state hash mismatch"):
        pf._model_smoke(runtime, arm, None, None, tmp_path / "diagnostic.pt", None, result)
    assert result["initial_state_sha256"] == "0" * 64
    assert not result["initial_state_checked"]
    assert not any(result[key] for key in pf.MODEL_CHECKS)
