"""Boundary tests, not model tests. Synthetic inputs never produce real-shard success."""
import copy
import io
import json
import os
from pathlib import Path
import tarfile
from types import SimpleNamespace

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


def test_k1_manifest_rejects_extra_shards_before_staging(payload, manifest):
    value, path = manifest
    arm = payload["arms"][1]
    entry = copy.deepcopy(value["arms"][0])
    entry.update(arm_id=arm["arm_id"], arm_identity=canonical_fingerprint(arm),
                 data=copy.deepcopy(arm["data"]))
    entry["files"] = [
        {"path": "k1-a.pt", "sha256": "2" * 64, "bytes": 5, "role": "train"},
        {"path": "k1-b.pt", "sha256": "3" * 64, "bytes": 5, "role": "train"},
    ]
    value["arms"] = [entry]
    with pytest.raises(ValueError, match="exactly one"):
        validate(payload, value, path)


@pytest.mark.parametrize("mutation", ["none", "wrong_scope", "wrong_shard", "wrong_role"])
def test_partial_worker_receipt_is_explicitly_bounded(tmp_path, monkeypatch, mutation):
    root = tmp_path / "source"
    module = root / "src/molgap"
    module.mkdir(parents=True)
    (module / "pcqm_topology.py").write_text("", encoding="ascii")
    entry = {"files": [{"path": "train.pt", "sha256": "a" * 64,
                        "bytes": 5, "role": "train"}]}
    arm = {"family": {"name": "neural_atom_k1", "version": "1"}}
    result = {
        "status": "PARTIAL_LOADER_VERIFIED_ONLY", "shard_verified": True,
        "loader_batch_built": True, "import_origins": {
            "molgap.pcqm_topology": "src/molgap/pcqm_topology.py"},
        "batches": {"train": {"graphs": 128, "rows": 128, "device": "cpu",
                              "shard": "train.pt", "sha256": "a" * 64,
                              "source_idx_min": 0, "source_idx_max": 127}},
        "device": "cpu", "scope": "selected-topology-shards", "error": None,
        "forward_checked": False, "backward_checked": False,
        "optimizer_step_checked": False, "checkpoint_roundtrip_checked": False,
        "initial_state_checked": False, "initial_state_sha256": None,
        "normalized_gap_l1": None,
    }
    if mutation == "wrong_scope":
        result["scope"] = "complete-frozen-shards"
    elif mutation == "wrong_shard":
        result["batches"]["train"]["sha256"] = "b" * 64
    elif mutation == "wrong_role":
        result["batches"]["development"] = result["batches"].pop("train")

    def worker(command, **kwargs):
        Path(command[-1]).write_bytes(pf._canonical(result))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(pf.subprocess, "run", worker)
    stage = tmp_path / "stage"
    stage.mkdir()
    if mutation == "none":
        assert pf._launch(root, tmp_path, entry, 30, stage, arm=arm)["scope"] == "selected-topology-shards"
    else:
        with pytest.raises(ValueError, match="partial|topology"):
            pf._launch(root, tmp_path, entry, 30, stage, arm=arm)


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


def test_k1_isolated_worker_dispatch_fails_closed_on_nonreal_manifest(tmp_path):
    root = tmp_path / "source"
    module = root / "src/molgap"
    module.mkdir(parents=True)
    (module / "__init__.py").write_text("", encoding="ascii")
    (module / "pcqm_topology.py").write_text(
        'def inspect_selected_topology_shards(*args): raise ValueError("not real topology")\n',
        encoding="ascii")
    data = tmp_path / "data"
    data.mkdir()
    raw = b"not a frozen manifest"
    (data / "fixed.json").write_bytes(raw)
    entry = {"fixed_manifest": {"path": "fixed.json", "sha256": pf._sha(raw)},
             "files": [{"path": "train.pt", "sha256": "a" * 64,
                        "bytes": 5, "role": "train"}]}
    arm = {"family": {"name": "neural_atom_k1", "version": "1"}}
    stage = tmp_path / "worker"
    stage.mkdir()
    result = pf._launch(root, data, entry, 30, stage, arm=arm)
    assert result["status"] == "LOADER_FAILED"
    assert result["error"]["message"] == "not real topology"
    assert result["import_origins"]["molgap.pcqm_topology"] == "src/molgap/pcqm_topology.py"
    assert not result["shard_verified"] and not result["loader_batch_built"]


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
        "MISSING_REAL_SHARD", "UNSUPPORTED_FAMILY_PREFLIGHT"]
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
