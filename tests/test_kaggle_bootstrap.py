"""CPU-only bootstrap observations; no pip, NVIDIA command or workers execute."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import types

import pytest


@pytest.fixture
def entry(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / "platforms/kaggle/run_experiment.py"
    spec = importlib.util.spec_from_file_location("kaggle_bootstrap_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "INPUT_ROOT", tmp_path / "input")
    monkeypatch.setattr(module, "WORK_ROOT", tmp_path / "work")
    monkeypatch.setattr(module, "OUTPUT_ROOT", tmp_path / "output")
    # Kaggle provides file_digest; the local verification venv is Python 3.10.
    if not hasattr(module.hashlib, "file_digest"):
        monkeypatch.setattr(module.hashlib, "file_digest",
                            lambda stream, algorithm: hashlib.new(algorithm, stream.read()),
                            raising=False)
    monkeypatch.setattr(module.subprocess, "check_output", lambda *a, **k: "Tesla T4\nTesla T4\n")
    return module


def report(entry):
    return json.loads((entry.OUTPUT_ROOT / "allocation_entry_observation.json").read_text())


@pytest.mark.parametrize("failure", [None, RuntimeError("preflight failed"),
                                    subprocess.TimeoutExpired("pip", 30), KeyboardInterrupt()])
def test_entry_finally_measures_complete_window(entry, monkeypatch, failure):
    ticks = iter([100.0, 145.0])
    monkeypatch.setattr(entry.time, "perf_counter", lambda: next(ticks))

    def bootstrap(started, state):
        assert started == 100.0
        if failure is not None:
            raise failure

    monkeypatch.setattr(entry, "_bootstrap", bootstrap)
    if failure is None:
        entry.main()
    else:
        with pytest.raises(type(failure)) as caught:
            entry.main()
        assert caught.value is failure
    value = report(entry)
    assert value["status"] == ("complete" if failure is None else "failed")
    assert value["error"] == (None if failure is None else {
        "type": type(failure).__name__, "message": str(failure)})
    assert value["monotonic_seconds"] == 45.0
    assert value["observed_gpu_count"] == 2
    assert value["allocated_device_seconds"] == 90.0
    assert value["gpu_count_source"] == "nvidia-smi"
    assert value["wall_start_utc"].endswith("+00:00")
    assert value["wall_end_utc"] >= value["wall_start_utc"]
    assert "idle assigned GPU" in value["scope"]
    assert "not observed" in value["platform_release_scope"]
    assert value["entire_platform_release_seconds"] is None


@pytest.mark.parametrize("metadata", [FileNotFoundError(), subprocess.CalledProcessError(1, "nvidia-smi"),
                                       subprocess.TimeoutExpired("nvidia-smi", 10), ""])
def test_early_failure_keeps_unknown_gpu_cost_null(entry, monkeypatch, metadata):
    def query(*args, **kwargs):
        assert args[0] == ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]
        assert kwargs == {"text": True, "timeout": 10}
        if isinstance(metadata, Exception):
            raise metadata
        return metadata

    monkeypatch.setattr(entry.subprocess, "check_output", query)
    with pytest.raises(RuntimeError, match="Expected one frozen"):
        entry.main()
    value = report(entry)
    assert value["observed_gpu_count"] is None
    assert value["gpu_count_source"] is None
    assert value["allocated_device_seconds"] is None
    assert value["status"] == "failed"


def frozen_launch(entry, limit, *, launch_overrides=None, recipe_fields=None):
    entry.INPUT_ROOT.mkdir()
    archive = entry.INPUT_ROOT / "source_payload.bin"
    recipe_record = {"allocation_wall_limit_seconds": limit}
    recipe_record.update(recipe_fields or {})
    with tarfile.open(archive, "w:gz") as bundle:
        for name, data in {
            "recipe.json": json.dumps(recipe_record).encode(),
            "src/molgap/experiment_preflight.py": (
                b"def _unpack(package, source):\n"
                b"    import tarfile\n"
                b"    with tarfile.open(package / 'source.tar.gz', 'r:gz') as bundle:\n"
                b"        stream = bundle.extractfile('recipe.json')\n"
                b"        if stream is None:\n"
                b"            raise RuntimeError('fixture recipe absent')\n"
                b"        (source / 'recipe.json').write_bytes(stream.read())\n"),
        }.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            bundle.addfile(info, io.BytesIO(data))
    config = {"jobs": [{"arm_id": "fixture_arm", "recipe": "recipe.json"}],
              "expected_source_archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
              "expected_package_identity": "package", "spec_identity": "spec"}
    config.update(launch_overrides or {})
    launch = entry.INPUT_ROOT / "experiment_launch.json"
    launch.write_text(json.dumps(config))
    entry.EXPECTED_LAUNCH_SHA256 = hashlib.sha256(launch.read_bytes()).hexdigest()
    for name in ("experiment_spec.json", "package_manifest.json", "SOURCE_COMMIT.txt",
                 "SOURCE_ARCHIVE_SHA256.txt", "SOURCE_FILES.json"):
        (entry.INPUT_ROOT / name).write_text("fixture")


@pytest.mark.parametrize("limit", [90, 120, 12600, 14400, 14401, 32400, 100000, None])
@pytest.mark.parametrize("reserve", [0, 60])
@pytest.mark.parametrize("phase_failure", [None, "pip", "runtime"])
def test_bootstrap_reuses_shared_writer_and_preserves_budget(entry, monkeypatch, limit, reserve, phase_failure):
    frozen_launch(entry, limit, launch_overrides={"cleanup_reserve_seconds": reserve})
    calls = []
    clock = [100.0]
    monkeypatch.setattr(entry.time, "perf_counter", lambda: clock[0])

    def pip(command, **kwargs):
        calls.append(("pip", kwargs["timeout"]))
        assert command[1:] == ["-m", "pip", "install", "--quiet", "numpy<2",
                               "torch-geometric==2.6.1", "ogb==1.3.6"]
        clock[0] += 5
        if phase_failure == "pip":
            raise RuntimeError("pip failed")

    def run(**kwargs):
        calls.append(("runtime", kwargs["maximum_wall_seconds"]))
        clock[0] += 20
        if phase_failure == "runtime":
            raise RuntimeError("runtime failed")

    shared_writes = []

    def shared_writer(path, value):
        shared_writes.append(path)
        entry._atomic_entry_json(path, value)

    # All shared imports are synthetic; extraction and launch checks use real bytes.
    modules = {
        "molgap": {},
        "molgap.training_reproducibility": {"atomic_json": shared_writer},
        "molgap.experiment_package": {"verify_experiment_source_package": lambda _: {
            "package_identity": "package", "spec_identity": "spec"}},
        "molgap.kaggle_pair_runtime": {"run_two_phase_pair": run},
    }
    for name, attrs in modules.items():
        module = types.ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(entry.subprocess, "run", pip)
    if phase_failure:
        with pytest.raises(RuntimeError, match=phase_failure + " failed"):
            entry.main()
    else:
        entry.main()
    assert calls[0] == ("pip", None if limit is None else limit - reserve)
    if phase_failure != "pip":
        assert calls[1] == ("runtime", None if limit is None else limit - reserve - 5)
    assert bool(shared_writes) == (phase_failure != "pip")
    assert report(entry)["monotonic_seconds"] == (5 if phase_failure == "pip" else 25)


@pytest.mark.parametrize("limit", [0, -1, 60, True, 12600.0, "32400"])
def test_bootstrap_rejects_invalid_ceiling_without_pip(entry, monkeypatch, limit):
    frozen_launch(entry, limit)
    monkeypatch.setattr(entry.subprocess, "run", lambda *a, **k: pytest.fail("pip must not execute"))
    with pytest.raises(ValueError, match="ceiling"):
        entry.main()
    assert report(entry)["status"] == "failed"


@pytest.mark.parametrize("changes", [
    {"allocation_wall_limit_seconds": 0}, {"allocation_wall_limit_seconds": True},
    {"allocation_wall_limit_seconds": 32400.0}, {"allocation_wall_limit_seconds": None},
    {"allocation_wall_limit_seconds": 32400}, {"cleanup_reserve_seconds": -1},
    {"cleanup_reserve_seconds": True}, {"cleanup_reserve_seconds": 14400},
])
def test_launch_rejects_invalid_or_conflicting_budget_before_pip(entry, monkeypatch, changes):
    frozen_launch(entry, 14400, launch_overrides=changes)
    monkeypatch.setattr(entry.subprocess, "run", lambda *a, **k: pytest.fail("pip must not execute"))
    with pytest.raises(ValueError):
        entry.main()


def test_launch_config_can_supply_budget_without_recipe_limit(entry, monkeypatch):
    frozen_launch(entry, None, launch_overrides={
        "allocation_wall_limit_seconds": 32400, "cleanup_reserve_seconds": 180,
    })
    calls = []
    monkeypatch.setattr(entry.time, "perf_counter", lambda: 100.0)
    def stop_after_observing_budget(command, **kwargs):
        calls.append(kwargs["timeout"])
        raise RuntimeError("synthetic stop before installation")
    monkeypatch.setattr(entry.subprocess, "run", stop_after_observing_budget)
    with pytest.raises(RuntimeError, match="synthetic stop"):
        entry.main()
    assert calls == [32220]


@pytest.mark.parametrize("runtime_failure", [False, True])
def test_shared_recipe_resolution_records_nested_metadata_without_shard_decode(
        entry, monkeypatch, capsys, runtime_failure):
    manifest_record = {"assets": {"topology": [
        {"role": "train", "file": "train.pt"}]}}
    manifest_bytes = json.dumps(manifest_record).encode()
    dataset_sha = hashlib.sha256(manifest_bytes).hexdigest()
    frozen_launch(entry, 600, recipe_fields={
        "format": "molgap-family-training-recipe-v1",
        "dataset": {"manifest": "slug/manifest.json", "manifest_sha256": dataset_sha,
                    "development": False},
    })
    manifest = entry.INPUT_ROOT / "datasets" / "release-owner" / "slug" / "manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_bytes(manifest_bytes)
    (manifest.parent / "train.pt").write_bytes(b"synthetic shard bytes")
    monkeypatch.setattr(entry.time, "perf_counter", lambda: 100.0)
    monkeypatch.setattr(entry.subprocess, "run", lambda *a, **k: None)

    def atomic_json(path, value):
        entry._atomic_entry_json(path, value)

    def read_recipe(path):
        record = json.loads(path.read_text(encoding="utf-8"))
        recipe = types.SimpleNamespace(dataset=types.SimpleNamespace(**record["dataset"]))
        return record, recipe

    def resolve_training_manifest(root, selection):
        assert root == entry.INPUT_ROOT
        assert selection.manifest_sha256 == dataset_sha
        return manifest

    def artifact_path(root, relative):
        path = root / relative
        assert path.is_file()
        return path

    def run(**kwargs):
        # Do not replace the runtime with a permissive no-op: its real entry
        # contract rejects receipt files created before it owns the directory.
        assert not kwargs["output"].exists() or not any(kwargs["output"].iterdir())
        if runtime_failure:
            raise RuntimeError("synthetic native preflight failure")

    modules = {
        "molgap": {},
        "molgap.training_reproducibility": {"atomic_json": atomic_json},
        "molgap.experiment_package": {"verify_experiment_source_package": lambda _: {
            "package_identity": "package", "spec_identity": "spec"}},
        "molgap.kaggle_pair_runtime": {"run_two_phase_pair": run},
        "molgap.family_training_runner": {"read_recipe": read_recipe},
        "molgap.family_training_data": {"resolve_training_manifest": resolve_training_manifest},
        "molgap.experiment_family_workflow": {"_artifact_path": artifact_path},
    }
    for name, attrs in modules.items():
        module = types.ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(sys, "path", list(sys.path))

    if runtime_failure:
        with pytest.raises(RuntimeError, match="synthetic native preflight failure"):
            entry.main()
    else:
        entry.main()

    value = json.loads((entry.OUTPUT_ROOT / "input_resolution.json").read_text())
    assert value == {"format": "molgap-kaggle-input-resolution-v1", "spec_identity": "spec",
                     "arms": {"fixture_arm": {
                         "manifest": "datasets/release-owner/slug/manifest.json",
                         "manifest_sha256": dataset_sha,
                         "selected_shard_paths_present": 1,
                         "shards_decoded": False,
                     }}}
    assert "INPUT_RESOLVED fixture_arm datasets/release-owner/slug/manifest.json" in capsys.readouterr().out


@pytest.mark.parametrize("existing_output", [False, True])
def test_shared_resolution_reaches_real_pair_freshness_guard(entry, monkeypatch, existing_output):
    from molgap import kaggle_pair_runtime as pair_runtime
    # Run the actual freshness guard, then stop before any package/data/hardware
    # probe. This catches the bootstrap/runtime interaction missed by no-op mocks.
    class ReachedPairParser(RuntimeError):
        pass

    def stop_at_config(path):
        raise ReachedPairParser("freshness guard passed")

    real_pair = pair_runtime.run_two_phase_pair
    monkeypatch.setattr(pair_runtime, "_json", stop_at_config)
    manifest_bytes = b'{"assets":{"topology":[{"role":"train","file":"train.pt"}]}}'
    frozen_launch(entry, None, recipe_fields={"format": "molgap-family-training-recipe-v1",
        "dataset": {"manifest": "data/manifest.json", "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                    "development": False}})
    manifest = entry.INPUT_ROOT / "data/manifest.json"
    manifest.parent.mkdir()
    manifest.write_bytes(manifest_bytes)
    (manifest.parent / "train.pt").write_bytes(b"synthetic")
    monkeypatch.setattr(entry.subprocess, "run", lambda *a, **k: None)
    modules = {
        "molgap": {},
        "molgap.training_reproducibility": {"atomic_json": entry._atomic_entry_json},
        "molgap.experiment_package": {"verify_experiment_source_package": lambda _: {
            "package_identity": "package", "spec_identity": "spec"}},
        "molgap.kaggle_pair_runtime": {"run_two_phase_pair": real_pair},
        "molgap.family_training_runner": {"read_recipe": lambda path: ({}, types.SimpleNamespace(
            dataset=types.SimpleNamespace(manifest="data/manifest.json", development=False,
                manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest())))},
        "molgap.family_training_data": {"resolve_training_manifest": lambda *args: manifest},
        "molgap.experiment_family_workflow": {"_artifact_path": lambda root, name: root / name},
    }
    for name, attrs in modules.items():
        module = types.ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(sys, "path", list(sys.path))
    if existing_output:
        entry.OUTPUT_ROOT.mkdir()
        sentinel = entry.OUTPUT_ROOT / "last_checkpoint.pt"
        sentinel.write_bytes(b"existing checkpoint; never delete")
        with pytest.raises(ValueError, match="fresh output root"):
            entry.main()
        assert sentinel.read_bytes() == b"existing checkpoint; never delete"
    else:
        with pytest.raises(ReachedPairParser, match="freshness guard passed"):
            entry.main()
    assert json.loads((entry.OUTPUT_ROOT / "input_resolution.json").read_bytes())["arms"]


def test_atomic_fallback_cleans_up_failed_replace(entry, monkeypatch):
    entry.OUTPUT_ROOT.mkdir()
    target = entry.OUTPUT_ROOT / "observation.json"
    target.write_text("original")

    def fail_replace(*args):
        raise OSError("replace failed")

    monkeypatch.setattr(entry.os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        entry._atomic_entry_json(target, {"status": "failed"})
    assert target.read_text() == "original"
    assert list(entry.OUTPUT_ROOT.iterdir()) == [target]


def test_publication_failure_preserves_bootstrap_error(entry, monkeypatch, capsys):
    def failed_write(*args):
        raise OSError("output unavailable")

    monkeypatch.setattr(entry, "_atomic_entry_json", failed_write)
    with pytest.raises(RuntimeError, match="Expected one frozen"):
        entry.main()
    assert "publication failed" in capsys.readouterr().err
