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


def frozen_launch(entry, limit):
    entry.INPUT_ROOT.mkdir()
    archive = entry.INPUT_ROOT / "source_payload.bin"
    with tarfile.open(archive, "w:gz") as bundle:
        for name, data in {
            "recipe.json": json.dumps({"allocation_wall_limit_seconds": limit}).encode(),
            "src/molgap/experiment_preflight.py": b"def _unpack(package, source):\n    pass\n",
        }.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            bundle.addfile(info, io.BytesIO(data))
    config = {"jobs": [{"recipe": "recipe.json"}],
              "expected_source_archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
              "expected_package_identity": "package", "spec_identity": "spec"}
    launch = entry.INPUT_ROOT / "experiment_launch.json"
    launch.write_text(json.dumps(config))
    entry.EXPECTED_LAUNCH_SHA256 = hashlib.sha256(launch.read_bytes()).hexdigest()
    for name in ("experiment_spec.json", "package_manifest.json", "SOURCE_COMMIT.txt",
                 "SOURCE_ARCHIVE_SHA256.txt", "SOURCE_FILES.json"):
        (entry.INPUT_ROOT / name).write_text("fixture")


@pytest.mark.parametrize("limit", [120, 12600, 14400, None])
@pytest.mark.parametrize("phase_failure", [None, "pip", "runtime"])
def test_bootstrap_reuses_shared_writer_and_preserves_budget(entry, monkeypatch, limit, phase_failure):
    frozen_launch(entry, limit)
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
    assert calls[0] == ("pip", None if limit is None else limit - 60)
    if phase_failure != "pip":
        assert calls[1] == ("runtime", None if limit is None else limit - 65)
    assert bool(shared_writes) == (phase_failure != "pip")
    assert report(entry)["monotonic_seconds"] == (5 if phase_failure == "pip" else 25)


@pytest.mark.parametrize("limit", [119, 14401, True, 12600.0])
def test_bootstrap_rejects_invalid_ceiling_without_pip(entry, monkeypatch, limit):
    frozen_launch(entry, limit)
    monkeypatch.setattr(entry.subprocess, "run", lambda *a, **k: pytest.fail("pip must not execute"))
    with pytest.raises(ValueError, match="outside supported bounds"):
        entry.main()
    assert report(entry)["status"] == "failed"


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
