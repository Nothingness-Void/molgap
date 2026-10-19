"""Tiny frozen archives exercise the server entry and the real freshness guard."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tarfile
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_entry(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("server_bootstrap_fixture", ROOT / "platforms/kaggle/run_experiment.py")
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    paths = {name: tmp_path / name for name in ("input", "working", "temp")}
    for path in paths.values():
        path.mkdir()
    real_path = Path
    aliases = {"/kaggle/input": paths["input"], "/kaggle/working": paths["working"],
        "/kaggle/working/experiment": paths["working"] / "experiment",
        "/kaggle/working/bootstrap_allocation.json": paths["working"] / "bootstrap_allocation.json",
        "/kaggle/temp/molgap-workflow": paths["temp"] / "workflow"}
    monkeypatch.setattr(entry, "Path", lambda value: aliases.get(str(value), real_path(value)))
    monkeypatch.setattr(entry.subprocess, "check_output", lambda *a, **k: "Tesla T4\nTesla T4\n")
    return entry, paths


@pytest.mark.parametrize("pinned", [False, True], ids=["generic", "pinned"])
@pytest.mark.parametrize("existing_output", [False, True], ids=["fresh", "existing-checkpoint"])
def test_real_pair_freshness_guard_is_reached_without_entry_receipts(tmp_path, monkeypatch, pinned, existing_output):
    from molgap import kaggle_pair_runtime as pair
    entry, paths = load_entry(tmp_path, monkeypatch)
    mount = paths["input"] / "source"
    mount.mkdir()
    extractor = b"import tarfile\ndef _unpack(package, source):\n    with tarfile.open(package / 'source.tar.gz') as archive:\n        archive.extractall(source)\n"
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w:gz") as archive:
        info = tarfile.TarInfo("src/molgap/experiment_preflight.py")
        info.size = len(extractor)
        archive.addfile(info, io.BytesIO(extractor))
    (mount / "source_payload.bin").write_bytes(payload.getvalue())
    config = {"expected_source_archive_sha256": hashlib.sha256(payload.getvalue()).hexdigest(),
        "expected_package_identity": "package", "spec_identity": "spec"}
    launch = mount / "experiment_launch.json"
    launch.write_text(json.dumps(config))
    entry.EXPECTED_LAUNCH_SHA256 = hashlib.sha256(launch.read_bytes()).hexdigest()
    declaration = {"experiment_id": "k1-infrastructure-t4-100k" if pinned else "generic-fixture",
        "arms": [{"family": {"name": "fixture", "version": "1"}, "addons": []}]}
    (mount / "experiment_spec.json").write_text(json.dumps(declaration))
    for name in ("package_manifest.json", "SOURCE_COMMIT.txt", "SOURCE_ARCHIVE_SHA256.txt", "SOURCE_FILES.json"):
        (mount / name).write_text("{}")
    class ReachedParser(RuntimeError):
        pass
    monkeypatch.setattr(pair, "_json", lambda path: (_ for _ in ()).throw(ReachedParser("freshness passed")))
    output = paths["working"] / "experiment"
    if existing_output:
        output.mkdir()
        (output / "last_checkpoint.pt").write_bytes(b"existing checkpoint")
    real_pair = pair.run_two_phase_pair
    def invoke_pair(**kwargs):
        if not existing_output:
            assert not output.exists() or not any(output.iterdir())
        return real_pair(**kwargs)
    # Keep a direct handle to the actual guard while substituting only bootstrap imports.
    pair_module = ModuleType("molgap.kaggle_pair_runtime")
    pair_module.run_two_phase_pair = invoke_pair
    package_module = ModuleType("molgap.experiment_package")
    package_module.verify_experiment_source_package = lambda _: {"package_identity": "package", "spec_identity": "spec"}
    spec_module = ModuleType("molgap.experiment_spec")
    spec_module.ExperimentSpec = SimpleNamespace(from_json=lambda _: SimpleNamespace())
    launch_module = ModuleType("molgap.experiment_launch_config")
    launch_module.validate_launch_config = lambda *a, **k: None
    for name, module in (("molgap.kaggle_pair_runtime", pair_module), ("molgap.experiment_package", package_module),
            ("molgap.experiment_spec", spec_module), ("molgap.experiment_launch_config", launch_module)):
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(sys, "path", list(sys.path))
    import runpy
    monkeypatch.setattr(runpy, "run_path", lambda _: {"prepare_python": lambda *a, **k: (Path("qualified-python"), {})})
    def subprocess_run(command, **kwargs):
        if "molgap.kaggle_pair_runtime" in command:
            assert (paths["working"] / "environment_qualification.json").is_file()
            return invoke_pair(source_root=paths["temp"] / "workflow/source", package_dir=mount,
                input_root=paths["input"], launch_path=launch, output=output)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(entry.subprocess, "run", subprocess_run)
    error = ValueError if existing_output else ReachedParser
    with pytest.raises(error, match="fresh output root" if existing_output else "freshness passed"):
        entry.main()
    if existing_output:
        assert (output / "last_checkpoint.pt").read_bytes() == b"existing checkpoint"
        assert sorted(p.name for p in output.iterdir()) == ["last_checkpoint.pt"]
    else:
        assert not output.exists()
    assert json.loads((paths["working"] / "bootstrap_allocation.json").read_bytes())["status"] == "failed"


@pytest.mark.parametrize("workload_fails", [False, True], ids=["success", "startup-failure"])
@pytest.mark.parametrize("within_handler", [False, True], ids=["normal-call", "caller-handling-error"])
def test_receipt_failure_keeps_original_startup_error_and_cleans_temporary(tmp_path, monkeypatch, workload_fails, within_handler):
    entry, paths = load_entry(tmp_path, monkeypatch)
    original = RuntimeError("original bootstrap failure")
    def workload(started):
        if workload_fails:
            raise original
    monkeypatch.setattr(entry, "_workload", workload)
    def fail_replace(*args):
        raise OSError("synthetic receipt publication failure")
    monkeypatch.setattr(entry.os, "replace", fail_replace)
    with pytest.raises(RuntimeError if workload_fails else OSError) as caught:
        if within_handler:
            try:
                raise LookupError("unrelated caller exception")
            except LookupError:
                entry.main()
        else:
            entry.main()
    if workload_fails:
        assert caught.value is original
    assert not (paths["working"] / "bootstrap_allocation.tmp").exists()
