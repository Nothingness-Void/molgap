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


@pytest.mark.parametrize("limit,reserve,phase_failure", [
    (None, 60, None),
    (90, 0, None),
    (120, 60, None),
    (100000, 60, None),
    (None, 60, "pip"),
    (120, 60, "pip"),
    (None, 60, "runtime"),
    (120, 60, "runtime"),
])
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


@pytest.mark.parametrize("scenario", ["success", "preflight_failure", "existing_output", "old_output_order"])
def test_frozen_package_launcher_reaches_actual_two_arm_preflight_barrier(tmp_path, scenario):
    """Only pip/NVIDIA/worker processes are replaced inside a fresh interpreter.

    The archived launcher, extractor, package verifier, recipe/mount resolver,
    pair scheduler, phase barrier and durable execution receipts all execute.
    """
    import shutil
    from molgap.experiment_execution import build_family_recipe
    from molgap.experiment_package import build_experiment_source_package, stage_source_inputs, initial_state_paths
    from molgap.experiment_source_inventory import SHARED_SOURCE_FILES
    from molgap.experiment_spec import ExperimentSpec
    from molgap.family_training_data import DatasetSelection
    from molgap.family_training_recipe import FamilyRecipe, ScheduleSpec
    from test_experiment_package import git
    from test_experiment_workflow import _arm, _spec_payload, _jobs

    root = Path(__file__).resolve().parents[1]
    repo = tmp_path / "frozen-repo"
    repo.mkdir()
    for name in SHARED_SOURCE_FILES:
        target = repo / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / name, target)
    if scenario == "old_output_order":
        launcher = repo / "platforms/kaggle/run_experiment.py"
        # Reproduce the historical fault using a separately frozen source
        # fixture, rather than weakening/mocking the current runtime guard.
        old = launcher.read_text(encoding="utf-8").replace(
            '        state["input_resolution"] = resolution',
            '        atomic_json(OUTPUT_ROOT / "input_resolution.json", resolution)')
        launcher.write_text(old, encoding="utf-8")
    mounted = tmp_path / "input"
    graph = mounted / "datasets/release-owner/data/manifest.json"
    graph.parent.mkdir(parents=True)
    graph.write_text(json.dumps({"assets": {"topology": [{"role": "train", "file": "train.pt"}]}}))
    (graph.parent / "train.pt").write_bytes(b"synthetic CPU boundary; never decoded")
    dataset_sha = hashlib.sha256(graph.read_bytes()).hexdigest()
    recipe = FamilyRecipe("neural_atom_k1", "reference", 42, "2" * 64,
        DatasetSelection("data/manifest.json", dataset_sha, False), 256, 0, 4,
        .001, .01, ScheduleSpec("cosine_step", 4), "final-live",
        "python-epoch-shuffle", checkpoint_every=2, target_mean_eV=3., target_std_eV=2.)
    record = build_family_recipe(("neural_atom_k1", "3"), configuration=recipe,
        train_role_identity="3" * 64, development_role_identity=None)
    recipes = repo / "recipes"
    recipes.mkdir()
    recipe_bytes = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    arms = []
    for name in ("first", "second"):
        (recipes / (name + ".json")).write_bytes(recipe_bytes)
        arms.append(_arm("neural_atom_k1", "3", name,
            recipe_sha256=hashlib.sha256(recipe_bytes).hexdigest()))
    spec = ExperimentSpec(_spec_payload(arms, platform="kaggle"))
    git(repo, "init")
    git(repo, "config", "user.email", "fixture@example.invalid")
    git(repo, "config", "user.name", "Fixture")
    git(repo, "config", "core.autocrlf", "false")
    git(repo, "add", ".")
    git(repo, "-c", "commit.gpgsign=false", "commit", "-m", "Frozen CPU integration fixture")
    package = tmp_path / "package"
    manifest = build_experiment_source_package(spec, repo,
        [*SHARED_SOURCE_FILES, "recipes/first.json", "recipes/second.json"], package)
    source_mount = mounted / "source"
    initials = {arm: tmp_path / (arm + ".pt") for arm in ("first", "second")}
    for path in initials.values():
        path.write_bytes(b"same frozen initialization; worker process boundary is replaced")
    # The failure journey also retains the historical per-arm layout.
    paths = ({arm: f"initial_states/{arm}.pt" for arm in initials}
             if scenario == "preflight_failure" else initial_state_paths(initials))
    stage_source_inputs(package, source_mount, initials, initial_paths=paths)
    jobs = [{**job, "initial_state": paths[job["arm_id"]]} for job in _jobs(spec)]
    assert len(list((source_mount / "initial_states").iterdir())) == (2 if scenario == "preflight_failure" else 1)
    launch = {"jobs": jobs, "expected_source_archive_sha256": manifest["archive_sha256"],
        "expected_package_identity": manifest["package_identity"], "spec_identity": spec.identity,
        "account": "fixture", "run_reference": "fixture/cpu-integration"}
    launch_path = source_mount / "experiment_launch.json"
    launch_path.write_text(json.dumps(launch))
    launcher = tmp_path / "run.py"
    launcher.write_text((repo / "platforms/kaggle/run_experiment.py").read_text(encoding="utf-8").replace(
        "EXPECTED_LAUNCH_SHA256 = None", "EXPECTED_LAUNCH_SHA256 = " +
        repr(hashlib.sha256(launch_path.read_bytes()).hexdigest())), encoding="utf-8")
    output = tmp_path / "output"
    if scenario == "existing_output":
        output.mkdir()
        (output / "last_checkpoint.pt").write_bytes(b"old checkpoint must survive")
    driver = tmp_path / "cpu_driver.py"
    driver.write_text(r'''
import hashlib, importlib.util, json, pathlib, subprocess, sys
launcher, mounted, work, output, scenario, boundary = sys.argv[1:]
module_spec = importlib.util.spec_from_file_location("frozen_launcher", launcher)
entry = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(entry)
entry.INPUT_ROOT, entry.WORK_ROOT, entry.OUTPUT_ROOT = map(pathlib.Path, (mounted, work, output))
if not hasattr(hashlib, "file_digest"):
    hashlib.file_digest = lambda stream, algorithm: hashlib.new(algorithm, stream.read())
commands = []
initial_paths = []
package_checks = []
def observe_package_check(frame, event, arg):
    if (event == "call" and frame.f_code.co_name == "verify_experiment_source_package"
            and frame.f_globals.get("__name__") == "molgap.experiment_package"):
        package_checks.append(str(frame.f_locals["package_dir"]))
subprocess.check_output = lambda *a, **kw: "Tesla T4\nTesla T4\n"
def pip(command, **kwargs):
    assert command[1:4] == ["-m", "pip", "install"]
    return subprocess.CompletedProcess(command, 0)
subprocess.run = pip
class Worker:
    def __init__(self, command, **kwargs):
        self.phase = command[command.index("--phase") + 1]
        self.arm = command[command.index("--arm") + 1]
        self.returncode = None
        commands.append({"phase": self.phase, "arm": self.arm})
        launch = json.loads(pathlib.Path(command[command.index("--launch") + 1]).read_bytes())
        job = next(job for job in launch["jobs"] if job["arm_id"] == self.arm)
        initial = pathlib.Path(command[command.index("--launch") + 1]).parent / job["initial_state"]
        assert initial.read_bytes() == b"same frozen initialization; worker process boundary is replaced"
        initial_paths.append(job["initial_state"])
        # The real scheduler must pass the all-arm barrier before formal train.
        if self.phase == "train":
            assert [row["arm"] for row in commands if row["phase"] == "preflight"] == ["first", "second"]
    def poll(self):
        if self.returncode is not None:
            return self.returncode
        if scenario == "preflight_failure" and self.phase == "preflight":
            return 1 if self.arm == "first" else None
        return 0
    def terminate(self):
        self.returncode = -15
    def kill(self):
        self.returncode = -9
    def wait(self, **kwargs):
        return self.returncode if self.returncode is not None else self.poll()
subprocess.Popen = Worker
error = None
sys.setprofile(observe_package_check)
try:
    entry.main()
except BaseException as exc:
    error = {"type": type(exc).__name__, "message": str(exc)}
finally:
    sys.setprofile(None)
pathlib.Path(boundary).write_text(json.dumps({"commands": commands, "error": error,
    "initial_paths": initial_paths,
    "package_checks": package_checks,
    "runtime_module": getattr(sys.modules.get("molgap.kaggle_pair_runtime"), "__file__", None)}))
''', encoding="utf-8")
    boundary = tmp_path / "worker_boundary.json"
    child = subprocess.run([sys.executable, "-I", str(driver), str(launcher), str(mounted),
        str(tmp_path / "work"), str(output), scenario, str(boundary)], capture_output=True, text=True, timeout=60)
    assert child.returncode == 0, child.stdout + child.stderr
    observed = json.loads(boundary.read_bytes())
    assert observed["package_checks"] == [str(tmp_path / "work/package")]
    assert Path(observed["runtime_module"]).is_relative_to(tmp_path / "work/source")
    if scenario == "success":
        assert set(observed["initial_paths"]) == {"initial_states/first.pt"}
        assert observed["error"] is None, observed
        assert observed["commands"] == [{"phase": phase, "arm": arm}
            for phase in ("preflight", "train") for arm in ("first", "second")]
        assert json.loads((output / "execution_report.json").read_bytes())["status"] == "complete"
    elif scenario == "preflight_failure":
        assert set(observed["initial_paths"]) == {"initial_states/first.pt", "initial_states/second.pt"}
        assert "Arm failed during preflight" in observed["error"]["message"]
        assert observed["commands"] == [{"phase": "preflight", "arm": arm} for arm in ("first", "second")]
        execution = json.loads((output / "execution_report.json").read_bytes())
        assert all(row["training_started"] is False for row in execution["arms"].values())
    else:
        assert "fresh output root" in observed["error"]["message"]
        assert observed["commands"] == []
    if scenario == "existing_output":
        assert (output / "last_checkpoint.pt").read_bytes() == b"old checkpoint must survive"
    if scenario != "old_output_order":
        assert set(json.loads((output / "input_resolution.json").read_bytes())["arms"]) == {"first", "second"}
