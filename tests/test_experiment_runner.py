"""Synthetic orchestration checks; no data, real model construction or forward."""
import copy
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from molgap import experiment_runner as runner
from molgap.experiment_spec import ExperimentSpec, FAMILIES, SCHEMA_VERSION, TERMINAL_PROTOCOL


def _ref(name):
    return {"name": name, "version": "1", "sha256": "a" * 64}


@pytest.fixture
def declaration():
    arms = []
    for index, family in enumerate(("gptrans_t", "neural_atom_k1")):
        contract = FAMILIES[(family, "1")]
        arms.append({
            "arm_id": family, "scientific_role": "reference" if index == 0 else "candidate",
            "family": {"name": family, "version": "1"}, "base": _ref("synthetic-base"),
            "initialization": {"kind": "random", "seed": 42, "state_sha256": "b" * 64},
            "data": {
                "dataset": _ref("pcqm4mv2"), "split": _ref("synthetic-no-data"),
                "roles": [{"role": role, "membership_sha256": "c" * 64,
                           "row_order_sha256": "d" * 64, "usage_sha256": "e" * 64}
                          for role in contract.roles],
                "feature_schema": contract.feature_schema, "feature_sha256": "f" * 64,
                "target": "pcqm4mv2-gap-eV-direct",
            },
            "training": {"recipe": _ref(contract.recipe), "overrides": {},
                         "objective": _ref("normalized-gap-l1"),
                         "sampler": _ref(contract.sampler), "transform": _ref(contract.transform)},
            "addons": [], "addon_semantics": "baseline",
        })
    return {
        "schema_version": SCHEMA_VERSION, "experiment_id": "synthetic-question",
        "logical_run_id": "synthetic-run", "arms": arms,
        "platform": {"name": "local", "accelerator": "synthetic", "device_count": 2,
                     "cpu_cores": 2, "memory_gib": 4, "atomic_checkpoints": True,
                     "retrievable_chunks": True},
        "prospective": {"trajectory_id": "synthetic-trajectory", "hypothesis": "No real data",
                        "cheapest_falsifier": "Metadata only", "stop_rule": "No training",
                        "budget_sha256": "a" * 64},
        "evidence": {"policy": _ref("molgap-v5"), "required_artifacts": [
            "v5_evidence", "costs", "roles", "trace_manifest", "terminal_artifact"]},
        "terminal_protocol": TERMINAL_PROTOCOL,
    }


@pytest.fixture
def spec(declaration):
    return ExperimentSpec(declaration)


def _read(path):
    raw = path.read_bytes()
    value = json.loads(raw)
    assert raw == json.dumps(value, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=True, allow_nan=False).encode("utf-8")
    return value


def _no_authority(value):
    if isinstance(value, dict):
        assert not {"replay_ready", "training_success", "rml_closed", "ready_for_desktop",
                    "READY_FOR_DESKTOP", "terminal_descriptor", "observed_steps",
                    "expected_steps"}.intersection(value)
        for nested in value.values():
            _no_authority(nested)
    elif isinstance(value, list):
        for nested in value:
            _no_authority(nested)


def test_spawn_dual_probe_order_devices_and_metadata(spec, tmp_path, monkeypatch):
    # Parent-side torch presence must not reject the run. No torch import is needed.
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "parent-unchanged")
    root = tmp_path / "probe"
    summary = runner.run_experiment(spec, root, ["7", 2], timeout_seconds=60)
    assert summary == _read(root / "run_summary.json")
    assert summary["status"] == "SUCCEEDED"
    assert summary["start_method"] == "spawn"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "parent-unchanged"
    arms = spec.to_dict()["arms"]
    assert [item["arm_id"] for item in summary["arms"]] == [arm["arm_id"] for arm in arms]
    assert len({item["pid"] for item in summary["arms"]}) == 2
    for item, arm, requested, visible in zip(summary["arms"], arms, ["7", 2], ["7", "2"]):
        assert item["pid"] != os.getpid()
        assert item["result_path"] == f"arms/{arm['arm_id']}/arm_result.json"
        result = _read(root / item["result_path"])
        assert result["requested_device"] == requested
        assert result["visible_device"] == visible
        assert result["spec_identity"] == spec.identity
        assert result["arm_identity"] == runner.canonical_fingerprint(arm)
        assert result["parameter_count"] is None
        assert result["recorded_by"] == "child"
        metadata = result["metadata"]
        assert metadata["family"]["name"] == arm["family"]["name"]
        assert metadata["family"]["version"] == "1"
        assert metadata["spec_identity"] == spec.identity
        assert metadata["arm_id"] == arm["arm_id"]
        assert metadata["addons"] == []
        assert metadata["addon_semantics"] == "baseline"
        assert metadata["factory"] and metadata["source_module"]
        assert metadata["checkpoint_resume_owner"]
        assert metadata["checkpoint_resume_implemented"] is False
        assert metadata["runner_limitations"]
        _no_authority(result)
    _no_authority(summary)
    before = {path: path.read_bytes() for path in root.rglob("*.json")}
    with pytest.raises(ValueError, match="already exists"):
        runner.run_experiment(spec, root, [7, 2])
    assert before == {path: path.read_bytes() for path in root.rglob("*.json")}
    assert not list(root.rglob(".arm-write-*"))


def _invoke_child(spec, directory, arm_index=0, worker="adapter_probe"):
    directory.mkdir()
    arm_id = spec.to_dict()["arms"][arm_index]["arm_id"]
    runner._child(spec.to_json(), arm_id, worker, None, "", str(directory))


@pytest.mark.parametrize("index,module,build_name", [
    (0, "gptrans_adapter", "build_gptrans_model"),
    (1, "k1_adapter", "build_k1_model"),
])
def test_construct_static_dispatch_with_lightweight_factories(
        spec, tmp_path, monkeypatch, index, module, build_name):
    from molgap import gptrans_adapter, k1_adapter

    adapters = {"gptrans_adapter": gptrans_adapter, "k1_adapter": k1_adapter}
    calls = []
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "before-child")
    monkeypatch.setitem(sys.modules, "numpy", SimpleNamespace(
        random=SimpleNamespace(seed=lambda seed: calls.append(("numpy", seed)))))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        manual_seed=lambda seed: calls.append(("torch", seed))))
    monkeypatch.setattr(runner.random, "seed", lambda seed: calls.append(("python", seed)))
    original_dispatch = runner._adapter

    def dispatch(arm):
        assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
        return original_dispatch(arm)

    def build(received, arm_id):
        assert received == spec
        assert arm_id == spec.to_dict()["arms"][index]["arm_id"]
        assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
        calls.append(("factory", arm_id))
        # No forward, checkpoint, device transfer or trainer interface exists.
        return SimpleNamespace(parameters=lambda: [SimpleNamespace(numel=lambda: 17)])

    monkeypatch.setattr(runner, "_adapter", dispatch)
    monkeypatch.setattr(adapters[module], build_name, build)
    directory = tmp_path / "construct"
    _invoke_child(spec, directory, index, "construct")
    result = _read(directory / "arm_result.json")
    assert result["status"] == "SUCCEEDED"
    assert result["parameter_count"] == 17
    assert calls[:3] == [("python", 42), ("numpy", 42), ("torch", 42)]
    assert calls[3][0] == "factory"
    _no_authority(result)


def test_probe_allows_preimported_torch_and_never_calls_build(spec, tmp_path, monkeypatch):
    from molgap import gptrans_adapter

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "before-child")
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    monkeypatch.setattr(gptrans_adapter, "build_gptrans_model",
                        lambda *a, **kw: pytest.fail("probe attempted construction"))
    _invoke_child(spec, tmp_path / "probe")
    assert _read(tmp_path / "probe" / "arm_result.json")["status"] == "SUCCEEDED"


@pytest.mark.parametrize("index", [0, 1])
def test_frozen_construct_fails_before_seeding_or_build(declaration, tmp_path, monkeypatch, index):
    from molgap import gptrans_adapter, k1_adapter

    declaration["arms"][index]["initialization"]["kind"] = "frozen_state"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "before-child")
    for module, name in ((gptrans_adapter, "build_gptrans_model"), (k1_adapter, "build_k1_model")):
        monkeypatch.setattr(module, name, lambda *a, **kw: pytest.fail("frozen build called"))
    monkeypatch.setattr(runner.random, "seed", lambda seed: pytest.fail("frozen seed called"))
    with pytest.raises(SystemExit) as caught:
        _invoke_child(ExperimentSpec(declaration), tmp_path / "frozen", index, "construct")
    assert caught.value.code == 1
    result = _read(tmp_path / "frozen" / "arm_result.json")
    assert result["status"] == "FAILED"
    assert "frozen_state" in result["error"]["message"]
    assert result["parameter_count"] is None
    assert result["metadata"]["checkpoint_resume_implemented"] is False


def _fake_context(monkeypatch, behaviors):
    """Exercise the parent loop without adding a production worker/callback API."""
    processes = []

    class Process:
        def __init__(self, *, target, args):
            self.target, self.args = target, args
            self.behavior = behaviors[len(processes)]
            self.pid = None
            self.exitcode = None
            self.alive = False
            self.killed = False
            processes.append(self)

        def start(self):
            if self.behavior == "start_error":
                raise OSError("synthetic spawn error")
            self.pid = os.getpid()
            if self.behavior in {"timeout", "kill"}:
                self.alive = True
                return
            if self.behavior == "crash":
                self.exitcode = 9
                return
            if self.behavior == "missing":
                self.exitcode = 0
                return
            with monkeypatch.context() as env:
                env.setenv("CUDA_VISIBLE_DEVICES", "fake-parent")
                try:
                    self.target(*self.args)
                    self.exitcode = 0
                except SystemExit as exc:
                    self.exitcode = exc.code
            if self.behavior == "invalid":
                runner._atomic(Path(self.args[-1]) / "arm_result.json", {"status": "SUCCEEDED"})

        def is_alive(self):
            return self.alive

        def terminate(self):
            if self.behavior != "kill":
                self.alive, self.exitcode = False, -15

        def kill(self):
            self.killed = True
            self.alive, self.exitcode = False, -9

        def join(self, timeout=None):
            pass

    def context(method):
        assert method == "spawn"
        return SimpleNamespace(Process=Process)

    monkeypatch.setattr(runner.multiprocessing, "get_context", context)
    ticks = iter(range(10000))
    monkeypatch.setattr(runner.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(runner.time, "sleep", lambda seconds: None)
    return processes


@pytest.mark.parametrize("behaviors,status,arm_statuses", [
    (["ok", "crash"], "PARTIAL_FAILURE", ["SUCCEEDED", "FAILED"]),
    (["crash", "crash"], "FAILED", ["FAILED", "FAILED"]),
    (["ok", "timeout"], "TIMED_OUT", ["SUCCEEDED", "TIMED_OUT"]),
    (["kill", "ok"], "TIMED_OUT", ["TIMED_OUT", "SUCCEEDED"]),
    (["start_error", "ok"], "PARTIAL_FAILURE", ["FAILED", "SUCCEEDED"]),
    (["missing", "ok"], "PARTIAL_FAILURE", ["FAILED", "SUCCEEDED"]),
    (["invalid", "ok"], "PARTIAL_FAILURE", ["FAILED", "SUCCEEDED"]),
])
def test_independent_parent_failure_and_timeout(spec, tmp_path, monkeypatch, behaviors, status, arm_statuses):
    processes = _fake_context(monkeypatch, behaviors)
    root = tmp_path / "run"
    summary = runner.run_experiment(spec, root, [0, 1], timeout_seconds=0.1)
    assert summary["status"] == status
    assert [arm["status"] for arm in summary["arms"]] == arm_statuses
    assert _read(root / "run_summary.json") == summary
    for arm, expected in zip(summary["arms"], arm_statuses):
        result = _read(root / arm["result_path"])
        assert result["status"] == expected
        if expected == "SUCCEEDED":
            assert result["recorded_by"] == "child"
            assert result["error"] is None
        else:
            assert result["recorded_by"] == "parent"
            assert result["error"]
            assert result["observed_duration_seconds"] is None
        _no_authority(result)
    assert all(not process.is_alive() for process in processes)
    if "kill" in behaviors:
        assert processes[behaviors.index("kill")].killed


@pytest.mark.parametrize("failed_indices", [(0,), (0, 1)])
def test_worker_errors_preserve_other_arm(declaration, tmp_path, monkeypatch, failed_indices):
    from molgap import gptrans_adapter, k1_adapter

    for index in failed_indices:
        declaration["arms"][index]["initialization"]["kind"] = "frozen_state"
    monkeypatch.setitem(sys.modules, "numpy", SimpleNamespace(random=SimpleNamespace(seed=lambda seed: None)))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(manual_seed=lambda seed: None))
    monkeypatch.setattr(runner.random, "seed", lambda seed: None)
    for module, name in ((gptrans_adapter, "build_gptrans_model"), (k1_adapter, "build_k1_model")):
        monkeypatch.setattr(module, name, lambda *args: SimpleNamespace(parameters=lambda: []))
    _fake_context(monkeypatch, ["ok", "ok"])
    root = tmp_path / "run"
    summary = runner.run_experiment(ExperimentSpec(declaration), root, [0, 1],
                                    worker="construct", timeout_seconds=100)
    assert summary["status"] == ("FAILED" if len(failed_indices) == 2 else "PARTIAL_FAILURE")
    for index, arm in enumerate(summary["arms"]):
        assert arm["status"] == ("FAILED" if index in failed_indices else "SUCCEEDED")
        if index in failed_indices:
            assert "frozen_state" in arm["error"]["message"]


@pytest.fixture
def forbid_spawn(monkeypatch):
    monkeypatch.setattr(runner.multiprocessing, "get_context",
                        lambda *a: pytest.fail("invalid input reached spawn"))


@pytest.mark.parametrize("devices", [
    [], [0], [0, 1, 2], (0, 1), [0, "0"], [None, "CPU"], ["cpu", "CPU"],
    [True, 1], [-1, 1], [0.0, 1], ["01", 1], ["0,1", 2], ["../0", 1], [object(), 1],
])
def test_invalid_devices_fail_before_spawn(spec, tmp_path, forbid_spawn, devices):
    with pytest.raises(ValueError):
        runner.run_experiment(spec, tmp_path / "run", devices)
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("kwargs", [
    {"worker": "train"}, {"worker": lambda: None}, {"timeout_seconds": 0},
    {"timeout_seconds": -1}, {"timeout_seconds": True},
    {"timeout_seconds": float("nan")}, {"timeout_seconds": float("inf")},
])
def test_invalid_worker_timeout(spec, tmp_path, forbid_spawn, kwargs):
    with pytest.raises(ValueError):
        runner.run_experiment(spec, tmp_path / "run", [0, 1], **kwargs)
    assert not (tmp_path / "run").exists()


def test_platform_count_mismatch(declaration, tmp_path, forbid_spawn):
    declaration["platform"]["device_count"] = 1
    with pytest.raises(ValueError, match="platform.device_count"):
        runner.run_experiment(ExperimentSpec(declaration), tmp_path / "run", [0, 1])


@pytest.mark.parametrize("kind", ["dict", "provider", "subclass", "extra", "noncanonical", "unknown_family"])
def test_forged_spec_rejected(spec, declaration, tmp_path, forbid_spawn, kind):
    class SubSpec(ExperimentSpec):
        pass

    if kind == "dict":
        bad = declaration
    elif kind == "provider":
        bad = SimpleNamespace(to_json=lambda: pytest.fail("provider called"))
    elif kind == "subclass":
        bad = SubSpec(declaration)
    else:
        bad = copy.copy(spec)
        if kind == "extra":
            object.__setattr__(bad, "to_json", lambda: pytest.fail("injected method called"))
        elif kind == "noncanonical":
            object.__setattr__(bad, "_canonical_json", json.dumps(declaration, indent=2))
        else:
            declaration["arms"][0]["family"]["name"] = "custom.provider"
            object.__setattr__(bad, "_canonical_json", runner._json(declaration))
    with pytest.raises((TypeError, ValueError)):
        runner.run_experiment(bad, tmp_path / "run", [0, 1])
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("name", ["CON", "NUL.txt", "COM1", "trailing."])
def test_unsafe_arm_names(declaration, tmp_path, forbid_spawn, name):
    declaration["arms"][0]["arm_id"] = name
    with pytest.raises(ValueError, match="Unsafe path"):
        runner.run_experiment(ExperimentSpec(declaration), tmp_path / "run", [0, 1])


def test_case_colliding_arms(declaration, tmp_path, forbid_spawn):
    declaration["arms"][0]["arm_id"] = "Arm"
    declaration["arms"][1]["arm_id"] = "arm"
    with pytest.raises(ValueError, match="Case-colliding"):
        runner.run_experiment(ExperimentSpec(declaration), tmp_path / "run", [0, 1])


@pytest.mark.parametrize("kind", ["relative", "root", "traversal", "reserved", "existing", "string", "missing_parent"])
def test_unsafe_output(spec, tmp_path, forbid_spawn, kind):
    output = {
        "relative": Path("relative"), "root": Path(tmp_path.anchor),
        "traversal": tmp_path / ".." / "escape", "reserved": tmp_path / "NUL",
        "existing": tmp_path, "string": str(tmp_path / "run"),
        "missing_parent": tmp_path / "absent" / "run",
    }[kind]
    with pytest.raises((TypeError, ValueError)):
        runner.run_experiment(spec, output, [0, 1])


def test_symlink_output_rejected(spec, tmp_path, forbid_spawn):
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    try:
        link.symlink_to(target, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("Directory symlinks are unavailable on this host")
    with pytest.raises(ValueError, match="Symlink/junction/reparse"):
        runner.run_experiment(spec, link / "run", [0, 1])
    assert not (target / "run").exists()


@pytest.mark.parametrize("corruption", [
    "missing", "syntax", "noncanonical", "duplicate", "nan", "identity", "pid", "metadata",
    "error", "count", "duration", "timestamp", "authority", "status", "exitcode", "running",
])
def test_child_result_fail_closed(spec, tmp_path, monkeypatch, corruption):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "parent")
    directory = tmp_path / "child"
    _invoke_child(spec, directory)
    path = directory / "arm_result.json"
    valid = _read(path)
    record = {"path": path, "root": tmp_path, "arm": spec.to_dict()["arms"][0],
              "requested": None, "visible": "",
              "process": SimpleNamespace(pid=os.getpid(), exitcode=0)}
    result = copy.deepcopy(valid)
    raw = None
    if corruption == "missing":
        path.unlink()
    elif corruption == "syntax":
        raw = b"{"
    elif corruption == "noncanonical":
        raw = json.dumps(result, indent=2).encode()
    elif corruption == "duplicate":
        raw = ('{"status":"FAILED",' + runner._json(result)[1:]).encode()
    elif corruption == "nan":
        raw = runner._json(result).replace('"parameter_count":null', '"parameter_count":NaN').encode()
    elif corruption == "identity":
        result["arm_identity"] = "f" * 64
    elif corruption == "pid":
        result["pid"] = os.getpid() + 100
    elif corruption == "metadata":
        result["metadata"] = None
    elif corruption == "error":
        result["error"] = {"type": "Failure", "message": "not success"}
    elif corruption == "count":
        result["parameter_count"] = 999
    elif corruption == "duration":
        result["observed_duration_seconds"] = True
    elif corruption == "timestamp":
        result["observed_end"] = "not-observed"
    elif corruption == "authority":
        result["metadata"]["replay_ready"] = True
    elif corruption == "status":
        result["status"] = []
    elif corruption == "exitcode":
        record["process"].exitcode = 9
    elif corruption == "running":
        result.update(status="RUNNING", metadata=None, observed_end=None, observed_duration_seconds=None)
    if raw is not None:
        path.write_bytes(raw)
    elif corruption != "missing":
        runner._atomic(path, result)
    summary = runner._finish(record, spec, "adapter_probe")
    assert summary["status"] == "FAILED"
    repaired = _read(path)
    assert repaired["recorded_by"] == "parent"
    assert repaired["status"] == "FAILED"
    assert repaired["error"]
    _no_authority(repaired)


def test_parent_rejects_fabricated_frozen_success(declaration, tmp_path, monkeypatch):
    declaration["arms"][0]["initialization"]["kind"] = "frozen_state"
    spec = ExperimentSpec(declaration)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "parent")
    directory = tmp_path / "child"
    _invoke_child(spec, directory)
    path = directory / "arm_result.json"
    result = _read(path)
    result.update(worker="construct", parameter_count=17)
    runner._atomic(path, result)
    record = {"path": path, "root": tmp_path, "arm": spec.to_dict()["arms"][0],
              "requested": None, "visible": "",
              "process": SimpleNamespace(pid=os.getpid(), exitcode=0)}
    assert runner._finish(record, spec, "construct")["status"] == "FAILED"
    assert "frozen_state" in _read(path)["error"]["message"]


def test_atomic_replace_failure_keeps_old_json_and_cleans_temp(tmp_path, monkeypatch):
    path = tmp_path / "result.json"
    runner._atomic(path, {"old": 1})
    before = path.read_bytes()

    def fail_replace(source, target):
        assert source.parent == path.parent
        assert target == path
        assert path.read_bytes() == before
        assert _read(source) == {"new": 2}
        raise OSError("synthetic replace failure")

    monkeypatch.setattr(runner.os, "replace", fail_replace)
    with pytest.raises(OSError):
        runner._atomic(path, {"new": 2})
    assert path.read_bytes() == before
    assert not list(tmp_path.glob(".arm-write-*"))
    with pytest.raises(ValueError):
        runner._atomic(path, {"invalid": float("nan")})
    assert path.read_bytes() == before
