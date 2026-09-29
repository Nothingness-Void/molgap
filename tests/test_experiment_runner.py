"""Synthetic runner tests; no training, real model construction, or data access."""
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from molgap import experiment_runner as runner
from molgap.experiment_spec import ExperimentSpec
from test_experiment_spec import payload as base_payload, addon


@pytest.fixture
def payload(base_payload):
    base_payload["platform"]["device_count"] = 2
    for arm in base_payload["arms"]:
        arm["initialization"]["kind"] = "random"
    return base_payload


@pytest.fixture
def spec(payload):
    return ExperimentSpec(payload)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def test_probe_isolated_order_identity_and_canonical(spec, tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "undeclared")
    root = tmp_path / "run"
    summary = runner.run_experiment_arms(spec, root, ["0", "1"])
    assert summary["status"] == "SUCCEEDED"
    assert summary["arm_ids"] == [a["arm_id"] for a in spec.to_dict()["arms"]]
    results = [read(a["result_path"]) for a in summary["arms"]]
    assert len({r["observed_pid"] for r in results} | {os.getpid()}) == 3
    for index, (record, result) in enumerate(zip(summary["arms"], results)):
        assert record["exit_code"] == 0
        assert result["spec_identity"] == spec.identity
        assert result["arm_id"] == summary["arm_ids"][index]
        assert result["observed_cuda_visible_devices"] == str(index)
        assert result["metadata"]["family"]["version"] == "1"
        assert result["metadata"]["arm_identity"]
        assert result["metadata"]["factory"]
        assert result["metadata"]["checkpoint_resume_owner"]
        assert result["metadata"]["checkpoint_resume_implemented"] is False
        assert result["observed_duration_seconds"] >= 0
        assert "parameter_count" not in result
    for path in [root / "run_summary.json", *(Path(a["result_path"]) for a in summary["arms"])]:
        assert path.read_text(encoding="utf-8") == runner._canonical(read(path))
        assert not list(path.parent.glob(".pending-*"))
        text = path.read_text(encoding="utf-8")
        for forbidden in ("training_success", "rml_closed", "replay_ready", "expected_steps"):
            assert forbidden not in text
    before = (root / "run_summary.json").read_bytes()
    with pytest.raises(ValueError):
        runner.run_experiment_arms(spec, root, ["0", "1"])
    assert (root / "run_summary.json").read_bytes() == before


@pytest.mark.parametrize("cpu", [None, "CPU", "cpu"])
def test_cpu_masks_inherited_gpu(spec, tmp_path, monkeypatch, cpu):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "99")
    result = runner.run_experiment_arms(spec, tmp_path / "run", [cpu, "0"])
    arm = read(result["arms"][0]["result_path"])
    assert arm["observed_cuda_visible_devices"] == ""
    assert arm["device_binding"]["requested"] == cpu
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "99"


@pytest.mark.parametrize("family_index,variant", [(0, None), (0, "pair_prenorm"), (1, None), (1, "k1_pair_value")])
def test_construct_static_dispatch_with_fake_dependencies(payload, monkeypatch, family_index, variant):
    arm = payload["arms"][family_index]
    if variant:
        arm.update(addons=[addon(variant)], addon_semantics="ordered")
    spec = ExperimentSpec(payload)
    events = []
    monkeypatch.setitem(sys.modules, "numpy", SimpleNamespace(random=SimpleNamespace(seed=lambda n: events.append(("numpy", n)))))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(manual_seed=lambda n: events.append(("torch", n))))
    from molgap import gptrans_adapter, k1_adapter
    def fake_build(s, arm_id):
        assert events == [("numpy", 42), ("torch", 42)]
        assert s.identity == spec.identity and arm_id == arm["arm_id"]
        events.append(("build", arm_id))
        return SimpleNamespace(parameters=lambda: [SimpleNamespace(numel=lambda: 7)])
    def wrong(*args, **kwargs):
        pytest.fail("Wrong family adapter")
    monkeypatch.setattr(gptrans_adapter, "build_gptrans_model", fake_build if family_index == 0 else wrong)
    monkeypatch.setattr(k1_adapter, "build_k1_model", fake_build if family_index == 1 else wrong)
    metadata, count = runner._execute(spec, arm, "construct")
    assert count == 7
    assert metadata["family"]["name"] == arm["family"]["name"]
    assert metadata["addons"] == arm["addons"]
    if family_index == 0:
        assert metadata["variant"] == (variant or "reference")
    else:
        assert metadata["addon"] == ([variant, "1"] if variant else None)


def test_frozen_construct_fails_without_loading_torch(payload, tmp_path):
    for arm in payload["arms"]:
        arm["initialization"]["kind"] = "frozen_state"
    summary = runner.run_experiment_arms(ExperimentSpec(payload), tmp_path / "run", ["0", "1"], "construct")
    assert summary["status"] == "FAILED"
    for arm in summary["arms"]:
        assert arm["exit_code"] == 1
        result = read(arm["result_path"])
        assert "frozen_state" in result["error"]["message"]
        assert result["parameter_count"] is None


def fixture_launcher(monkeypatch, outcomes):
    # Test-only replacement of a private launcher; no production callback API.
    def launch(snapshot, arm_id, worker, binding, directory):
        code = '''import sys,json,time,os
sys.path.insert(0,sys.argv[1])
from molgap import experiment_runner as r
args=json.loads(sys.argv[2])
outcome=sys.argv[3]
if outcome == 'timeout': time.sleep(60)
if outcome == 'crash': os._exit(7)
original=r._execute
def execute(spec,arm,worker):
    if outcome == 'fail': raise RuntimeError('synthetic failure')
    return original(spec,arm,worker)
r._execute=execute
sys.exit(r._child_main(**args))
'''
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=binding["cuda_visible_devices"])
        args = dict(snapshot=snapshot, arm_id=arm_id, worker=worker, binding=binding, directory=str(directory))
        return subprocess.Popen([sys.executable, "-c", code, str(Path(runner.__file__).parents[1]), json.dumps(args), outcomes[arm_id]], env=env)
    monkeypatch.setattr(runner, "_launch", launch)


@pytest.mark.parametrize("outcomes,aggregate,codes", [
    (("pass", "fail"), "PARTIAL_FAILURE", (0, 1)),
    (("fail", "fail"), "FAILED", (1, 1)),
    (("pass", "crash"), "PARTIAL_FAILURE", (0, 7)),
    (("pass", "timeout"), "TIMED_OUT", (0, None)),
])
def test_independent_failure_timeout_and_exit_codes(spec, tmp_path, monkeypatch, outcomes, aggregate, codes):
    fixture_launcher(monkeypatch, dict(zip([a["arm_id"] for a in spec.to_dict()["arms"]], outcomes)))
    summary = runner.run_experiment_arms(spec, tmp_path / "run", ["0", "1"], timeout_seconds=10)
    assert summary["status"] == aggregate
    for record, outcome, code in zip(summary["arms"], outcomes, codes):
        if code is not None:
            assert record["exit_code"] == code
        else:
            assert record["exit_code"] is not None
            assert record["status"] == "TIMED_OUT"
        if outcome in ("timeout", "crash"):
            assert record["result_path"] is None
            assert not Path(record["expected_result_path"]).exists()
        else:
            child = read(record["result_path"])
            assert child["status"] == ("SUCCEEDED" if outcome == "pass" else "FAILED")
            assert set(Path(record["result_path"]).parent.iterdir()) == {Path(record["result_path"])}


def test_launch_failure_does_not_cancel_peer(spec, tmp_path, monkeypatch):
    original = runner._launch
    def launch(snapshot, arm_id, *args):
        if arm_id == spec.to_dict()["arms"][0]["arm_id"]:
            raise OSError("synthetic spawn failure")
        return original(snapshot, arm_id, *args)
    monkeypatch.setattr(runner, "_launch", launch)
    summary = runner.run_experiment_arms(spec, tmp_path / "run", ["0", "1"])
    assert summary["status"] == "PARTIAL_FAILURE"
    assert summary["arms"][0]["exit_code"] is None
    assert summary["arms"][1]["exit_code"] == 0


@pytest.mark.parametrize("devices", [[], ["0"], ["0", "0"], [None, "cpu"], ["00", "1"], ["0,1", "2"], ["", "1"], [0, "1"], ["-1", "1"], ("0", "1")])
def test_invalid_devices_before_launch(spec, tmp_path, monkeypatch, devices):
    monkeypatch.setattr(runner, "_launch", lambda *a: pytest.fail("Must reject before launch"))
    with pytest.raises(ValueError):
        runner.run_experiment_arms(spec, tmp_path / "run", devices)
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("mutation", ["count", "duplicate", "case", "reserved", "trailing", "provider", "forged", "method", "noncanonical"])
def test_invalid_spec_or_arm_path_before_launch(payload, tmp_path, monkeypatch, mutation):
    monkeypatch.setattr(runner, "_launch", lambda *a: pytest.fail("Must reject before launch"))
    spec = ExperimentSpec(payload)
    if mutation == "count": payload["platform"]["device_count"] = 1
    if mutation == "duplicate": payload["arms"][1]["arm_id"] = payload["arms"][0]["arm_id"]
    if mutation == "case": payload["arms"][1]["arm_id"] = payload["arms"][0]["arm_id"].upper()
    if mutation == "reserved": payload["arms"][0]["arm_id"] = "CON.txt"
    if mutation == "trailing": payload["arms"][0]["arm_id"] = "arm."
    with pytest.raises((TypeError, ValueError)):
        if mutation == "provider": spec = SimpleNamespace(to_json=lambda: spec.to_json())
        elif mutation == "forged": object.__setattr__(spec, "_canonical_json", '{}')
        elif mutation == "method": object.__setattr__(spec, "to_json", lambda: "{}")
        elif mutation == "noncanonical": object.__setattr__(spec, "_canonical_json", json.dumps(payload, indent=2))
        else: spec = ExperimentSpec(payload)
        runner.run_experiment_arms(spec, tmp_path / "run", ["0", "1"])


@pytest.mark.parametrize("worker", ["unknown", "molgap.k1_adapter", None, lambda: None])
def test_unknown_worker(spec, tmp_path, worker):
    with pytest.raises(ValueError):
        runner.run_experiment_arms(spec, tmp_path / "run", ["0", "1"], worker)


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan"), True, "1"])
def test_invalid_timeout(spec, tmp_path, timeout):
    with pytest.raises(ValueError):
        runner.run_experiment_arms(spec, tmp_path / "run", ["0", "1"], timeout_seconds=timeout)


@pytest.mark.parametrize("case", ["existing_arm", "file", "traversal", "source", "data", "package"])
def test_unsafe_output(spec, tmp_path, monkeypatch, case):
    root = tmp_path / "run"
    if case == "existing_arm": (root / "arms" / "gptrans_t").mkdir(parents=True)
    if case == "file": root.write_text("keep")
    if case == "traversal": root = tmp_path / "other" / ".." / "run"
    if case in ("source", "data"): root = tmp_path / case / "run"
    if case == "package":
        (tmp_path / "package_manifest.json").write_text("{}")
    monkeypatch.setattr(runner, "_launch", lambda *a: pytest.fail("Must reject before launch"))
    with pytest.raises(ValueError):
        runner.run_experiment_arms(spec, root, ["0", "1"])


def test_atomic_json_retains_previous_on_replace_failure(tmp_path, monkeypatch):
    path = tmp_path / "result.json"
    path.write_bytes(b"previous")
    def fail(*args): raise OSError("synthetic replace failure")
    monkeypatch.setattr(runner.os, "replace", fail)
    with pytest.raises(OSError): runner._atomic_json(path, {"status": "FAILED"})
    assert path.read_bytes() == b"previous"
    assert not list(tmp_path.glob(".pending-*"))
