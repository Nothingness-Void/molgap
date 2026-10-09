"""Synthetic binding/lifecycle tests: no retained model, cache or roles read."""
import json
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch

from molgap import k1_bn_diagnostic as diagnostic
from molgap import k1_frozen_inference as inference
from molgap.training_reproducibility import atomic_json, atomic_torch_save, sha256_file
from molgap.v4_runtime import normalized_source_sha256


def _binding(path):
    return {"path": str(path), "sha256": sha256_file(path)}


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    from molgap import experiment_package
    monkeypatch.setattr(experiment_package, "verify_experiment_source_package", lambda path: {})
    root = tmp_path / "frozen"
    (root / "src/molgap").mkdir(parents=True)
    factory = root / "src/molgap/qm9_neural_atom.py"
    factory.write_bytes(b"# synthetic factory\r\n")
    protocol = tmp_path / "protocol.md"
    protocol.write_bytes(b"synthetic frozen protocol\r\n")
    name = "experiments/synthetic/protocol.md"
    protocol_pin = {"path": str(protocol), "sha256": normalized_source_sha256(protocol)}
    inventory = tmp_path / "SOURCE_FILES.json"
    atomic_json(inventory, {"format": "molgap-v4-source-inventory-v1", "files": [
        {"path": name, "sha256": protocol_pin["sha256"], "bytes": 26},
        {"path": "src/molgap/qm9_neural_atom.py", "sha256": normalized_source_sha256(factory), "bytes": 20}]})
    archive = tmp_path / "source.tar.gz"
    archive.write_bytes(b"synthetic archive; verifier mocked")
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"synthetic checkpoint; never loaded")
    arms = {}
    for arm in diagnostic.ARMS:
        trajectory = tmp_path / f"{arm}.json"
        trajectory_id = f"synthetic-{arm}"
        atomic_json(trajectory, {"record_mode": "prospective", "trajectory_id": trajectory_id,
                               "decision_state": {"source_hashes": {name: protocol_pin["sha256"]}}})
        arms[arm] = {"trajectory": _binding(trajectory), "trajectory_id": trajectory_id,
                     "checkpoint": _binding(checkpoint), "saved_predictions": _binding(checkpoint)}
    sources = [Path(diagnostic.__file__), Path(inference.__file__),
               Path(diagnostic.__file__).with_name("k1_bn_calibration.py")]
    return {"format": "molgap-k1-clean-bn-pair-v1", "settings": dict(diagnostic.SETTINGS),
            "protocol": protocol_pin, "source_archive": _binding(archive),
            "source_inventory": _binding(inventory), "frozen_source_root": str(root),
            "cache_root": str(tmp_path), "manifest_sha256": "a" * 64,
            "executed_source_files": {str(p): sha256_file(p) for p in sources}, "arms": arms}


def test_validation_accepts_lf_inventory_with_crlf_source(inputs):
    assert diagnostic.validate_inputs(inputs) is inputs


@pytest.mark.parametrize("field", ["protocol", "source_archive", "source_inventory"])
def test_stale_pin_rejected_before_worker_start(inputs, tmp_path, field):
    inputs[field]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="bytes differ"):
        diagnostic.run_clean_bn_pair(inputs, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_original_trajectory_pin_and_protocol_link_enforced(inputs):
    arm = diagnostic.ARMS[0]
    path = Path(inputs["arms"][arm]["trajectory"]["path"])
    data = json.loads(path.read_text())
    data["decision_state"]["source_hashes"] = {}
    atomic_json(path, data)
    inputs["arms"][arm]["trajectory"] = _binding(path)
    with pytest.raises(ValueError, match="prospective trajectory differs"):
        diagnostic.validate_inputs(inputs)


@pytest.mark.parametrize("change", [
    lambda inputs: inputs["settings"].update(cpu_threads=8),
    lambda inputs: inputs.update(extra=True),
    lambda inputs: inputs["arms"].pop(diagnostic.ARMS[0]),
    lambda inputs: inputs["executed_source_files"].clear(),
])
def test_input_contract_fail_closed(inputs, change):
    change(inputs)
    with pytest.raises(ValueError):
        diagnostic.validate_inputs(inputs)


def test_native_loader_rejects_unknown_arm_before_loading(monkeypatch):
    from molgap import training_reproducibility
    monkeypatch.setattr(training_reproducibility, "sha256_file", lambda p: pytest.fail("no file IO"))
    with pytest.raises(ValueError, match="Unknown native500K K1 arm"):
        inference.load_native500k_k1(Path("unused"), expected_sha256="a" * 64,
            expected_source_sha256="b" * 64, expected_epoch=48,
            checkpoint_kind="selected", expected_arm="other")


@pytest.mark.parametrize("arm", diagnostic.ARMS)
def test_native_loader_binds_both_allowed_arms(monkeypatch, arm):
    from molgap import training_reproducibility, v4_runtime
    state = {"contract": {"arm": arm, "parameters": 3658817,
             "benchmark_id": "pcqm-composed500k-dev50k-60pass-v1", "precision": "fp32",
             "geometry_used": False, "teacher_used": False}, "epoch": 48,
             "source_sha256": "b" * 64, "mean": 0.0, "std": 1.0,
             "model": {"weight": torch.ones(1)}}
    monkeypatch.setattr(training_reproducibility, "sha256_file", lambda p: "a" * 64)
    monkeypatch.setattr(v4_runtime, "torch_load_compat", lambda *a, **k: state)
    class Encoder:
        def load_state_dict(self, value, strict):
            assert strict
        def parameters(self):
            class Count:
                def numel(self):
                    return 3658817
            return [Count()]
        def state_dict(self):
            return state["model"]
        def eval(self):
            return self
        def requires_grad_(self, value):
            assert value is False
            return self
    monkeypatch.setattr(inference, "make_encoder", lambda mode: Encoder())
    _, metadata = inference.load_native500k_k1(Path("unused"), expected_sha256="a" * 64,
        expected_source_sha256="b" * 64, expected_epoch=48, checkpoint_kind="selected", expected_arm=arm)
    assert metadata["contract"]["arm"] == arm
    with pytest.raises(ValueError, match="contract differs"):
        inference.load_native500k_k1(Path("unused"), expected_sha256="a" * 64,
            expected_source_sha256="b" * 64, expected_epoch=48, checkpoint_kind="selected",
            expected_arm=diagnostic.ARMS[1 - diagnostic.ARMS.index(arm)])


@pytest.fixture
def arm_execution(inputs, tmp_path, monkeypatch):
    from molgap import k1_bn_calibration, training_reproducibility
    from torch_geometric import loader
    model = torch.nn.BatchNorm1d(1).eval().requires_grad_(False)
    original = {"source_idx": torch.arange(500000, 550000), "target_eV": torch.zeros(50000),
                "prediction_eV": torch.ones(50000)}
    saved_path = tmp_path / "saved.pt"
    atomic_torch_save(saved_path, {"source_idx": original["source_idx"],
        "target": original["target_eV"], "prediction": original["prediction_eV"]})
    arm = diagnostic.ARMS[0]
    inputs["arms"][arm]["saved_predictions"] = _binding(saved_path)
    monkeypatch.setattr(diagnostic, "_load_graphs", lambda *args: ([], []))
    monkeypatch.setattr(diagnostic, "_verify_frozen_modules", lambda inputs: None)
    monkeypatch.setattr(loader, "DataLoader", lambda *args, **kwargs: [])
    monkeypatch.setattr(inference, "load_native500k_k1", lambda *args, **kwargs: (model, {"mean": 0, "std": 1}))
    monkeypatch.setattr(training_reproducibility, "build_runtime_manifest", lambda settings: {"synthetic": True})
    calls = []
    def predict(*args, **kwargs):
        result = {name: value.clone() for name, value in original.items()}
        if len(calls) == 1:
            result["prediction_eV"] += 0.01
        calls.append(result)
        return result, {"wall_seconds": 0.0, "process_cpu_seconds": 0.0}
    monkeypatch.setattr(inference, "predict_clean", predict)
    @contextmanager
    def calibrate(*args, **kwargs):
        assert kwargs["source_bounds"] == (0, 500000)
        indices = kwargs["source_idx"]
        import numpy as np
        assert np.array_equal(indices, np.random.default_rng(20261008).choice(500000, 16384, replace=False))
        report = {}
        before = model.running_mean.clone()
        model.running_mean.fill_(3)
        try:
            yield report
        finally:
            model.running_mean.copy_(before)
            report["buffers_restored"] = True
    monkeypatch.setattr(k1_bn_calibration, "recalibrated_batch_norm", calibrate)
    output = tmp_path / "arm"
    output.mkdir()
    progress = []
    return inputs, arm, output, lambda phase, **fields: progress.append((phase, fields)), calls, progress


def test_arm_saves_three_predictions_buffers_and_row_hashes(arm_execution):
    inputs, arm, output, progress, calls, reports = arm_execution
    diagnostic._execute_arm(inputs, arm, output, float("inf"), progress)
    assert {path.name for path in output.iterdir()} == {
        "original.pt", "calibrated.pt", "restored.pt", "calibrated_buffers.pt"}
    assert torch.equal(calls[0]["prediction_eV"], calls[2]["prediction_eV"])
    assert reports[-1][1]["restored_prediction_exact"] is True
    rows = next(fields for phase, fields in reports if phase == "sample_bound")
    assert len(rows["calibration_source_idx_sha256"]) == 64
    assert len(rows["development_source_idx_sha256"]) == 64


def test_reconstruction_failure_preserves_original_only(arm_execution):
    inputs, arm, output, progress, calls, reports = arm_execution
    path = Path(inputs["arms"][arm]["saved_predictions"]["path"])
    saved = torch.load(path, weights_only=True)
    saved["prediction"] += 0.001
    atomic_torch_save(path, saved)
    inputs["arms"][arm]["saved_predictions"] = _binding(path)
    with pytest.raises(ValueError, match="reconstruct"):
        diagnostic._execute_arm(inputs, arm, output, float("inf"), progress)
    assert {path.name for path in output.iterdir()} == {"original.pt"}
    assert len(calls) == 1


def test_restoration_requires_exact_predictions(arm_execution, monkeypatch):
    inputs, arm, output, progress, calls, reports = arm_execution
    predictor = inference.predict_clean
    def changed(*args, **kwargs):
        result, timing = predictor(*args, **kwargs)
        if len(calls) == 3:
            result["prediction_eV"][0] += 1e-7
        return result, timing
    monkeypatch.setattr(inference, "predict_clean", changed)
    with pytest.raises(ValueError, match="not EXACT"):
        diagnostic._execute_arm(inputs, arm, output, float("inf"), progress)
    assert (output / "restored.pt").exists()


@pytest.mark.parametrize("timeout", [False, True])
def test_pair_stops_on_failed_first_arm_without_retry(inputs, tmp_path, monkeypatch, timeout):
    events = []
    class Worker:
        pid = 12
        exitcode = None if timeout else 1
        alive = timeout
        def __init__(self, target, args):
            self.args = args
        def start(self):
            events.append(self.args[1])
            atomic_json(Path(self.args[2]) / "report.json", {
                "status": "running" if timeout else "failed", "process_cpu_seconds": 0.25})
        def join(self, seconds=None):
            if seconds is not None:
                assert 0 <= seconds <= 600
        def is_alive(self):
            return self.alive
        def terminate(self):
            self.alive = False
            self.exitcode = -15
    class Context:
        Process = Worker
    monkeypatch.setattr(diagnostic.multiprocessing, "get_context", lambda method: Context())
    output = tmp_path / "pair"
    report = diagnostic.run_clean_bn_pair(inputs, output)
    assert report["status"] == "failed"
    assert events == [diagnostic.ARMS[0]]
    assert report["arms"][diagnostic.ARMS[0]]["process_cpu_seconds"] == 0.25
    assert not (output / diagnostic.ARMS[1]).exists()
    with pytest.raises(FileExistsError):
        diagnostic.run_clean_bn_pair(inputs, output)


def test_json_duplicate_keys_fail_closed(tmp_path):
    path = tmp_path / "input.json"
    path.write_text('{"arms": {}, "arms": {}}')
    with pytest.raises(ValueError, match="Duplicate"):
        diagnostic._json(path)
