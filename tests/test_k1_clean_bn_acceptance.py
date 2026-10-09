"""Synthetic saved-artifact acceptance; no checkpoints, cache or role data."""
import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from molgap.training_reproducibility import atomic_json, atomic_torch_save, canonical_fingerprint, sha256_file
from molgap.v4_runtime import state_dict_sha256


@pytest.fixture
def acceptance():
    path = Path(__file__).resolve().parents[1] / "experiments/pcqm_k1_consistency_ablation_500k/clean_bn/accept.py"
    spec = importlib.util.spec_from_file_location("synthetic_bn_acceptance", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def report(acceptance, monkeypatch):
    monkeypatch.setattr(acceptance, "validate_inputs", lambda inputs: inputs)
    settings = copy.deepcopy(acceptance.SETTINGS)
    indices = np.random.default_rng(20261008).choice(500000, 16384, replace=False)
    runtime = {"accelerator": None, "determinism": {
        "device": "cpu", "cuda_device_count_visible": 0, "precision": "fp32", "tf32_enabled": False,
        "seed": 20261008, "deterministic_algorithms": True, "float32_matmul_precision": "highest"},
        "python": "synthetic", "python_executable": "synthetic", "platform": "synthetic", "torch": "synthetic",
        "installed_distributions": ["synthetic==1"], "installed_distributions_sha256": "a" * 64}
    runtime["runtime_fingerprint"] = canonical_fingerprint(runtime)
    base = {"status": "complete", "phase": "complete", "worker_exitcode": 0,
        "parent_observed_wall_seconds": 5, "wall_seconds": 4, "process_cpu_seconds": 3,
        "reconstruction_max_abs_eV": 1e-4, "device": "cpu", "cpu_threads": 4, "autocast_enabled": False,
        "runtime": runtime, "restored_prediction_exact": True,
        "calibration_source_idx_sha256": state_dict_sha256({"source_idx": torch.as_tensor(indices)}),
        "development_source_idx_sha256": state_dict_sha256({"source_idx": torch.arange(500000, 550000)}),
        "parameter_sha256_before": "a" * 64, "parameter_sha256_calibrated": "a" * 64,
        "parameter_sha256_restored": "a" * 64, "buffer_sha256_before": "b" * 64, "buffer_sha256_restored": "b" * 64,
        "calibration": {"rows": 16384, "batches": 128, "bn_modules": 1,
            "parameters_unchanged": True, "non_bn_buffers_unchanged": True, "dropout_disabled": True,
            "buffers_restored": True, "labels_used_for_calibration": False, "buffer_sha256_before": "b" * 64}}
    return {"status": "complete", "wall_seconds": 10, "inputs": {"settings": settings},
            "arms": {arm: {**copy.deepcopy(base), "arm": arm} for arm in acceptance.ARMS}}


def test_matching_report_and_frozen_inputs_accepted(acceptance, report):
    acceptance.validate_report(report, copy.deepcopy(report["inputs"]))


@pytest.mark.parametrize("field,value", [
    ("reconstruction_max_abs_eV", 0.0001001), ("reconstruction_max_abs_eV", float("nan")),
    ("parent_observed_wall_seconds", 601), ("wall_seconds", -1), ("device", "cuda"),
    ("cpu_threads", 8), ("autocast_enabled", True), ("arm", "wrong"),
    ("calibration_source_idx_sha256", "0" * 64), ("development_source_idx_sha256", "0" * 64),
    ("parameter_sha256_calibrated", "0" * 64), ("buffer_sha256_restored", "0" * 64),
    ("restored_prediction_exact", False),
])
def test_bad_arm_assertion_rejected(acceptance, report, field, value):
    report["arms"][acceptance.ARMS[0]][field] = value
    with pytest.raises(ValueError):
        acceptance.validate_report(report, report["inputs"])


@pytest.mark.parametrize("field,value", [
    ("cuda_device_count_visible", 1), ("tf32_enabled", True), ("precision", "bf16"),
    ("seed", 42), ("deterministic_algorithms", False),
])
def test_runtime_cpu_contract_rejected_even_with_consistent_fingerprint(acceptance, report, field, value):
    runtime = report["arms"][acceptance.ARMS[0]]["runtime"]
    runtime["determinism"][field] = value
    runtime["runtime_fingerprint"] = canonical_fingerprint({key: value for key, value in runtime.items() if key != "runtime_fingerprint"})
    with pytest.raises(ValueError, match="runtime assertions"):
        acceptance.validate_report(report, report["inputs"])


def test_input_provenance_settings_and_validator_enforced(acceptance, report, monkeypatch):
    inputs = copy.deepcopy(report["inputs"])
    report["inputs"]["settings"]["sample_seed"] = 42
    with pytest.raises(ValueError, match="provenance"):
        acceptance.validate_report(report, inputs)
    def rejected(inputs):
        raise ValueError("synthetic stale source pin")
    monkeypatch.setattr(acceptance, "validate_inputs", rejected)
    with pytest.raises(ValueError, match="stale source pin"):
        acceptance.validate_report(report, inputs)


def test_pair_wall_and_runtime_fingerprint_enforced(acceptance, report):
    report["wall_seconds"] = 1201
    with pytest.raises(ValueError, match="pair wall"):
        acceptance.validate_report(report, report["inputs"])
    report["wall_seconds"] = 10
    report["arms"][acceptance.ARMS[0]]["runtime"]["torch"] = "changed"
    with pytest.raises(ValueError, match="fingerprint"):
        acceptance.validate_report(report, report["inputs"])


@pytest.fixture
def artifacts(tmp_path):
    original = {"source_idx": torch.arange(500000, 550000), "target_eV": torch.zeros(50000), "prediction_eV": torch.ones(50000)}
    buffers = {"bn.running_mean": torch.ones(2), "bn.running_var": torch.ones(2), "bn.num_batches_tracked": torch.tensor(128)}
    for name in ("original", "calibrated", "restored"):
        atomic_torch_save(tmp_path / f"{name}.pt", original)
    atomic_torch_save(tmp_path / "calibrated_buffers.pt", buffers)
    observation = {"artifacts": {path.name: sha256_file(path) for path in tmp_path.glob("*.pt")},
        "calibration": {"buffer_sha256_calibrated": state_dict_sha256(buffers)}}
    return tmp_path, observation


def test_four_artifacts_and_calibrated_buffer_state_accepted(acceptance, artifacts):
    directory, observation = artifacts
    assert set(acceptance.load_artifacts(directory, observation)) == {"original", "calibrated", "restored"}


@pytest.mark.parametrize("change", ["empty", "missing", "extra", "bufferhash", "filehash", "restoration", "target", "shape"])
def test_artifact_validation_fail_closed(acceptance, artifacts, change):
    directory, observation = artifacts
    if change == "empty":
        observation["artifacts"] = {}
    elif change == "missing":
        observation["artifacts"].pop("calibrated_buffers.pt")
    elif change == "extra":
        observation["artifacts"]["../outside.pt"] = "a" * 64
    elif change == "bufferhash":
        observation["calibration"]["buffer_sha256_calibrated"] = "0" * 64
    elif change == "filehash":
        observation["artifacts"]["original.pt"] = "0" * 64
    else:
        name = "restored.pt" if change == "restoration" else "calibrated.pt"
        path = directory / name
        record = torch.load(path, weights_only=True)
        if change == "restoration":
            record["prediction_eV"][0] += 1e-7
        elif change == "target":
            record["target_eV"][0] += 1
        else:
            record["prediction_eV"] = record["prediction_eV"].view(-1, 1)
        atomic_torch_save(path, record)
        observation["artifacts"][name] = sha256_file(path)
    with pytest.raises(ValueError):
        acceptance.load_artifacts(directory, observation)


@pytest.mark.parametrize("field", [None, "epoch_or_pass", "optimizer_step", "sample_presentations"])
def test_every_epoch_step_and_presentation_is_validated(acceptance, field):
    trace = {"observations": [{"epoch_or_pass": index, "optimizer_step": 3906 * (index + 1),
                              "sample_presentations": 499968 * (index + 1)} for index in range(60)]}
    if field is None:
        acceptance.validate_exposure(trace)
    else:
        trace["observations"][17][field] = None
        with pytest.raises(ValueError, match="exposure"):
            acceptance.validate_exposure(trace)


@pytest.mark.parametrize("gain,interval,passes", [
    (-0.002, [-0.003, -0.001], False), (0.0009, [0.0001, 0.0015], False),
    (0.002, [-0.0001, 0.003], False), (0.001, [0.0001, 0.002], True),
])
def test_primary_consistency_gate_remains_one_sided(acceptance, gain, interval, passes):
    record = {"paired_gain_eV": gain, "row_bootstrap_95pct_eV": interval}
    acceptance.add_gate_fields(record, primary=True)
    assert record["primary_consistency_gate_passed"] is passes
    assert record["original_1mev_point_gate_passed"] is (abs(gain) >= 0.001)


@pytest.mark.parametrize("gain,interval,material,removal", [
    (-0.001, [-0.002, -0.0001], True, True),
    (0.001, [0.0001, 0.002], True, False),
    (-0.002, [-0.003, 0.0001], False, False),
    (-0.0009, [-0.0015, -0.0001], False, False),
    (0.0009, [0.0001, 0.0015], False, False),
    (-0.001, [-0.002, 0.0], False, False),
])
def test_original_material_nomination_requires_absolute_gain_and_aligned_ci(
        acceptance, gain, interval, material, removal):
    record = {"paired_gain_eV": gain, "row_bootstrap_95pct_eV": interval}
    acceptance.add_gate_fields(record, primary=True)
    assert record["original_material_nomination_gate_passed"] is material
    assert record["primary_removal_gate_passed"] is removal
    if gain < 0:
        assert record["primary_consistency_gate_passed"] is False


def test_secondary_removal_not_labeled_primary_promotion(acceptance):
    record = {"paired_gain_eV": -0.002, "row_bootstrap_95pct_eV": [-0.003, -0.001]}
    acceptance.add_gate_fields(record, primary=False)
    assert record["secondary_removal_direction_gate_passed"] is True
    assert record["secondary_consistency_gate_passed"] is False
    assert record["primary_promotion_gate_applicable"] is False
    assert "primary_consistency_gate_passed" not in record
