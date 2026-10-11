"""JSON transport regression checks; no accelerators or scientific inputs."""
import json
from collections import UserDict

import pytest
import torch

from molgap import training_reproducibility as owner


@pytest.mark.parametrize("value", [{"rows": []}, UserDict({"epoch": 1}), [],
                                 [{"epoch": 1, "train_seconds": 2.5}]])
def test_atomic_json_roundtrip(tmp_path, value):
    path = tmp_path / "nested" / "execution_times.json"
    owner.atomic_json(path, value)
    expected = dict(value) if isinstance(value, UserDict) else value
    assert json.loads(path.read_text()) == expected
    assert path.read_text() == json.dumps(expected, indent=2) + "\n"
    assert not path.with_name("." + path.name + ".tmp").exists()


@pytest.mark.parametrize("value", ["text", None, 1, (("epoch", 1),),
                                 [{"unserializable": object()}]])
def test_bad_json_preserves_previous_file(tmp_path, value):
    path = tmp_path / "timings.json"
    owner.atomic_json(path, {"previous": True})
    before = path.read_bytes()
    with pytest.raises(TypeError):
        owner.atomic_json(path, value)
    assert path.read_bytes() == before
    assert not path.with_name("." + path.name + ".tmp").exists()


def test_replace_failure_preserves_previous_file_and_cleans_temp(tmp_path, monkeypatch):
    path = tmp_path / "timings.json"
    owner.atomic_json(path, {"previous": True})
    before = path.read_bytes()

    def failed_replace(*args):
        raise OSError("synthetic replace failure")

    monkeypatch.setattr(owner.os, "replace", failed_replace)
    with pytest.raises(OSError, match="synthetic replace failure"):
        owner.atomic_json(path, [{"epoch": 1}])
    assert path.read_bytes() == before
    assert not path.with_name("." + path.name + ".tmp").exists()


@pytest.mark.parametrize("failure", ["serialize", "replace"])
def test_atomic_checkpoint_failure_preserves_previous_file(tmp_path, monkeypatch, failure):
    path = tmp_path / "checkpoint.pt"
    owner.atomic_torch_save(path, {"weight": torch.ones(2)})
    before = path.read_bytes()

    def fail_save(value, temporary):
        temporary.write_bytes(b"partial checkpoint")
        raise RuntimeError("synthetic torch serialization failure")

    def fail_replace(*args):
        raise OSError("synthetic checkpoint replace failure")

    if failure == "serialize":
        monkeypatch.setattr(torch, "save", fail_save)
    else:
        monkeypatch.setattr(owner.os, "replace", fail_replace)
    with pytest.raises((RuntimeError, OSError), match="synthetic"):
        owner.atomic_torch_save(path, {"weight": torch.zeros(2)})
    assert path.read_bytes() == before
    assert not path.with_name("." + path.name + ".tmp").exists()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_batched_metrics_keep_original_addition_order_rng_and_single_transfer(monkeypatch, dtype):
    # This cancellation is lost by a tensor reduction or compensated Python sum.
    values = [torch.tensor(v, dtype=dtype, requires_grad=True) for v in (1e20, 1., -1e20, 3.)]
    expected = 0.0
    for value in values:
        expected += float(value.detach())
    transfers = []
    original = torch.Tensor.cpu

    def transfer(value, *args, **kwargs):
        transfers.append(tuple(value.shape))
        return original(value, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "cpu", transfer)
    before = torch.get_rng_state().clone()
    assert owner.sum_scalar_metrics(values) == expected == 3.
    assert transfers == [(4,)]
    assert torch.equal(before, torch.get_rng_state())
    assert all(v.grad is None for v in values)


def test_host_metrics_use_the_same_sequential_sum():
    assert owner.sum_scalar_metrics([1e20, 1., -1e20, 3.]) == 3.
    assert owner.sum_scalar_metrics([]) == 0.


@pytest.mark.parametrize("values", [[True], [float("nan")], [float("inf")],
    [torch.ones(2)], [torch.tensor(1), torch.tensor(2)],
    [torch.tensor(1.), 2.], [2., torch.tensor(1.)],
    [torch.tensor(1., dtype=torch.float32), torch.tensor(2., dtype=torch.float64)],
    [torch.tensor(float("nan"))], [1e308, 1e308]])
def test_metrics_reject_invalid_or_nonfinite_payloads(values):
    with pytest.raises(ValueError):
        owner.sum_scalar_metrics(values)


def test_tensor_only_rng_snapshot_restores_cpu_next_draws(tmp_path):
    import random
    import numpy as np
    from molgap.experiment_family_workflow import tensor_safe_rng_state

    original = owner.capture_rng_state()
    try:
        path = tmp_path / "diagnostic_rng.pt"
        owner.atomic_torch_save(path, {"rng_state": tensor_safe_rng_state(original)})
        expected = (random.random(), np.random.random(), torch.rand(3))
        saved = torch.load(path, map_location="cpu", weights_only=True)
        owner.restore_rng_state(saved["rng_state"])
        actual = (random.random(), np.random.random(), torch.rand(3))
        assert actual[:2] == expected[:2]
        assert torch.equal(actual[2], expected[2])
    finally:
        owner.restore_rng_state(original)


def test_finite_state_check_keeps_original_named_errors_and_does_not_mutate_state():
    state = {"good": torch.tensor([1., 2.], requires_grad=True),
             "nan": torch.tensor([float("nan")]), "metadata": "ignored",
             "inf": torch.tensor([float("inf")]), "integer": torch.tensor([1])}
    before = state["good"].clone()
    with pytest.raises(RuntimeError, match=r"model contains non-finite tensors: \['nan', 'inf'\]"):
        owner.assert_finite_state_dict(state, label="model")
    assert torch.equal(before, state["good"]) and state["good"].grad is None
    owner.assert_finite_state_dict({"empty": torch.empty(0), "good": state["good"]}, label="model")


def test_device_finite_flags_use_one_transfer_per_device_and_preserve_key_order(monkeypatch):
    # Simulate accelerator routing with CPU boolean flags; no CUDA allocation.
    class DeviceValue:
        def __init__(self, device, finite):
            self.device, self.finite = torch.device(device), finite

    state = {"gpu0_bad": DeviceValue("cuda:0", False),
             "cpu_bad": DeviceValue("cpu", False),
             "gpu1_good": DeviceValue("cuda:1", True),
             "gpu0_good": DeviceValue("cuda:0", True),
             "gpu1_bad": DeviceValue("cuda:1", False)}
    original_cpu = torch.Tensor.cpu
    calls = []

    def transfer(value, *args, **kwargs):
        calls.append(tuple(value.shape))
        return original_cpu(value, *args, **kwargs)

    monkeypatch.setattr(torch, "is_tensor", lambda value: isinstance(value, DeviceValue))
    monkeypatch.setattr(torch, "isfinite", lambda value: torch.tensor(value.finite))
    monkeypatch.setattr(torch.Tensor, "cpu", transfer)
    with pytest.raises(RuntimeError, match=r"\['gpu0_bad', 'cpu_bad', 'gpu1_bad'\]"):
        owner.assert_finite_state_dict(state, label="model")
    assert calls == [(2,), (2,)]
