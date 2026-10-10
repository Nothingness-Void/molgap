"""Synthetic CPU equivalence to the retained trainer's _cpu, not trainer execution."""
import builtins
import copy
import io
import random
from collections import OrderedDict

import numpy as np
import pytest
import torch

from molgap.checkpoint_snapshot import checkpoint_to_cpu


def original_cpu(value):
    # Exact helper semantics from colab_k1_screen.py:105-113, owner commit
    # 0bdb1adfec5085d8a32799d60a29f9db4cff5812; never import/run that trainer.
    import torch
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: original_cpu(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(original_cpu(item) for item in value)
    return copy.deepcopy(value)


class ScalarSubclass(int):
    def __deepcopy__(self, memo):
        result = type(self)(int(self))
        result.was_deepcopied = True
        return result


class ListSubclass(list):
    pass


class TupleSubclass(tuple):
    pass


class Metadata:
    def __init__(self):
        self.items = [1, {"metadata": np.arange(3)}]
        self.alias = self.items


def assert_exact(left, right):
    assert type(left) is type(right)
    if torch.is_tensor(left):
        assert left.dtype == right.dtype and left.device == right.device
        assert left.shape == right.shape and left.stride() == right.stride()
        assert left.requires_grad == right.requires_grad
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        assert left.dtype == right.dtype
        np.testing.assert_array_equal(left, right)
        if left.dtype == object:
            for a, b in zip(left.flat, right.flat):
                assert_exact(a, b)
    elif isinstance(left, dict):
        assert list(left) == list(right)
        for key in left:
            assert_exact(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_exact(a, b)
    elif isinstance(left, Metadata):
        assert_exact(left.__dict__, right.__dict__)
    else:
        assert left == right
        if hasattr(left, "__dict__"):
            assert_exact(left.__dict__, right.__dict__)


def payload_fixture():
    tensor = torch.arange(24, dtype=torch.float32).view(4, 6).t().requires_grad_(True)
    array = np.arange(12, dtype=np.float64).reshape(3, 4)
    step = torch.tensor(128.0, dtype=torch.float32, device="cpu")
    return {
        "model": OrderedDict(weight=tensor, running_mean=torch.arange(4, dtype=torch.float64),
                             num_batches_tracked=torch.tensor(2, dtype=torch.int64)),
        "optimizer": {"state": {0: {"step": step, "exp_avg": tensor.detach() * 0.1,
                                  "exp_avg_sq": tensor.detach().square()}},
                      "param_groups": [{"params": [0], "lr": 4e-4, "betas": (0.9, 0.999),
                                        "eps": 1e-8, "weight_decay": 1e-5}]},
        "step_trace": [{"step": i, "loss": i / 16, "finite": True, "tag": None}
                       for i in range(16)],
        "repeat_tensor": [tensor, tensor], "repeat_array": [array, array],
        "subcontainers": [ListSubclass([tensor]), TupleSubclass((array,))],
        "metadata": Metadata(), "subscalar": ScalarSubclass(7),
        ("arbitrary", 3): {b"binary key": complex(1, 2), None: "value"},
        "scalars": (None, True, 123, -0.5, complex(2, 3), "name", b"bytes"),
        "array_objects": np.array([{"items": [1, 2]}], dtype=object),
    }


def decoded(value):
    stream = io.BytesIO()
    torch.save(value, stream)
    stream.seek(0)
    return torch.load(stream, map_location="cpu", weights_only=False)


def test_complete_decoded_payload_exact_and_cpu_step_scalar_preserved():
    payload = payload_fixture()
    expected, actual = original_cpu(payload), checkpoint_to_cpu(payload)
    assert_exact(expected, actual)
    assert_exact(decoded(expected), decoded(actual))
    step = actual["optimizer"]["state"][0]["step"]
    assert step.shape == torch.Size([])
    assert step.dtype is torch.float32 and step.device.type == "cpu"
    assert step.item() == 128.0
    assert type(actual["model"]) is dict
    assert type(actual["subcontainers"][0]) is ListSubclass
    assert type(actual["subcontainers"][1]) is TupleSubclass


def test_independent_tensors_buffers_arrays_containers_and_fallback_aliases():
    source = payload_fixture()
    snapshot = checkpoint_to_cpu(source)
    original = original_cpu(source)
    assert snapshot is not source
    for field in ("weight", "running_mean", "num_batches_tracked"):
        a, b = source["model"][field], snapshot["model"][field]
        assert a.data_ptr() != b.data_ptr()
        assert not b.requires_grad and b.grad_fn is None
    a, b = snapshot["repeat_tensor"]
    assert a.data_ptr() != b.data_ptr()
    with torch.no_grad():
        source["model"]["weight"].add_(10)
    source["model"]["running_mean"].add_(10)
    source["repeat_array"][0][:] = -1
    source["step_trace"][0]["loss"] = 100
    source["metadata"].items[1]["metadata"][:] = -1
    source["array_objects"][0]["items"].append(3)
    assert_exact(original, snapshot)
    assert not np.shares_memory(snapshot["repeat_array"][0], snapshot["repeat_array"][1])
    assert snapshot["metadata"].items is snapshot["metadata"].alias
    snapshot["repeat_tensor"][0].add_(100)
    assert torch.equal(snapshot["repeat_tensor"][1], original["repeat_tensor"][1])


@pytest.mark.parametrize("value", [None, True, 7, 1.25, complex(1, 2), "text", b"raw"])
def test_exact_builtin_scalar_root_needs_no_torch_import_or_deepcopy(value, monkeypatch):
    import_function = builtins.__import__
    def restricted(name, *args, **kwargs):
        if name == "torch":
            raise AssertionError("Scalar root must not import torch")
        return import_function(name, *args, **kwargs)
    def forbidden(*args, **kwargs):
        raise AssertionError("Exact immutable scalar must not use deepcopy")
    monkeypatch.setattr(builtins, "__import__", restricted)
    monkeypatch.setattr(copy, "deepcopy", forbidden)
    assert checkpoint_to_cpu(value) is value


def test_subclass_fallback_and_single_import_per_snapshot(monkeypatch):
    imports = []
    import_function = builtins.__import__
    def counted(name, *args, **kwargs):
        if name == "torch":
            imports.append(name)
        return import_function(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", counted)
    value = ScalarSubclass(9)
    copied = checkpoint_to_cpu({"nested": [1, None, (2.5, b"bytes", value)]})
    assert imports == ["torch"]
    scalar = copied["nested"][2][2]
    assert type(scalar) is ScalarSubclass and scalar is not value
    assert scalar.was_deepcopied is True


def test_arbitrary_metadata_keys_are_retained_not_deepcopied():
    key = Metadata()
    value = checkpoint_to_cpu(OrderedDict([(key, {"items": [1, 2]}), (42, "number")]))
    assert type(value) is dict
    assert next(iter(value)) is key
    assert list(value) == [key, 42]


def test_cpu_snapshot_never_calls_cuda_or_changes_rng(monkeypatch):
    payload = payload_fixture()
    python_rng, numpy_rng, torch_rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
    def forbidden(*args, **kwargs):
        raise AssertionError("CPU snapshot must not call CUDA")
    for name in ("_lazy_init", "init", "is_available", "current_device", "synchronize",
                 "get_rng_state", "get_rng_state_all"):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    monkeypatch.setattr(torch.Tensor, "cuda", forbidden)
    checkpoint_to_cpu(payload)
    assert python_rng == random.getstate()
    after_numpy = np.random.get_state()
    assert numpy_rng[0] == after_numpy[0]
    np.testing.assert_array_equal(numpy_rng[1], after_numpy[1])
    assert numpy_rng[2:] == after_numpy[2:]
    assert torch.equal(torch_rng, torch.get_rng_state())
