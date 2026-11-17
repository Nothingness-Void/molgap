import pytest
import torch

from molgap.experiment_family_workflow import _finite_tree
from molgap.training_reproducibility import (
    _batch_finite_tensor_failures,
    assert_finite_state_dict,
    finite_validation_scope,
)


def test_finite_tree_accepts_mixed_supported_tensor_kinds_and_aliases() -> None:
    base = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    shared = base[:, ::2]
    state = {
        "model": {"weights": shared, "alias": shared},
        "optimizer": {
            "step": torch.tensor(2, dtype=torch.int64),
            "flag": torch.tensor(True, dtype=torch.bool),
            "empty": torch.empty(0, dtype=torch.float64),
            "complex": torch.tensor([1 + 2j], dtype=torch.complex64),
        },
        "cursor": [1.0, (2.0,)],
    }

    _finite_tree(state, "checkpoint")
    assert_finite_state_dict(state["model"], label="model")


def test_finite_validation_scope_reuses_only_unchanged_successful_tensors(monkeypatch) -> None:
    first = torch.ones(2)
    second = torch.zeros(2)
    original_isfinite = torch.isfinite
    calls = 0

    def counted_isfinite(value):
        nonlocal calls
        calls += 1
        return original_isfinite(value)

    monkeypatch.setattr(torch, "isfinite", counted_isfinite)
    with finite_validation_scope():
        _finite_tree({"first": first, "second": second}, "checkpoint")
        assert_finite_state_dict({"first": first, "second": second}, label="model")
        assert calls == 2
        first.add_(1.0)
        _finite_tree({"first": first}, "checkpoint")
        assert calls == 3
    _finite_tree({"first": first}, "checkpoint")
    assert calls == 4


def test_finite_validation_scope_does_not_cache_inference_tensors(monkeypatch) -> None:
    with torch.inference_mode():
        value = torch.ones(2)
    original_isfinite = torch.isfinite
    calls = 0

    def counted_isfinite(tensor):
        nonlocal calls
        calls += 1
        return original_isfinite(tensor)

    monkeypatch.setattr(torch, "isfinite", counted_isfinite)
    with finite_validation_scope():
        _finite_tree({"value": value}, "checkpoint")
        _finite_tree({"value": value}, "checkpoint")
    assert calls == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for device-batched validation")
def test_accelerated_validation_batches_host_observation_and_handles_mixed_inputs(monkeypatch) -> None:
    base = torch.arange(24, device="cuda", dtype=torch.float32).reshape(4, 6)
    shared_view = base[:, ::2]
    entries = [
        (0, "cpu", torch.ones(2)),
        (1, "gpu-noncontiguous", shared_view),
        (2, "gpu-empty", torch.empty(0, device="cuda", dtype=torch.complex64)),
        (3, "gpu-integer", torch.ones(2, device="cuda", dtype=torch.int64)),
        (4, "gpu-alias", shared_view),
    ]
    original_isfinite = torch.isfinite
    calls = 0

    def counted_isfinite(value):
        nonlocal calls
        calls += 1
        return original_isfinite(value)

    monkeypatch.setattr(torch, "isfinite", counted_isfinite)
    assert _batch_finite_tensor_failures(entries) == []
    # The repeated alias is checked once; no failure means no diagnostic pass.
    assert calls == 3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for device-batched validation")
def test_accelerated_validation_reports_first_failure_path() -> None:
    state = {
        "model": {"finite": torch.ones(2, device="cuda")},
        "optimizer": {
            "first_bad": torch.tensor([1.0, float("nan")], device="cuda"),
            "later_bad": torch.tensor([float("inf")], device="cuda"),
        },
    }

    with pytest.raises(ValueError, match=r"Nonfinite checkpoint state: checkpoint\.optimizer\.first_bad"):
        _finite_tree(state, "checkpoint")
