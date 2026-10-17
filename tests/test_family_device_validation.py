"""CPU-only regressions for the staged-device FamilyKernel forward ABI."""
from types import SimpleNamespace

import pytest
import torch

from molgap.family_training_kernel import FamilyKernel


class _ReturningModel:
    """Small callable stub; it never allocates or claims an accelerator."""

    def __init__(self, output):
        self.output = output
        self.calls = 0

    def __call__(self, batch):
        self.calls += 1
        return self.output


def _kernel(requested_device):
    kernel = object.__new__(FamilyKernel)
    kernel.device = torch.device(requested_device)
    return kernel


def _staged_batch(rows=3, *, include_y=False):
    fields = {
        "x": torch.zeros((rows, 2), dtype=torch.float32),
        "num_graphs": rows,
    }
    if include_y:
        fields["y"] = torch.zeros(rows, dtype=torch.float32)
    return SimpleNamespace(**fields)


def _forward(kernel, model, batch):
    return FamilyKernel._model_forward(kernel, model, batch)


@pytest.mark.parametrize("requested_device", ["cpu", "cpu:0"])
@pytest.mark.parametrize("column_output", [False, True])
def test_forward_accepts_fp32_n_or_n1_against_staged_cpu(
    requested_device, column_output
):
    rows = 3
    values = torch.arange(rows, dtype=torch.float32)
    output = values[:, None] if column_output else values
    result = _forward(
        _kernel(requested_device),
        _ReturningModel(output),
        _staged_batch(rows),
    )

    assert result.shape == (rows,)
    assert result.dtype is torch.float32
    torch.testing.assert_close(result, values)


def test_inference_forward_uses_x_topology_when_y_is_not_staged():
    """Inference batches need no target tensor; x is the device authority."""
    rows = 2
    batch = _staged_batch(rows, include_y=False)
    model = _ReturningModel(torch.zeros(rows, dtype=torch.float32))

    result = _forward(_kernel("cpu:0"), model, batch)

    assert model.calls == 1
    assert result.device == batch.x.device


def test_cpu_synthetic_forward_reuses_staged_device_validation():
    """The CPU-synthetic public entry point must share the repaired forward."""
    rows = 2
    model = _ReturningModel(torch.zeros(rows, dtype=torch.float32))
    kernel = object.__new__(FamilyKernel)
    kernel.device = torch.device("cpu:0")
    kernel.recipe = SimpleNamespace(cpu_synthetic_only=True)
    kernel._qualification_admission = None
    kernel.model = model

    result = kernel.forward(_staged_batch(rows))

    assert model.calls == 1
    assert result.device == torch.device("cpu")


@pytest.mark.parametrize(
    "output",
    [
        torch.zeros(4, 2, dtype=torch.float32),
        torch.zeros(3, dtype=torch.float64),
        torch.tensor([0.0, float("nan"), 1.0], dtype=torch.float32),
        [0.0, 1.0, 2.0],
    ],
    ids=["bad-shape", "bad-dtype", "nonfinite", "non-tensor"],
)
def test_forward_preserves_strict_model_output_validation(output):
    model = _ReturningModel(output)

    with pytest.raises(ValueError) as exc_info:
        _forward(_kernel("cpu"), model, _staged_batch(3))

    message = str(exc_info.value)
    if isinstance(output, list):
        assert "must return a tensor" in message
    elif output.dtype is not torch.float32:
        assert "finite FP32" in message
    elif not bool(torch.isfinite(output).all()):
        assert "finite FP32" in message
    else:
        assert "one normalized Gap per graph" in message


def test_cross_device_output_is_rejected_before_finite_check():
    """A meta output proves device rejection without requiring CUDA."""
    rows = 3
    output = torch.empty(rows, dtype=torch.float32, device="meta")

    with pytest.raises(
        ValueError,
        match=r"output=meta, input=cpu",
    ):
        _forward(_kernel("cpu"), _ReturningModel(output), _staged_batch(rows))


def test_stale_cuda_request_rejects_cpu_staging_before_model_call():
    """A CUDA descriptor is inert here; no CUDA device is queried or allocated."""
    model = _ReturningModel(torch.zeros(3, dtype=torch.float32))

    with pytest.raises(
        ValueError,
        match=r"input=cpu, requested=cuda:0",
    ):
        _forward(_kernel("cuda:0"), model, _staged_batch(3))

    assert model.calls == 0


@pytest.mark.parametrize(
    ("requested_device", "staged_device"),
    [("cuda:0", "cuda:1"), ("cuda:1", "cuda:0")],
)
def test_explicit_cuda_index_mismatch_rejects_before_model_call(
    requested_device, staged_device
):
    """Compare descriptors only; constructing x must not initialize a device."""
    batch = SimpleNamespace(
        x=SimpleNamespace(device=torch.device(staged_device)),
        num_graphs=1,
    )
    model = _ReturningModel(object())

    with pytest.raises(
        ValueError,
        match=rf"input={staged_device}, requested={requested_device}",
    ):
        _forward(_kernel(requested_device), model, batch)

    assert model.calls == 0

