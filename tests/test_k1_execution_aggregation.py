"""Synthetic checks for the cached K1 aggregation providers."""
from __future__ import annotations

import pytest
import torch

from molgap.k1_execution_aggregation import AggregationPlan, SegmentAggregationPlan


def _scatter_reference(source: torch.Tensor, index: torch.Tensor, output_rows: int):
    output = source.new_zeros((output_rows, source.shape[1]))
    return output.index_add(0, index.to(device=source.device), source)


@pytest.mark.parametrize("backend", ["cached_scatter", "segmented"])
@pytest.mark.parametrize("source_dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_unordered_duplicate_edges_match_scatter_and_gradient(
    backend, source_dtype, index_dtype
):
    index = torch.tensor([2, 0, 2, 1, 0, 2], dtype=index_dtype)
    plan = SegmentAggregationPlan.prepare_from_cpu(index, output_rows=4, device="cpu")
    source = torch.tensor(
        [[1.0, -2.0], [3.0, 4.0], [-5.0, 6.0], [7.0, 8.0], [9.0, -1.0], [2.0, 3.0]],
        dtype=source_dtype,
        requires_grad=True,
    )
    weight = torch.tensor([[1.0, 2.0], [3.0, -2.0], [0.5, 4.0], [2.0, 1.0]], dtype=source_dtype)

    actual = plan.reduce(source, backend=backend)
    expected = _scatter_reference(source, index, 4)
    torch.testing.assert_close(actual, expected)
    native_scatter = source.new_zeros((4, source.shape[1]))
    native_scatter.scatter_add_(
        0,
        index.to(torch.long)[:, None].expand(-1, source.shape[1]),
        source,
    )
    torch.testing.assert_close(actual, native_scatter)
    (actual * weight).sum().backward()
    actual_grad = source.grad.detach().clone()

    source_ref = source.detach().clone().requires_grad_()
    (_scatter_reference(source_ref, index, 4) * weight).sum().backward()
    torch.testing.assert_close(actual_grad, source_ref.grad)


@pytest.mark.parametrize("backend", ["cached_scatter", "segmented"])
def test_isolated_nodes_and_empty_edges_preserve_shape_and_gradient(backend):
    index = torch.empty(0, dtype=torch.long)
    plan = SegmentAggregationPlan.prepare_from_cpu(index, output_rows=5, device="cpu")
    source = torch.empty((0, 3), dtype=torch.float32, requires_grad=True)
    actual = plan.reduce(source, backend=backend)
    torch.testing.assert_close(actual, torch.zeros((5, 3)))
    actual.sum().backward()
    assert source.grad is not None
    assert source.grad.shape == source.shape


@pytest.mark.parametrize("backend", ["cached_scatter", "segmented"])
def test_zero_channel_source_keeps_edge_rows_in_backward(backend):
    plan = SegmentAggregationPlan.from_cpu(
        torch.tensor([1, 0], dtype=torch.long), output_rows=2, device="cpu"
    )
    source = torch.empty((2, 0), dtype=torch.float32, requires_grad=True)
    result = plan.reduce(source, backend=backend)
    assert result.shape == (2, 0)
    result.sum().backward()
    assert source.grad is not None
    assert source.grad.shape == source.shape


def test_cached_flattened_index_is_reused_by_width():
    plan = SegmentAggregationPlan.prepare_from_cpu(
        torch.tensor([2, 0, 2, 1], dtype=torch.long),
        output_rows=3,
        device="cpu",
    )
    source = torch.randn(4, 7, requires_grad=True)
    plan.reduce(source)
    first = plan._flat_index_cache[7]
    plan.reduce(source)
    assert plan._flat_index_cache[7] is first
    assert first.shape == (4 * 7,)
    assert first.dtype == torch.long


def test_nine_local_aggregation_calls_build_one_flat_index_per_width():
    plan = AggregationPlan.from_cpu(
        torch.tensor([2, 0, 2, 1], dtype=torch.long),
        output_rows=3,
        device="cpu",
    )
    for _ in range(9):
        plan.reduce(torch.randn(4, 7))
    assert len(plan._flat_index_cache) == 1
    assert plan._flat_index_cache[7].shape == (4 * 7,)


def test_cached_provider_does_not_eagerly_build_segment_metadata():
    plan = AggregationPlan.from_cpu(
        torch.tensor([2, 0, 2, 1], dtype=torch.long),
        output_rows=3,
        device="cpu",
    )
    plan.reduce(torch.randn(4, 7))
    assert "metadata" not in plan._segment_cache
    plan.reduce(torch.randn(4, 7), backend="segmented")
    assert "metadata" in plan._segment_cache


def test_gradcheck_covers_cached_and_segmented_backward():
    index = torch.tensor([3, 1, 3, 0, 1], dtype=torch.long)
    plan = SegmentAggregationPlan.prepare_from_cpu(index, output_rows=4, device="cpu")
    source = torch.randn(5, 3, dtype=torch.double, requires_grad=True)
    for backend in ("cached_scatter", "segmented"):
        assert torch.autograd.gradcheck(
            lambda value: plan.reduce(value, backend=backend),
            (source,),
            eps=1e-6,
            atol=1e-5,
            rtol=1e-4,
        )


def test_plan_rejects_invalid_topology_and_source():
    with pytest.raises(ValueError, match="CPU"):
        SegmentAggregationPlan.prepare_from_cpu(
            torch.tensor([0], dtype=torch.long, device="meta"),
            output_rows=1,
            device="cpu",
        )
    with pytest.raises(ValueError, match="outside"):
        SegmentAggregationPlan.prepare_from_cpu(
            torch.tensor([0, 2], dtype=torch.long), output_rows=2, device="cpu"
        )
    plan = SegmentAggregationPlan.prepare_from_cpu(
        torch.tensor([0, 1], dtype=torch.long), output_rows=2, device="cpu"
    )
    source_index = torch.tensor([0, 1], dtype=torch.long)
    cloned_plan = AggregationPlan.from_cpu(source_index, output_rows=2, device="cpu")
    source_index[0] = 1
    torch.testing.assert_close(
        cloned_plan.reduce(torch.tensor([[1.0], [2.0]])),
        torch.tensor([[1.0], [2.0]]),
    )
    with pytest.raises(ValueError, match="shape"):
        plan.reduce(torch.randn(2))
    with pytest.raises(ValueError, match="rows"):
        plan.reduce(torch.randn(1, 3))
    with pytest.raises(TypeError, match="floating"):
        plan.reduce(torch.ones(2, 3, dtype=torch.int64))
    with pytest.raises(ValueError, match="device"):
        plan.reduce(torch.ones(2, 3, device="meta"))
    with pytest.raises(ValueError, match="backend"):
        plan.reduce(torch.ones(2, 3), backend="unknown")
