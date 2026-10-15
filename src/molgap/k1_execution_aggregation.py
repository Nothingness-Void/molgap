"""Deterministic, topology-cached segment aggregation for K1 execution.

The plan is prepared from a CPU destination index once per batch.  It sorts
only the temporary aggregation view of a source tensor; callers keep their
original edge order for message construction and dropout.  The source rows
are restored in the custom backward with a cached inverse permutation.

The K1 execution contract uses FP32 source tensors.  Floating point dtypes are
accepted here as well so the primitive can be checked with double precision
and ``torch.autograd.gradcheck`` without creating a second implementation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Union

import torch


_DeviceLike = Union[str, torch.device]
_AggregationBackend = Literal["cached_scatter", "segmented"]


class _SegmentAggregationFunction(torch.autograd.Function):
    """Sum sorted source rows by cached segments and restore source gradients."""

    @staticmethod
    def forward(
        ctx,
        source: torch.Tensor,
        permutation: torch.Tensor,
        inverse_permutation: torch.Tensor,
        destinations: torch.Tensor,
        lengths: torch.Tensor,
        output_rows: int,
    ) -> torch.Tensor:
        edge_count = int(source.shape[0])
        channels = int(source.shape[1])
        ctx.edge_count = edge_count
        ctx.channels = channels
        ctx.save_for_backward(inverse_permutation, destinations, lengths)

        # segment_reduce does not accept an empty lengths tensor.  Keeping the
        # branch here also avoids constructing a device-side reduction plan for
        # molecules with no bonds while preserving the requested output shape.
        if edge_count == 0:
            return source.new_zeros((output_rows, channels))

        sorted_source = source.index_select(0, permutation)
        reduced = torch.segment_reduce(
            sorted_source,
            "sum",
            lengths=lengths,
            axis=0,
        )
        output = source.new_zeros((output_rows, channels))
        # destinations are unique, so this write has no duplicate-index race.
        output.index_copy_(0, destinations, reduced)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inverse_permutation, destinations, lengths = ctx.saved_tensors
        if ctx.edge_count == 0:
            grad_source = grad_output.new_empty((0, ctx.channels))
        else:
            active_grad = grad_output.index_select(0, destinations)
            # output_size is supplied so CUDA does not synchronize to discover
            # the result length from the device-side lengths tensor.
            sorted_grad = torch.repeat_interleave(
                active_grad,
                lengths,
                dim=0,
                output_size=ctx.edge_count,
            )
            grad_source = sorted_grad.index_select(0, inverse_permutation)
        return grad_source, None, None, None, None, None


class _CachedScatterAggregationFunction(torch.autograd.Function):
    """Cached flattened ``index_put_(accumulate=True)`` scatter provider."""

    @staticmethod
    def forward(
        ctx,
        source: torch.Tensor,
        flat_index: torch.Tensor,
        output_rows: int,
    ) -> torch.Tensor:
        edge_count = int(source.shape[0])
        channels = int(source.shape[1])
        ctx.edge_count = edge_count
        ctx.channels = channels
        ctx.save_for_backward(flat_index)
        output = source.new_zeros((output_rows, channels))
        if flat_index.numel():
            # This is the same flattened destination view used by the native
            # deterministic scatter path.  It leaves source/edge ordering
            # untouched and only reuses its coordinate tensor.
            output.reshape(-1).index_put_(
                (flat_index,),
                source.reshape(-1),
                accumulate=True,
            )
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (flat_index,) = ctx.saved_tensors
        if flat_index.numel():
            grad_source = grad_output.reshape(-1).index_select(0, flat_index)
            grad_source = grad_source.reshape(ctx.edge_count, ctx.channels)
        else:
            # ``flat_index`` is empty both for an empty edge set and for the
            # valid zero-channel shape [edges, 0].  Preserve the source row
            # count in the latter case so autograd receives a compatible
            # gradient shape.
            grad_source = grad_output.new_empty((ctx.edge_count, ctx.channels))
        return grad_source, None, None


@dataclass(frozen=True)
class SegmentAggregationPlan:
    """Reusable destination grouping for one batch topology.

    Preparation clones the CPU destination index and moves only that original
    index to the execution device.  The stable permutation, inverse mapping,
    unique destinations and segment lengths are built lazily on the first
    ``segmented`` call; the default ``cached_scatter`` provider never pays for
    them.  All plan tensors used by :meth:`reduce` are device resident, and no
    Python scalar is read from a device tensor in the hot path.
    """

    output_rows: int
    edge_count: int
    device: torch.device
    edge_destinations: torch.Tensor
    _index_cpu: torch.Tensor = field(repr=False, compare=False)
    _flat_index_cache: dict = field(default_factory=dict, init=False, repr=False, compare=False)
    _segment_cache: dict = field(default_factory=dict, init=False, repr=False, compare=False)

    @classmethod
    def from_cpu(
        cls,
        index: torch.Tensor,
        output_rows: int,
        device: _DeviceLike,
    ) -> "SegmentAggregationPlan":
        """Alias used by execution providers for CPU topology preparation."""
        return cls.prepare_from_cpu(index, output_rows, device)

    @classmethod
    def prepare_from_cpu(
        cls,
        index: torch.Tensor,
        output_rows: int,
        device: _DeviceLike,
    ) -> "SegmentAggregationPlan":
        """Prepare a stable destination grouping from a CPU index tensor.

        ``index`` is the destination row for each source/message row and must
        be one-dimensional.  It is deliberately required to be on CPU: graph
        topology preparation can happen once without forcing a CUDA scalar
        read in every aggregation call.
        """
        if not isinstance(index, torch.Tensor):
            raise TypeError("index must be a torch.Tensor")
        if index.device.type != "cpu":
            raise ValueError("index must be a CPU tensor during preparation")
        if index.ndim != 1:
            raise ValueError("index must have shape [edges]")
        if index.dtype not in {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        }:
            raise TypeError("index must use an integer dtype")
        if isinstance(output_rows, bool) or not isinstance(output_rows, int):
            raise TypeError("output_rows must be an int")
        if output_rows < 0:
            raise ValueError("output_rows must be non-negative")

        index_cpu = index.to(dtype=torch.long).clone()
        if index_cpu.numel():
            if bool((index_cpu < 0).any()):
                raise ValueError("index contains a negative destination")
            if bool((index_cpu >= output_rows).any()):
                raise ValueError("index contains a destination outside output_rows")

        target_device = torch.device(device)
        if target_device.type == "cuda" and target_device.index is None:
            target_device = torch.device("cuda", torch.cuda.current_device())
        return cls(
            output_rows=output_rows,
            edge_count=int(index_cpu.numel()),
            device=target_device,
            edge_destinations=index_cpu.to(target_device, non_blocking=True),
            _index_cpu=index_cpu,
        )

    def _segment_metadata(self):
        """Build the stable aggregation view only for the segmented provider."""
        metadata = self._segment_cache.get("metadata")
        if metadata is not None:
            return metadata
        permutation_cpu = torch.argsort(self._index_cpu, stable=True)
        sorted_index = self._index_cpu.index_select(0, permutation_cpu)
        if sorted_index.numel():
            destinations_cpu, lengths_cpu = torch.unique_consecutive(
                sorted_index,
                return_counts=True,
            )
        else:
            destinations_cpu = torch.empty(0, dtype=torch.long)
            lengths_cpu = torch.empty(0, dtype=torch.long)

        inverse_cpu = torch.empty_like(permutation_cpu)
        if permutation_cpu.numel():
            inverse_cpu[permutation_cpu] = torch.arange(
                permutation_cpu.numel(),
                dtype=torch.long,
            )
        metadata = (
            permutation_cpu.to(self.device),
            inverse_cpu.to(self.device),
            destinations_cpu.to(self.device),
            lengths_cpu.to(self.device),
        )
        self._segment_cache["metadata"] = metadata
        return metadata

    def _flat_index_for_channels(self, channels: int) -> torch.Tensor:
        """Return and cache flattened ``[edge, channel]`` destination offsets."""
        cached = self._flat_index_cache.get(channels)
        if cached is not None:
            return cached
        channel_offsets = torch.arange(channels, device=self.device, dtype=torch.long)
        flat_index = (
            self.edge_destinations.unsqueeze(1) * channels + channel_offsets
        ).reshape(-1)
        self._flat_index_cache[channels] = flat_index
        return flat_index

    def reduce(
        self,
        source: torch.Tensor,
        *,
        backend: _AggregationBackend = "cached_scatter",
    ) -> torch.Tensor:
        """Return ``[output_rows, channels]`` sums for the cached topology.

        ``cached_scatter`` is the compatibility provider and is the default.
        It reuses the flattened coordinate view used by the native
        deterministic scatter implementation.  ``segmented`` is an opt-in
        candidate that sorts only the aggregation view and uses
        ``torch.segment_reduce``.
        """
        if not isinstance(source, torch.Tensor):
            raise TypeError("source must be a torch.Tensor")
        if backend not in {"cached_scatter", "segmented"}:
            raise ValueError("backend must be 'cached_scatter' or 'segmented'")
        if source.ndim != 2:
            raise ValueError("source must have shape [edges, channels]")
        if source.shape[0] != self.edge_count:
            raise ValueError(
                f"source has {source.shape[0]} rows; expected {self.edge_count}"
            )
        if source.device != self.device:
            raise ValueError(
                f"source is on {source.device}; expected plan device {self.device}"
            )
        if not source.is_floating_point():
            raise TypeError("source must use a floating-point dtype")
        if backend == "cached_scatter":
            flat_index = self._flat_index_for_channels(int(source.shape[1]))
            return _CachedScatterAggregationFunction.apply(
                source,
                flat_index,
                self.output_rows,
            )
        permutation, inverse_permutation, destinations, lengths = self._segment_metadata()
        return _SegmentAggregationFunction.apply(
            source,
            permutation,
            inverse_permutation,
            destinations,
            lengths,
            self.output_rows,
        )


# Descriptive alias for callers that want to make the deterministic choice
# explicit in a recipe or addon declaration.
DeterministicAggregationPlan = SegmentAggregationPlan


AggregationPlan = SegmentAggregationPlan


__all__ = [
    "AggregationPlan",
    "DeterministicAggregationPlan",
    "SegmentAggregationPlan",
]
