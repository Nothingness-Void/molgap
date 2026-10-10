"""Opt-in CPU-derived dense layout for K1; no model or checkpoint parameters."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass


@dataclass(frozen=True)
class DenseLayout:
    source_batch: object
    flat_index: object
    valid: object
    nodes: int
    graphs: int
    max_nodes: int

    def pack(self, hidden, membership):
        if (membership is not self.source_batch.batch or hidden.shape[0] != self.nodes
                or hidden.device != self.flat_index.device):
            raise ValueError("Dense layout and hidden tensor differ")
        dense = hidden.new_zeros((self.graphs * self.max_nodes, hidden.shape[-1]))
        dense[self.flat_index] = hidden
        return dense.view(self.graphs, self.max_nodes, hidden.shape[-1]), self.valid

    def unpack(self, dense):
        return dense.reshape(self.graphs * self.max_nodes, dense.shape[-1]).index_select(
            0, self.flat_index)


def layout_from_cpu_batch(batch, device):
    import torch

    ptr, membership = batch.ptr, batch.batch
    if ptr.device.type != "cpu" or membership.device.type != "cpu":
        raise ValueError("Layout must be prepared from the original CPU batch")
    if ptr.dtype != torch.long or membership.dtype != torch.long or ptr.ndim != 1:
        raise ValueError("Expected integral PyG batch boundaries")
    counts = ptr[1:] - ptr[:-1]
    if (ptr.numel() < 2 or int(ptr[0]) != 0 or int(ptr[-1]) != batch.x.shape[0]
            or not bool((counts > 0).all())):
        raise ValueError("Empty or malformed graph boundaries")
    graphs, nodes, maximum = counts.numel(), membership.numel(), int(counts.max())
    expected = torch.repeat_interleave(torch.arange(graphs), counts)
    if not torch.equal(membership, expected):
        raise ValueError("Unordered or inconsistent graph membership")
    flat = torch.arange(nodes) - ptr[:-1][membership] + membership * maximum
    valid = torch.zeros(graphs * maximum, dtype=torch.bool)
    valid[flat] = True
    return DenseLayout(batch, flat.to(device), valid.view(graphs, maximum).to(device),
                       nodes, graphs, maximum)


@contextmanager
def cpu_layout_context(model, cpu_batch, device):
    """Share one ephemeral layout for this batch; restore defaults on any exit."""
    if not hasattr(model, "neural_atom_mixers") or model.pooling != "mean":
        raise ValueError("CPU layout requires the existing mean-pooled K1 encoder")
    mixers = tuple(model.neural_atom_mixers.values())
    if not mixers or any(m.active_slots != 1 for m in mixers):
        raise ValueError("Only single-slot K1 is qualified by this adapter")
    if hasattr(model, "_execution_graph_count") or any(
            hasattr(m, "_execution_dense_layout") for m in mixers):
        raise ValueError("Nested or stale execution layout")
    layout = layout_from_cpu_batch(cpu_batch, device)
    try:
        model._execution_graph_count = layout.graphs
        for mixer in mixers:
            mixer._execution_dense_layout = layout
        yield
    finally:
        if hasattr(model, "_execution_graph_count"):
            del model._execution_graph_count
        for mixer in mixers:
            if hasattr(mixer, "_execution_dense_layout"):
                del mixer._execution_dense_layout
