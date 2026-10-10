"""Scoped, opt-in algebraic projection reorders; no checkpoint state changes."""
from __future__ import annotations

from contextlib import contextmanager

import torch
from torch import nn

from .gps import _PersistentEdgeUpdate
from .k1_execution_layout import cpu_layout_context, layout_from_cpu_batch


_GATHER = "_execution_project_before_gather"
_PACK = "_execution_project_before_pack"


def _fresh(model, modules):
    if hasattr(model, "_execution_graph_count") or any(
        hasattr(module, name)
        for module in (*model.modules(), *modules)
        for name in (_GATHER, _PACK, "_execution_dense_layout")
    ):
        raise ValueError("Nested or stale execution projection/layout")


def _linear(module, inputs=None):
    return (type(module) is nn.Linear and module.bias is None
            and (inputs is None or module.in_features == inputs))


def _batch_guard(batch, names=("ptr", "batch")):
    # Tensor versions detect in-place topology changes, even if later reverted.
    tensors = tuple(getattr(batch, name) for name in names)
    versions = tuple(t._version for t in tensors)

    def check():
        if any(t._version != version for t, version in zip(tensors, versions)):
            raise ValueError("Stale execution batch topology")
        for name, tensor in zip(names, tensors):
            current = getattr(batch, name)
            if current.device == tensor.device and current is not tensor:
                raise ValueError("Stale execution batch tensor identity")

    return check


@contextmanager
def project_before_gather_context(model, cpu_batch, device):
    """Project real nodes once, then gather directed endpoints for this batch."""
    updates = tuple(getattr(model, "edge_updates", ()))
    _fresh(model, updates)
    if not updates or any(
        type(update) is not _PersistentEdgeUpdate
        or not _linear(update.source) or not _linear(update.target)
        or update.source.weight.shape != update.target.weight.shape
        for update in updates
    ):
        raise ValueError("Unsupported persistent edge-update class or projections")
    layout = layout_from_cpu_batch(cpu_batch, "cpu")
    expected_device = torch.device(device)
    if expected_device.type == "cuda" and expected_device.index is None:
        expected_device = torch.device("cuda", torch.cuda.current_device())
    edges = cpu_batch.edge_index
    if (edges.device.type != "cpu" or edges.dtype != torch.long
            or edges.ndim != 2 or edges.shape[0] != 2
            or (edges.numel() and (int(edges.min()) < 0 or int(edges.max()) >= layout.nodes))):
        raise ValueError("Unsupported edge-index shape or endpoints")
    check = _batch_guard(cpu_batch, ("ptr", "batch", "edge_index"))

    def guard(module, args, kwargs):
        check()
        names = ("h", "edge_index", "edge_state")
        if len(args) > 3 or any(name not in names for name in kwargs):
            raise ValueError("Unsupported edge-update arguments")
        inputs = dict(zip(names, args))
        if any(name in inputs for name in kwargs):
            raise ValueError("Duplicate edge-update arguments")
        inputs.update(kwargs)
        if set(inputs) != set(names):
            raise ValueError("Missing edge-update arguments")
        h, edge_index, state = (inputs[name] for name in names)
        if (h.ndim != 2 or h.shape != (layout.nodes, module.source.in_features)
                or h.device != expected_device
                or edge_index is not cpu_batch.edge_index
                or edge_index.device != h.device
                or edge_index.shape != edges.shape
                or state.shape != (edges.shape[1], module.source.out_features)
                or state.device != h.device):
            raise ValueError("Stale batch or unsupported edge-update input shape")

    handles = []
    activated = []
    try:
        for update in updates:
            handles.append(update.register_forward_pre_hook(guard, with_kwargs=True))
            setattr(update, _GATHER, True)
            activated.append(update)
        yield
    finally:
        for handle in handles:
            handle.remove()
        for update in activated:
            delattr(update, _GATHER)


class _GuardedLayout:
    def __init__(self, layout, check):
        self.layout = layout
        self.check = check

    def pack(self, hidden, membership):
        self.check()
        if hidden.ndim != 2:
            raise ValueError("Unsupported mixer projection shape")
        return self.layout.pack(hidden, membership)

    def unpack(self, dense):
        self.check()
        return self.layout.unpack(dense)


@contextmanager
def project_before_pack_context(model, cpu_batch, device):
    """Reuse CPU layout and project normalized real nodes in single-slot mixers."""
    mixers = tuple(getattr(model, "neural_atom_mixers", {}).values())
    _fresh(model, mixers)
    owner = "_NeuralAtomMixerFactory.make.<locals>.NeuralAtomMixer"
    if not mixers or any(
        type(m).__module__ != "molgap.qm9_neural_atom"
        or type(m).__qualname__ != owner or m.active_slots != 1
        or not _linear(m.node_key) or not _linear(m.node_value, m.node_key.in_features)
        or m.node_key.weight.shape != m.node_value.weight.shape
        or m.node_key.out_features != m.latent_channels
        or type(m.node_norm) is not nn.LayerNorm
        or m.node_norm.normalized_shape != (m.node_key.in_features,)
        for m in mixers
    ):
        raise ValueError("Unsupported single-slot mixer class or projections")
    check = _batch_guard(cpu_batch)
    activated = []
    with cpu_layout_context(model, cpu_batch, device):
        try:
            for mixer in mixers:
                mixer._execution_dense_layout = _GuardedLayout(
                    mixer._execution_dense_layout, check)
                setattr(mixer, _PACK, True)
                activated.append(mixer)
            yield
        finally:
            for mixer in activated:
                delattr(mixer, _PACK)
