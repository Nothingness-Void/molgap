"""Model-only prepared topology context; no trainer/registry/platform imports."""
from contextvars import ContextVar
from dataclasses import dataclass

import torch

from .modern2d_inputs import GraphView, validated_topology_views


_ISSUER = object()
_PREPARED_CALL = ContextVar("molgap_modern2d_prepared_call", default=None)


def current_prepared(model, batch):
    """A call-scoped layout, never an attribute accepted from dataset input."""
    active = _PREPARED_CALL.get()
    return active[2] if active is not None and active[0] is model and active[1] is batch else None


def _bindings(cpu, native, max_nodes, max_edges):
    def view(graph):
        return (graph.source_idx, tuple((id(tensor), tensor._version, tensor.dtype,
            str(tensor.device), tuple(tensor.shape), tuple(tensor.stride()), tensor.storage_offset())
            for tensor in (graph.x, graph.edge_index, graph.edge_attr)))
    return (max_nodes, max_edges, tuple(map(view, cpu)), tuple(map(view, native)))


@dataclass(frozen=True)
class PreparedTopology:
    cpu_views: tuple
    device_views: tuple
    max_nodes: int
    max_edges: int
    _issuer: object
    _snapshot: tuple | None = None

    def validate_for(self, config, device):
        if (self._issuer is not _ISSUER or not self.device_views or
                self.max_nodes != config.max_nodes or self.max_edges != config.max_edges or
                any(g.x.device != device for g in self.device_views) or
                self._snapshot != _bindings(self.cpu_views, self.device_views,
                                            self.max_nodes, self.max_edges)):
            raise ValueError("Prepared topology mutated or differs from its bounded model/device")


@torch.inference_mode(False)
@torch.no_grad()
def prepare_topology(cpu_batch, *, device, max_nodes=1024, max_edges=4096):
    """Own version-tracked snapshots; invalid/protected input fails on CPU.

    CPU snapshots are independent of the caller's batch. Later edits or view
    replacement fail closed rather than bypassing the original validation.
    No labels/RWSE enter these transient layouts; no corpus cache is built.
    """
    names = ("x", "edge_index", "edge_attr", "batch", "ptr", "source_idx")
    if any(not torch.is_tensor(getattr(cpu_batch, name, None)) or
           getattr(cpu_batch, name).device.type != "cpu" for name in names):
        raise ValueError("Prepared execution requires CPU topology with explicit ptr/source")
    device = torch.device(device)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Prepared execution supports explicit CPU/CUDA devices only")
    views = validated_topology_views(cpu_batch, max_nodes=max_nodes,
        max_edges=max_edges, packed=True)
    atoms = torch.cat([g.x for g in views])
    edges = torch.cat([g.edge_index for g in views], 1)
    bonds = torch.cat([g.edge_attr for g in views])
    # Graph-local edges are packed on CPU; no GPU subtraction/sort per graph.
    staged = tuple(value.to(device, non_blocking=True) for value in (atoms, edges, bonds))
    cpu, native, node_start, edge_start = [], [], 0, 0
    for graph in views:
        n, e = len(graph.x), graph.edge_index.shape[1]
        cpu.append(GraphView(atoms[node_start:node_start+n], edges[:, edge_start:edge_start+e],
            bonds[edge_start:edge_start+e], graph.source_idx))
        native.append(GraphView(staged[0][node_start:node_start+n],
            staged[1][:, edge_start:edge_start+e], staged[2][edge_start:edge_start+e], graph.source_idx))
        node_start, edge_start = node_start+n, edge_start+e
    cpu, native = tuple(cpu), tuple(native)
    return PreparedTopology(cpu, native, max_nodes, max_edges, _ISSUER,
                            _bindings(cpu, native, max_nodes, max_edges))
