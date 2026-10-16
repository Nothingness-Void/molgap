"""Pure CPU topology views for CPU-only Modern2D profiles; never labels/IO."""
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from .ogb_features import ATOM_FEATURE_DIMS, BOND_FEATURE_DIMS


@dataclass(frozen=True)
class GraphView:
    x: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    source_idx: int


def topology_views(batch):
    """Validate only topology/identity. y/RWSE may be present but are not read."""
    allowed = {"x", "edge_index", "edge_attr", "batch", "ptr", "source_idx",
               "y", "random_walk_pe", "num_nodes", "role_name"}
    if set(batch.keys()) - allowed:
        raise ValueError("Unknown/geometry field in Modern2D topology")
    if getattr(batch, "role_name", "CPU_SYNTHETIC_ONLY") != "CPU_SYNTHETIC_ONLY":
        raise ValueError("Prototype permits synthetic role only; protected/real roles blocked")
    names = ("x", "edge_index", "edge_attr", "batch", "source_idx")
    if any(not torch.is_tensor(getattr(batch, k, None)) for k in names):
        raise ValueError("Missing topology or source identity")
    if any(getattr(batch, k).device.type != "cpu" for k in names):
        raise ValueError("Modern2D prototype is CPU-only")
    x, edges, bonds, groups = batch.x, batch.edge_index, batch.edge_attr, batch.batch
    if any(getattr(batch, k).dtype != torch.int64 for k in names):
        raise ValueError("Topology/source tensors must be int64")
    if x.ndim != 2 or x.shape[1] != 9 or not len(x):
        raise ValueError("Expected nonempty atom9")
    if edges.ndim != 2 or edges.shape[0] != 2 or bonds.shape != (edges.shape[1], 3):
        raise ValueError("Expected aligned edge_index/bond3")
    if bool(((x < 0) | (x >= x.new_tensor(ATOM_FEATURE_DIMS))).any()) or bool(
            ((bonds < 0) | (bonds >= bonds.new_tensor(BOND_FEATURE_DIMS))).any()):
        raise ValueError("Categorical feature out of range")
    if groups.shape != (len(x),) or bool((groups < 0).any()):
        raise ValueError("Invalid graph membership")
    count = int(groups.max()) + 1
    if not torch.equal(torch.unique(groups), torch.arange(count)):
        raise ValueError("Graph IDs must be contiguous")
    source = batch.source_idx.reshape(-1)
    if source.numel() != count or bool((source < 0).any()) or source.unique().numel() != count:
        raise ValueError("Invalid source identity/order")
    if edges.numel() and (bool(((edges < 0) | (edges >= len(x))).any()) or
            bool((groups[edges[0]] != groups[edges[1]]).any())):
        raise ValueError("Invalid or cross-graph edge")
    views = []
    for i in range(count):
        nodes = (groups == i).nonzero().flatten()
        if nodes.numel() > 256:
            raise ValueError("CPU prototype node safety cap exceeded")
        remap = torch.full((len(x),), -1, dtype=torch.long)
        remap[nodes] = torch.arange(len(nodes))
        selected = groups[edges[0]] == i
        if int(selected.sum()) > 1024:
            raise ValueError("CPU prototype directed-edge safety cap exceeded")
        views.append(GraphView(x[nodes], remap[edges[:, selected]], bonds[selected], int(source[i])))
    return views


def categorical_one_hot(values, dimensions):
    return torch.cat([F.one_hot(values[:, i], d) for i, d in enumerate(dimensions)], -1).float()


