"""Pure-2D exact-hop content with a bounded real diagonal recurrence.

This is a four-step stable recurrence, not a reproduction of paper GRED.
Mixed and ordered arms share every parameter and operation except whether
distance-partitioned content is replaced by its four-slot equal mixture.
"""
from contextlib import contextmanager
from dataclasses import dataclass, replace

import torch
from torch import nn

from .model_substrate import ModelSubstrate, SlotBinding, SlotName


@dataclass(frozen=True)
class HopShellBatch:
    source_batch: object
    shells: tuple
    union_count: object
    nodes: int


def shells_from_cpu_batch(batch, device):
    """Build exact shortest distances 1..4 from bonds before any H2D copy."""
    ptr, membership, edges = batch.ptr, batch.batch, batch.edge_index
    if any(value.device.type != "cpu" for value in (ptr, membership, edges, batch.x)):
        raise ValueError("Hop shells require the original CPU batch")
    if any(value.dtype != torch.long for value in (ptr, membership, edges)):
        raise ValueError("Hop shells require integral PyG topology")
    counts = ptr[1:] - ptr[:-1]
    nodes = batch.x.shape[0]
    if (ptr.ndim != 1 or ptr.numel() < 2 or int(ptr[0]) != 0
            or int(ptr[-1]) != nodes or not bool((counts > 0).all())
            or not torch.equal(membership, torch.repeat_interleave(torch.arange(counts.numel()), counts))
            or edges.ndim != 2 or edges.shape[0] != 2):
        raise ValueError("Malformed hop-shell graph boundaries")
    if edges.numel() and (int(edges.min()) < 0 or int(edges.max()) >= nodes
            or not torch.equal(membership[edges[0]], membership[edges[1]])):
        raise ValueError("Hop-shell bonds cross or exceed graph boundaries")
    graphs, maximum = counts.numel(), int(counts.max())
    adjacency = torch.zeros((graphs, maximum, maximum), dtype=torch.bool)
    graph = membership[edges[0]]
    source = edges[0] - ptr[graph]
    target = edges[1] - ptr[graph]
    adjacency[graph, source, target] = True
    adjacency[graph, target, source] = True
    diagonal = torch.arange(maximum)
    adjacency[:, diagonal, diagonal] = False
    seen = torch.eye(maximum, dtype=torch.bool).expand(graphs, -1, -1).clone()
    frontier = adjacency
    adjacency_float = adjacency.float()
    shells = []
    union_count = torch.zeros(nodes, dtype=torch.long)
    for hop in range(4):
        exact = frontier & ~seen
        seen |= exact
        graph, target, source = exact.nonzero(as_tuple=True)
        target, source = target + ptr[graph], source + ptr[graph]
        union_count.index_add_(0, target, torch.ones_like(target))
        shells.append((target.to(device, non_blocking=True), source.to(device, non_blocking=True)))
        if hop < 3:
            frontier = torch.bmm(exact.float(), adjacency_float) > 0
    return HopShellBatch(batch, tuple(shells), union_count.clamp_min(1).to(device, non_blocking=True), nodes)


class HopShellResidual(nn.Module):
    def __init__(self, *, ordered):
        super().__init__()
        self.ordered = ordered
        self.node_norm = nn.LayerNorm(192)
        self.project = nn.Linear(192, 64, bias=False)
        self.decay_logit = nn.Parameter(torch.zeros(64))
        self.output_norm = nn.LayerNorm(64)
        self.return_projection = nn.Linear(64, 192, bias=False)
        nn.init.zeros_(self.return_projection.weight)
        self._batch = None

    @contextmanager
    def cpu_batch_context(self, cpu_batch, device, shared):
        if self._batch is not None:
            raise ValueError("Nested or stale hop-shell batch context")
        if HopShellBatch not in shared:
            shared[HopShellBatch] = shells_from_cpu_batch(cpu_batch, device)
        self._batch = shared[HopShellBatch]
        try:
            yield
        finally:
            self._batch = None

    def forward(self, payload):
        topology = self._batch
        if topology is None:
            raise RuntimeError("Hop-shell forward requires its original CPU batch context")
        node = payload.node
        if node.ndim != 2 or node.shape != (topology.nodes, 192):
            raise ValueError("Hop-shell slot requires K1 node states [nodes, 192]")
        if node.device != topology.union_count.device:
            raise ValueError("Hop-shell context and node device differ")
        if (payload.edge_index is not topology.source_batch.edge_index
                or payload.graph_index is not topology.source_batch.batch):
            raise ValueError("Hop-shell forward differs from its owning batch")
        projected = self.project(self.node_norm(node))
        content = []
        for target, source in topology.shells:
            value = projected.new_zeros((topology.nodes, 64))
            value.index_add_(0, target, projected.index_select(0, source))
            content.append(value / topology.union_count.to(projected.dtype).unsqueeze(-1))
        content = torch.stack(content)
        if not self.ordered:
            mixed = content.sum(dim=0) / 4
            content = [mixed] * 4
        # The same diagonal contraction and four updates serve both arms.
        # The epsilon margin preserves strict (0, 1) decay in floating point.
        epsilon = torch.finfo(projected.dtype).eps
        decay = epsilon + (1 - 2 * epsilon) * self.decay_logit.sigmoid()
        state = torch.zeros_like(projected)
        for value in reversed(content):
            state = decay * state + (1 - decay) * value
        residual = self.return_projection(self.output_norm(state))
        return replace(payload, node=node + residual)


def apply_addon(model, config):
    """Attach the frozen three-layer intervention with identical arm RNG use."""
    expected = {"hops": 4, "width": 64, "layer_policy": "3-6-9", "seed": 42}
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("Hop-shell configuration differs from its frozen contract")
    if type(config.get("ordered")) is not int or config["ordered"] not in (0, 1):
        raise ValueError("Hop-shell ordered must be integer 0 or 1")
    owner = getattr(model, "_molgap_substrate", None)
    if owner is None:
        owner = ModelSubstrate(model, "neural_atom_k1")
    if owner.family != "neural_atom_k1":
        raise ValueError("Hop-shell addon requires the K1 substrate")
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(config["seed"])
        for layer in (3, 6, 9):
            owner.attach("hop_shell_" + str(layer), HopShellResidual(ordered=bool(config["ordered"])),
                         SlotBinding("neural_atom_k1", SlotName.PRE_MESSAGE, layer))
    return model
