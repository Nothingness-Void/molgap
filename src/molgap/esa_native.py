"""ESA/3 device-local integration candidate; native qualification is pending.

This is an explicit MolGap adaptation, not the pinned author's whole model.
ESA/2's invariant PMA, pre-LN equations, gated MLPs and safe self attention are
retained. Each real directed bond retains both OGB categorical endpoints and
its bond features. Each otherwise unrepresented atom contributes one atom-only
token: its first atom slot is populated, its second atom/bond slots are zero
(absence, not category zero), and a learned token-type embedding distinguishes
it from a bond token. Incidence metadata uses (atom, -1), never a chemical edge.

M attention connects shared actual endpoints and self; isolated atom tokens
therefore attend only themselves in M. S attention and PMA see every real token
of their own graph. Padding queries have a safe self key and are zeroed after
each encoder block; padding keys are invisible to real queries and PMA. Features
and masks are derived per batch, with O(batch * max_tokens**2) mask storage and
max_tokens <= max_edges + max_nodes. No corpus feature cache is constructed.

Forward returns a normalized Gap vector; target statistics belong to the
existing training owner. CPU/CUDA support describes tensor placement only.
Real-input acceptance, optimizer-inclusive GPU preflight, precision/kernel
parity, memory/throughput and scientific training remain separate gates.
MIT-derived ESA components retain the attribution in esa_model.py and the
pinned reference's tests/vendor/esa_reference/LICENSE.

MIT License

Copyright (c) 2024 David Buterez

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from copy import deepcopy
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

from .esa_features import edge_tokens
from .esa_invariant import ESAInvariant, ESAInvariantConfig
from .esa_model import ESAConfig, ESABlock
from .modern2d_inputs import GraphView, categorical_one_hot, validated_topology_views
from .ogb_features import ATOM_FEATURE_DIMS, BOND_FEATURE_DIMS

if TYPE_CHECKING:
    from .modern2d_execution_context import PreparedTopology


BOND_TOKEN = 0
ATOM_ONLY_TOKEN = 1
ALLOWED_ROLES = ("CPU_SYNTHETIC_ONLY", "train", "development", "train_100k",
                 "train_500k", "common_dev")


@dataclass(frozen=True)
class ESANativeConfig(ESAConfig):
    readout_version: str = "pma-no-input-residual-v2"
    isolated_atom_policy: str = "explicit-atom-only-token-v3"
    tokenization_version: str = "directed-bond-and-isolated-atom-v3"
    max_nodes: int = 1024
    max_edges: int = 4096
    tensor_device: str = "cpu_or_cuda"

    def __post_init__(self):
        super().__post_init__()
        if (not self.include_self or
                self.readout_version != "pma-no-input-residual-v2" or
                self.isolated_atom_policy != "explicit-atom-only-token-v3" or
                self.tokenization_version != "directed-bond-and-isolated-atom-v3"):
            raise ValueError("ESA/3 requires explicit atom-only tokens, safe self mask and invariant PMA")
        if (type(self.max_nodes) is not int or not 1 <= self.max_nodes <= 1024 or
                type(self.max_edges) is not int or not 1 <= self.max_edges <= 4096):
            raise ValueError("ESA/3 topology caps must be positive and at most 1024 nodes/4096 directed edges")
        if self.tensor_device not in ("cpu", "cuda", "cpu_or_cuda"):
            raise ValueError("ESA/3 tensor_device must be cpu, cuda or cpu_or_cuda; this is not a native certificate")


@dataclass(frozen=True)
class NativeTokens:
    """Already validated graph tokens; endpoint -1 means an absent endpoint."""

    features: torch.Tensor
    token_types: torch.Tensor
    endpoints: torch.Tensor


def native_tokens(graph):
    """Retain every directed bond and every isolated atom of a validated view."""
    edges = graph.edge_index.shape[1]
    covered = torch.zeros(len(graph.x), dtype=torch.bool, device=graph.x.device)
    covered[graph.edge_index.reshape(-1)] = True
    isolated = (~covered).nonzero().flatten()
    atoms = categorical_one_hot(graph.x[isolated], ATOM_FEATURE_DIMS)
    atom_dim = sum(ATOM_FEATURE_DIMS)
    bond_dim = sum(BOND_FEATURE_DIMS)
    bonds = edge_tokens(graph) if edges else atoms.new_empty((0, 2 * atom_dim + bond_dim))
    atom_only = torch.cat((atoms, atoms.new_zeros((len(isolated), atom_dim + bond_dim))), -1)
    types = torch.cat((graph.x.new_zeros(edges), graph.x.new_ones(len(isolated))))
    incidence = torch.stack((isolated, torch.full_like(isolated, -1)))
    return NativeTokens(torch.cat((bonds, atom_only)), types,
                        torch.cat((graph.edge_index, incidence), 1))


class _PaddingAwareBlock(ESABlock):
    def __init__(self, block):
        # Keep the initialized ESA modules and state names. The sole difference
        # is that S must honor graph padding, which the unpadded v2 did not need.
        nn.Module.__init__(self)
        self.kind = block.kind
        self.norm = block.norm
        self.norm_mlp = block.norm_mlp
        self.attention = block.attention
        self.mlp = block.mlp

    def forward(self, x, allowed=None):
        if self.kind == "M":
            return super().forward(x, allowed)
        x = self.norm(x)
        out = x + self.attention(x, x, allowed)
        return out + self.mlp(self.norm_mlp(out))


def _attention_masks(endpoints, valid):
    """Vectorized graph-local incidence mask, including safe padding queries."""
    i, j = endpoints.unbind(-1)
    real_pair = valid[:, :, None] & valid[:, None, :]
    shared = ((i[:, :, None] == i[:, None, :]) |
              (i[:, :, None] == j[:, None, :]) |
              (j[:, :, None] == i[:, None, :]) |
              ((j[:, :, None] == j[:, None, :]) & (j[:, :, None] >= 0)))
    diagonal = torch.eye(valid.shape[1], dtype=torch.bool, device=valid.device)[None]
    # A padded query may see only itself. Valid queries never see padding keys.
    return ((shared & real_pair) | diagonal)[:, None], (real_pair | diagonal)[:, None]


class ESANative(ESAInvariant):
    def __init__(self, config=ESANativeConfig()):
        if type(config) is not ESANativeConfig:
            raise TypeError("Frozen ESANativeConfig required")
        base = {field.name: getattr(config, field.name) for field in fields(ESAConfig)}
        super().__init__(ESAInvariantConfig(**base))
        self.config = config
        self.encoder = nn.ModuleList(_PaddingAwareBlock(block) for block in self.encoder)
        self.token_type_embedding = nn.Embedding(2, config.hidden)
        nn.init.normal_(self.token_type_embedding.weight, std=config.hidden ** -0.5)

    def _forward_tokens(self, tokens):
        features = pad_sequence([t.features for t in tokens], batch_first=True)
        types = pad_sequence([t.token_types for t in tokens], batch_first=True)
        endpoints = pad_sequence([t.endpoints.t() for t in tokens], batch_first=True,
                                 padding_value=-1)
        lengths = types.new_tensor([len(t.token_types) for t in tokens])
        valid = torch.arange(types.shape[1], device=types.device)[None] < lengths[:, None]
        return self._forward_dense(features, types, endpoints, valid)

    def _forward_dense(self, features, types, endpoints, valid):
        padding = ~valid[:, :, None]
        # Embedding backward allocates a dense gradient for both rows, including
        # a finite zero row when a batch contains only one token kind. All MLP
        # parameters are shared, so neither kind leaves an unused module behind.
        x = (self.token_mlp(features) + self.token_type_embedding(types)).masked_fill(padding, 0)
        initial = x
        masked, global_mask = _attention_masks(endpoints, valid)
        for block in self.encoder:
            x = block(x, masked if block.kind == "M" else global_mask).masked_fill(padding, 0)
        x = x + initial
        pooled = self.pool(x, valid[:, None, None, :]).mean(1)
        return self.head(F.mish(self.graph_projection(pooled))).reshape(-1)

    def graph_forward(self, graph):
        """Single already validated GraphView; batch forward owns ABI validation."""
        if len(graph.x) > self.config.max_nodes or graph.edge_index.shape[1] > self.config.max_edges:
            raise ValueError("ESA/3 topology safety cap exceeded; no member may be dropped")
        return self._forward_tokens([native_tokens(graph)]).reshape(())

    def forward_reference(self, batch):
        views = validated_topology_views(batch, max_nodes=self.config.max_nodes,
                                         max_edges=self.config.max_edges,
                                         allowed_roles=ALLOWED_ROLES)
        device = views[0].x.device
        if self.config.tensor_device != "cpu_or_cuda" and device.type != self.config.tensor_device:
            raise ValueError("ESA/3 topology device differs from configured tensor_device capability")
        if device != self.head.weight.device:
            raise ValueError("ESA/3 model and topology must share a device; move the model explicitly")
        return self._forward_tokens([native_tokens(graph) for graph in views])

    def forward(self, batch):
        from .modern2d_execution_context import current_prepared
        prepared = current_prepared(self, batch)
        if prepared is not None:
            return self.forward_prepared(prepared)
        views = validated_topology_views(batch, max_nodes=self.config.max_nodes,
            max_edges=self.config.max_edges, allowed_roles=ALLOWED_ROLES, packed=True)
        device = views[0].x.device
        if self.config.tensor_device != "cpu_or_cuda" and device.type != self.config.tensor_device:
            raise ValueError("ESA/3 topology device differs from configured tensor_device capability")
        if device != self.head.weight.device:
            raise ValueError("ESA/3 model and topology must share a device; move the model explicitly")
        features, types, endpoints, valid = packed_native_tokens(views)
        return self._forward_dense(features, types, endpoints, valid)

    def forward_prepared(self, prepared: "PreparedTopology"):
        """Opt-in execution of a CPU-validated, bounded topology snapshot.

        Prepare invariant categorical tokens before transferring them. Learned
        projections, attention masks and every model operation still run through
        the historical dense path on the model device, without cached activations.
        Use ``modern2d_execution.prepared_forward`` with substrate addons so the
        model's ``__call__`` hook frame is retained. Outside that explicit context,
        ``forward`` follows its historical path. No native qualification follows.
        """
        from .modern2d_execution_context import PreparedTopology

        if type(prepared) is not PreparedTopology:
            raise TypeError("ESA prepared execution requires exact PreparedTopology")
        device = self.head.weight.device
        prepared.validate_for(self.config, torch.device("cpu"))
        if self.config.tensor_device != "cpu_or_cuda" and device.type != self.config.tensor_device:
            raise ValueError("ESA/3 topology device differs from configured tensor_device capability")
        return self._forward_dense(*prepare_native_tokens(prepared.cpu_views, device))


def packed_native_tokens(views):
    """Build the identical directed-bond/isolated-atom padded ABI once per batch."""
    device, b = views[0].x.device, len(views)
    counts = torch.tensor([len(g.x) for g in views], device=device)
    edge_counts = counts.new_tensor([g.edge_index.shape[1] for g in views])
    node_starts = torch.cat((counts.new_zeros(1), counts.cumsum(0)[:-1]))
    edge_starts = torch.cat((counts.new_zeros(1), edge_counts.cumsum(0)[:-1]))
    graph_ids = torch.arange(b, device=device)
    edge_graph = torch.repeat_interleave(graph_ids, edge_counts)
    local_edges = torch.cat([g.edge_index for g in views], 1)
    edges = local_edges + node_starts[edge_graph][None]
    atoms = torch.cat([g.x for g in views])
    attrs = torch.cat([g.edge_attr for g in views])
    covered = torch.zeros(len(atoms), dtype=torch.bool, device=device)
    covered[edges.reshape(-1)] = True
    isolated = (~covered).nonzero().flatten()
    node_graph = torch.repeat_interleave(graph_ids, counts)
    isolated_graph = node_graph[isolated]
    isolated_counts = torch.bincount(isolated_graph, minlength=b)
    isolated_starts = torch.cat((counts.new_zeros(1), isolated_counts.cumsum(0)[:-1]))
    lengths = edge_counts + isolated_counts
    width = int(lengths.max())
    one_hot = categorical_one_hot(atoms, ATOM_FEATURE_DIMS)
    bond_hot = categorical_one_hot(attrs, BOND_FEATURE_DIMS)
    features = one_hot.new_zeros((b, width, 2 * sum(ATOM_FEATURE_DIMS) + sum(BOND_FEATURE_DIMS)))
    types = counts.new_zeros((b, width))
    endpoints = counts.new_full((b, width, 2), -1)
    edge_positions = torch.arange(len(attrs), device=device) - edge_starts[edge_graph]
    features[edge_graph, edge_positions] = torch.cat((one_hot[edges[0]], one_hot[edges[1]], bond_hot), -1)
    endpoints[edge_graph, edge_positions] = local_edges.t()
    positions = torch.arange(len(isolated), device=device) - isolated_starts[isolated_graph] + edge_counts[isolated_graph]
    atom_only = torch.cat((one_hot[isolated], one_hot.new_zeros((len(isolated), sum(ATOM_FEATURE_DIMS) + sum(BOND_FEATURE_DIMS)))), -1)
    features[isolated_graph, positions] = atom_only
    types[isolated_graph, positions] = 1
    endpoints[isolated_graph, positions, 0] = isolated - node_starts[isolated_graph]
    valid = torch.arange(width, device=device)[None] < lengths[:, None]
    # Match pad_sequence's zero padding for the type channel; masks ignore it.
    return features, types, endpoints, valid


def prepare_native_tokens(cpu_views, device):
    """Pack validated CPU GraphViews, then transfer the four invariant tensors.

    Returns ``(features, types, endpoints, valid)`` with the exact padded ABI of
    ``packed_native_tokens``. Role, identity, category and model-cap validation
    belongs to the CPU topology owner before this low-level helper is called.
    There is no model/RNG access, dtype conversion or learned projection cache.
    """
    if (not isinstance(cpu_views, (tuple, list)) or not cpu_views or
            any(type(graph) is not GraphView for graph in cpu_views)):
        raise TypeError("ESA token preparation requires nonempty validated CPU GraphViews")
    if any(tensor.device.type != "cpu" for graph in cpu_views
           for tensor in (graph.x, graph.edge_index, graph.edge_attr)):
        raise ValueError("ESA token preparation requires CPU topology before device transfer")
    device = torch.device(device)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("ESA token preparation supports CPU/CUDA devices only")
    tokens = packed_native_tokens(cpu_views)
    return tuple(tensor.to(device, non_blocking=True) for tensor in tokens)


def make_model(arm):
    values = deepcopy(arm.get("model_config", {}))
    if "layer_types" in values:
        values["layer_types"] = tuple(values["layer_types"])
    return ESANative(ESANativeConfig(**values))


CONFIG_TYPE = ESANativeConfig
