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

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

from .esa_features import edge_tokens
from .esa_invariant import ESAInvariant, ESAInvariantConfig
from .esa_model import ESAConfig, ESABlock
from .modern2d_inputs import categorical_one_hot, validated_topology_views
from .ogb_features import ATOM_FEATURE_DIMS, BOND_FEATURE_DIMS


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

    def forward(self, batch):
        views = validated_topology_views(batch, max_nodes=self.config.max_nodes,
                                         max_edges=self.config.max_edges,
                                         allowed_roles=ALLOWED_ROLES)
        device = views[0].x.device
        if self.config.tensor_device != "cpu_or_cuda" and device.type != self.config.tensor_device:
            raise ValueError("ESA/3 topology device differs from configured tensor_device capability")
        if device != self.head.weight.device:
            raise ValueError("ESA/3 model and topology must share a device; move the model explicitly")
        return self._forward_tokens([native_tokens(graph) for graph in views])


def make_model(arm):
    values = deepcopy(arm.get("model_config", {}))
    if "layer_types" in values:
        values["layer_types"] = tuple(values["layer_types"])
    return ESANative(ESANativeConfig(**values))


CONFIG_TYPE = ESANativeConfig
