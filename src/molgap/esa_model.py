"""ESA pre-LN CPU model with explicit safe mask adaptation and fidelity scope.

Attention/MLP/residual equations follow the MIT author implementation at
861a0b12d445ae80cc69b3948af623b4331321d4 (David Buterez,2024).
Whole-model identity differs at self/padding masks; see the integration record.
"""
from copy import deepcopy
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .esa_features import edge_tokens, shared_endpoint_mask
from .modern2d_inputs import topology_views
from .ogb_features import ATOM_FEATURE_DIMS, BOND_FEATURE_DIMS


@dataclass(frozen=True)
class ESAConfig:
    hidden: int = 192
    heads: int = 8
    layer_types: tuple = ("M", "S", "M", "S", "M", "P")
    seeds: int = 32
    graph_dim: int = 256
    mlp_hidden: int = 384
    include_self: bool = True
    epsilon: float = 1e-8

    def __post_init__(self):
        if any(type(v) is not int or v < 1 for v in (self.hidden, self.heads, self.seeds, self.graph_dim, self.mlp_hidden)):
            raise ValueError("Positive ESA dimensions required")
        if self.hidden % self.heads or type(self.include_self) is not bool or self.epsilon != 1e-8:
            raise ValueError("Invalid ESA heads/mask/LN epsilon")
        if type(self.layer_types) is not tuple or not self.layer_types or self.layer_types[-1] != "P" or \
                self.layer_types.count("P") != 1 or any(k not in {"M", "S", "P"} for k in self.layer_types):
            raise ValueError("CPU ESA supports encoder M/S then one terminal P")


class ESAAttention(nn.Module):
    """Author MAB, including out + mish(fc_o(out)), not a generic MHA swap."""
    def __init__(self, hidden, heads):
        super().__init__()
        self.hidden, self.heads = hidden, heads
        self.fc_q = nn.Linear(hidden, hidden)
        self.fc_k = nn.Linear(hidden, hidden)
        self.fc_v = nn.Linear(hidden, hidden)
        self.fc_o = nn.Linear(hidden, hidden)
        for layer in (self.fc_q, self.fc_k, self.fc_v, self.fc_o):
            nn.init.xavier_normal_(layer.weight)

    def forward(self, q, kv, allowed=None):
        def split(x):
            return x.reshape(x.shape[0], -1, self.heads, self.hidden // self.heads).transpose(1, 2)
        query, key, value = split(self.fc_q(q)), split(self.fc_k(kv)), split(self.fc_v(kv))
        mask = allowed
        if mask is not None and mask.ndim == 2:
            mask = mask[None, None]
        out = F.scaled_dot_product_attention(query, key, value, attn_mask=mask)
        out = out.transpose(1, 2).reshape(q.shape[0], -1, self.hidden)
        return out + F.mish(self.fc_o(out))


class GatedMLPSingle(nn.Module):
    def __init__(self, in_dim, inter_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 2 * inter_dim)
        self.fc2 = nn.Linear(inter_dim, out_dim)

    def forward(self, x):
        y, gate = self.fc1(x).chunk(2, -1)
        return self.fc2(F.silu(gate) * y)


def gated_mlp(hidden, inter):
    # Author GatedMLPMulti(num_layers=2), including the final Mish.
    return nn.Sequential(GatedMLPSingle(hidden, inter, inter), nn.Mish(),
                         GatedMLPSingle(inter, inter, hidden), nn.Mish())


class ESABlock(nn.Module):
    def __init__(self, config, kind):
        super().__init__()
        self.kind = kind
        self.norm = nn.LayerNorm(config.hidden, eps=config.epsilon)
        self.norm_mlp = nn.LayerNorm(config.hidden, eps=config.epsilon)
        self.attention = ESAAttention(config.hidden, config.heads)
        self.mlp = gated_mlp(config.hidden, config.mlp_hidden)
        if kind == "P":
            self.seeds = nn.Parameter(torch.empty(1, config.seeds, config.hidden))
            nn.init.xavier_normal_(self.seeds)

    def forward(self, x, allowed=None):
        # This preserves the author's actual pre branch, whose residual input
        # has already passed LN; it is not silently changed to canonical pre-LN.
        x = self.norm(x)
        if self.kind == "P":
            out = self.attention(self.seeds.expand(x.shape[0], -1, -1), x, allowed)
            if out.shape[-2] == x.shape[-2]:
                out = out + x
        else:
            out = x + self.attention(x, x, allowed if self.kind == "M" else None)
        return out + self.mlp(self.norm_mlp(out))


class ESA(nn.Module):
    def __init__(self, config=ESAConfig()):
        super().__init__()
        if type(config) is not ESAConfig:
            raise TypeError("Frozen ESAConfig required")
        self.config = config
        token_dim = 2 * sum(ATOM_FEATURE_DIMS) + sum(BOND_FEATURE_DIMS)
        self.token_mlp = nn.Sequential(nn.Linear(token_dim, 128), nn.Mish(), nn.Linear(128, config.hidden))
        self.encoder = nn.ModuleList(ESABlock(config, kind) for kind in config.layer_types[:-1])
        self.pool = ESABlock(config, "P")
        self.graph_projection = nn.Linear(config.hidden, config.graph_dim)
        self.head = nn.Linear(config.graph_dim, 1)

    def graph_forward(self, graph):
        x = self.token_mlp(edge_tokens(graph))[None]
        initial = x
        allowed = shared_endpoint_mask(graph.edge_index, include_self=self.config.include_self)
        if not self.config.include_self:
            # Scoped author float-mask reference. Its finite sentinel is
            # deliberately distinguished from the safe candidate bool mask.
            allowed = torch.where(allowed, 0.0, -99999.0)
        for layer in self.encoder:
            x = layer(x, allowed)
        x = x + initial
        pooled = self.pool(x).mean(1)
        return self.head(F.mish(self.graph_projection(pooled))).reshape(())

    def forward(self, batch):
        return torch.stack([self.graph_forward(g) for g in topology_views(batch)])


def make_model(arm):
    values = deepcopy(arm.get("model_config", {}))
    if "layer_types" in values:
        values["layer_types"] = tuple(values["layer_types"])
    return ESA(ESAConfig(**values))


CONFIG_TYPE = ESAConfig
