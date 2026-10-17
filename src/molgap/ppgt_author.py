"""PPGT/2: scoped port of LiamMa/PPGT 62cafd3 (MIT), not the v1 prototype.

The pinned PCQM branch is pre-norm AdaRMSNorm/GELU with unscaled QK, a
sqrt(head_dim)-scaled key penalty, a linear +1 URPE gate, LayerScale, RRWP
and an edge FFN stem. Raw source/config/license and adaptation boundaries
are in tests/vendor/ppgt_reference. A published *base* configuration is
available via ``PPGTAuthorConfig.author_pcqm_base()``. Defaults retain the
MolGap budget structure; they are not an author small benchmark recipe.

MolGap adaptations: bounded graph-wise execution without batch mutation or
persistent PE caches; the existing atom9/bond3 contract; normalized scalar
Gap output (normalization belongs to the trainer); approved topology roles.
No trainer, optimizer, loss, scheduler, data access or native release lives
here. Fixed-weight oracle tests must pass before any scoped parity claim.

MIT License

Copyright (c) 2026 Liheng Ma

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
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from .modern2d_inputs import validated_topology_views
from .ogb_features import ATOM_FEATURE_DIMS, BOND_FEATURE_DIMS
from .ppgt_author_features import positional_inputs, sinusoidal_pe


UPSTREAM_COMMIT = "62cafd3abbb8d72e9a8b3146f947a192cf0aacaf"
MODEL_VERSION = "2"
FEATURE_VERSION = "ppgt-author-rrwp-spe-v2"
UPSTREAM_LICENSE = """MIT License

Copyright (c) 2026 Liheng Ma

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
ALLOWED_ROLES = ("CPU_SYNTHETIC_ONLY", "train", "development", "train_100k",
                 "train_500k", "common_dev")


@dataclass(frozen=True)
class PPGTAuthorConfig:
    hidden: int = 256
    blocks: int = 16
    heads: int = 8
    pair_dim: int = 128
    pe_mlp_hidden: int = 512
    stem_blocks: int = 8
    expansion: int = 2
    stem_expansion: int = 2
    rrwp_steps: int = 24
    spe_bases: int = 10
    use_phi: bool = True
    sym_norm: bool = True
    pooling: str = "sum"
    epsilon: float = 1e-6
    layer_scale: float = 1e-4
    attention_dropout: float = 0.0
    stochastic_depth: float = 0.0
    max_nodes: int = 256
    max_edges: int = 1024

    def __post_init__(self):
        dimensions = ("hidden", "blocks", "heads", "pair_dim", "pe_mlp_hidden",
                      "stem_blocks", "expansion", "stem_expansion", "rrwp_steps",
                      "max_nodes", "max_edges")
        if any(type(getattr(self, name)) is not int or getattr(self, name) < 1 for name in dimensions):
            raise ValueError("Positive integer PPGT author dimensions/bounds required")
        if self.hidden < 8 or self.hidden % self.heads or type(self.spe_bases) is not int or self.spe_bases < 0:
            raise ValueError("Invalid PPGT author head dimensions/SPE")
        if self.max_nodes > 1024 or self.max_edges > 4096:
            raise ValueError("PPGT author bounds exceed reviewed ceiling 1024 nodes/4096 edges")
        if type(self.use_phi) is not bool or type(self.sym_norm) is not bool or self.pooling not in {"sum", "mean"}:
            raise ValueError("Explicit PPGT author gate/normalization/readout required")
        if not self.use_phi:
            raise ValueError("Pinned PCQM author branch requires the linear +1 URPE gate")
        if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0
               for v in (self.epsilon, self.layer_scale)):
            raise ValueError("Positive finite norm epsilon and LayerScale required")
        if any(type(v) not in (int, float) or not 0 <= v < 1
               for v in (self.attention_dropout, self.stochastic_depth)):
            raise ValueError("Invalid PPGT dropout/stochastic-depth probability")

    @classmethod
    def author_pcqm_base(cls, **bounds):
        """Architecture/dropout fields from the sole pinned PCQM YAML.

        Bounds are MolGap rejection limits, never a policy to drop members.
        Training batch/LR/optimizer fields are intentionally not represented.
        """
        if set(bounds) - {"max_nodes", "max_edges"}:
            raise ValueError("Only memory bounds may override the author base preset")
        return cls(hidden=768, blocks=12, heads=12, pair_dim=128,
                   pe_mlp_hidden=512, stem_blocks=2, expansion=4,
                   stem_expansion=4, rrwp_steps=8, spe_bases=5, pooling="mean",
                   attention_dropout=0.1, stochastic_depth=0.2, **bounds)


class AuthorAdaRMSNorm(nn.Module):
    """Released F.rms_norm branch and epsilon; no v1 overflow reformulation."""
    def __init__(self, dim, epsilon=1e-6):
        super().__init__()
        self.beta = nn.Parameter(torch.ones(dim))
        self.alpha = nn.Parameter(torch.zeros(dim))
        self.epsilon = epsilon

    def forward(self, x):
        rescale = ((x * self.alpha + self.beta).square().mean(-1, keepdim=True) + self.epsilon).sqrt()
        return F.rms_norm(x, (x.size(-1),), weight=None, eps=self.epsilon) * rescale


class AuthorAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads, self.head_dim = config.heads, config.hidden // config.heads
        self.use_phi = config.use_phi
        self.Q = nn.Linear(config.hidden, config.hidden)
        self.KV = nn.Linear(config.hidden, 2 * config.hidden)
        self.O = nn.Linear(config.hidden, config.hidden)
        self.P = nn.Linear(config.pair_dim, config.heads * (2 if config.use_phi else 1))
        self.attn_dropout = nn.Dropout(config.attention_dropout)

    def forward(self, x, pair):
        n, h, d = len(x), self.heads, self.head_dim
        q = self.Q(x).view(n, h, d).transpose(0, 1)
        k, v = self.KV(x).view(n, 2, h, d).permute(1, 2, 0, 3).unbind(0)
        positional = self.P(pair).permute(2, 0, 1)
        # The pinned source scales only K_norm, not QK or the positional bias.
        logits = q @ k.transpose(-1, -2)
        logits = logits + (positional[:h] - 0.5 * k.square().sum(-1).unsqueeze(-2) / math.sqrt(d))
        weights = logits.softmax(-1)
        if self.use_phi:
            weights = weights * (positional[h:] + 1)
        values = self.attn_dropout(weights) @ v
        return self.O(values.transpose(0, 1).reshape(n, h * d))


def _graph_drop(value, probability, training):
    if probability == 0 or not training:
        return value
    # One sample per graph/branch; upstream uses F.dropout on graph-wise ones.
    return value * F.dropout(value.new_ones(1, dtype=torch.float32),
                             p=probability, training=True)


class AuthorBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1_h = AuthorAdaRMSNorm(config.hidden, config.epsilon)
        self.attention = AuthorAttention(config)
        self.norm2_h = AuthorAdaRMSNorm(config.hidden, config.epsilon)
        self.FFN = nn.Sequential(nn.Linear(config.hidden, config.hidden * config.expansion),
                                 nn.Identity(), nn.GELU(), nn.Identity(),
                                 nn.Linear(config.hidden * config.expansion, config.hidden))
        self.attention_scale = nn.Parameter(torch.full((config.hidden,), config.layer_scale))
        self.ffn_scale = nn.Parameter(torch.full((config.hidden,), config.layer_scale))
        # YAML uses uniform drop-path, not an invented depth schedule.
        self.drop_path = config.stochastic_depth

    def forward(self, x, pair):
        x = x + self.attention_scale * _graph_drop(self.attention(self.norm1_h(x), pair), self.drop_path, self.training)
        return x + self.ffn_scale * _graph_drop(self.FFN(self.norm2_h(x)), self.drop_path, self.training)


class AuthorStemBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm = AuthorAdaRMSNorm(config.pair_dim, config.epsilon)
        self.mlp = nn.Sequential(nn.Linear(config.pair_dim, config.pair_dim * config.stem_expansion),
                                 nn.GELU(), nn.Dropout(0.0),
                                 nn.Linear(config.pair_dim * config.stem_expansion, config.pair_dim))

    def forward(self, pair):
        return pair + self.mlp(self.norm(pair))


def _truncated_linear(module):
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        nn.init.zeros_(module.bias)


class PPGTAuthor(nn.Module):
    """Bounded graph-wise CPU/CUDA port; accelerator support is unqualified."""
    def __init__(self, config=PPGTAuthorConfig()):
        super().__init__()
        if type(config) is not PPGTAuthorConfig:
            raise TypeError("Frozen PPGTAuthorConfig required")
        self.config = config
        c = config
        self.atoms = nn.ModuleList(nn.Embedding(d, c.hidden) for d in ATOM_FEATURE_DIMS)
        self.bonds = nn.ModuleList(nn.Embedding(d, c.pair_dim) for d in BOND_FEATURE_DIMS)
        self.node_pe = nn.Linear(c.rrwp_steps + 1, c.hidden)
        self.node_order = nn.Linear(1, c.hidden)
        self.pe_mlp = nn.Sequential(nn.Linear((c.rrwp_steps + 2) * (1 + 2 * c.spe_bases), c.pe_mlp_hidden),
                                    nn.GELU(), nn.Linear(c.pe_mlp_hidden, c.pair_dim))
        self.stem_pre_norm = AuthorAdaRMSNorm(c.pair_dim, c.epsilon)
        self.stem = nn.ModuleList(AuthorStemBlock(c) for _ in range(c.stem_blocks))
        self.stem_post_norm = AuthorAdaRMSNorm(c.pair_dim, c.epsilon)
        self.pre_backbone_norm = AuthorAdaRMSNorm(c.hidden, c.epsilon)
        self.blocks = nn.ModuleList(AuthorBlock(c) for _ in range(c.blocks))
        self.post_backbone_norm = AuthorAdaRMSNorm(c.hidden, c.epsilon)
        self.head = nn.Sequential(nn.Linear(c.hidden, c.hidden), nn.GELU(), nn.Dropout(0.0),
                                  nn.Linear(c.hidden, c.hidden), nn.GELU(), nn.Dropout(0.0),
                                  nn.Linear(c.hidden, 1))
        self.reset_parameters()

    def reset_parameters(self):
        # GraphNet.apply visits descendants before each encoder's init_weights:
        # trunk/stem/head are truncated; positional encoder overrides are Kaiming.
        self.apply(_truncated_linear)
        for embedding in (*self.atoms, *self.bonds):
            nn.init.xavier_uniform_(embedding.weight)
        for module in (self.node_pe, self.node_order, self.pe_mlp[2]):
            nn.init.kaiming_uniform_(module.weight, a=1)
            nn.init.zeros_(module.bias)
        nn.init.kaiming_uniform_(self.pe_mlp[0].weight, a=0)
        nn.init.zeros_(self.pe_mlp[0].bias)
        # Explicit reset also restores learned affine and residual parameters.
        for module in self.modules():
            if isinstance(module, AuthorAdaRMSNorm):
                nn.init.zeros_(module.alpha)
                nn.init.ones_(module.beta)
            elif isinstance(module, AuthorBlock):
                nn.init.constant_(module.attention_scale, self.config.layer_scale)
                nn.init.constant_(module.ffn_scale, self.config.layer_scale)

    def graph_forward(self, graph):
        c = self.config
        if graph.x.device != self.node_pe.weight.device:
            raise ValueError("Model and topology must occupy the same device")
        walk, positional, order = positional_inputs(graph, c.rrwp_steps,
            sym_norm=c.sym_norm, max_nodes=c.max_nodes, max_edges=c.max_edges)
        dtype = self.node_pe.weight.dtype
        walk, positional, order = (v.to(dtype=dtype) for v in (walk, positional, order))
        x = sum(embedding(graph.x[:, i]) for i, embedding in enumerate(self.atoms))
        x = x + self.node_pe(walk.diagonal(dim1=0, dim2=1).t()) + self.node_order(order)
        n = len(x)
        pair = self.pe_mlp(sinusoidal_pe(positional, c.spe_bases).reshape(n * n, -1)).reshape(n, n, c.pair_dim)
        bonds = sum(embedding(graph.edge_attr[:, i]) for i, embedding in enumerate(self.bonds))
        # Author to_dense_adj(...).transpose(1,2): query=target, key=source.
        edge_values = torch.zeros_like(pair)
        edge_values.index_put_((graph.edge_index[1], graph.edge_index[0]), bonds, accumulate=True)
        pair = self.stem_pre_norm(pair + edge_values)
        for stem_block in self.stem:
            pair = stem_block(pair)
        pair = self.stem_post_norm(pair)
        x = self.pre_backbone_norm(x)
        for block in self.blocks:
            x = block(x, pair)
        x = self.post_backbone_norm(x)
        pooled = x.sum(0) if c.pooling == "sum" else x.mean(0)
        return self.head(pooled).reshape(())

    def forward(self, batch):
        c = self.config
        graphs = validated_topology_views(batch, max_nodes=c.max_nodes,
                                          max_edges=c.max_edges, allowed_roles=ALLOWED_ROLES)
        return torch.stack([self.graph_forward(graph) for graph in graphs])


def make_model(arm):
    return PPGTAuthor(PPGTAuthorConfig(**deepcopy(arm.get("model_config", {}))))


CONFIG_TYPE = PPGTAuthorConfig
