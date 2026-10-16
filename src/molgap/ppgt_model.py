"""Paper-equation PPGT CPU model. Upstream/full-benchmark fidelity is pending."""
from copy import deepcopy
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from .modern2d_inputs import topology_views
from .ogb_features import ATOM_FEATURE_DIMS, BOND_FEATURE_DIMS
from .ppgt_features import positional_inputs, sinusoidal_pe


@dataclass(frozen=True)
class PPGTConfig:
    hidden: int = 256
    blocks: int = 16
    heads: int = 8
    pair_dim: int = 128
    pe_mlp_hidden: int = 512
    stem_blocks: int = 8
    expansion: int = 2
    rrwp_steps: int = 24
    spe_bases: int = 10
    use_phi: bool = True
    pooling: str = "sum"
    epsilon: float = 1e-8
    attention_dropout: float = 0.0
    stochastic_depth: float = 0.0

    def __post_init__(self):
        for k in ("hidden", "blocks", "heads", "pair_dim", "pe_mlp_hidden", "stem_blocks", "expansion", "rrwp_steps"):
            if type(getattr(self, k)) is not int or getattr(self, k) < 1:
                raise ValueError("Positive integer PPGT dimensions required")
        if self.hidden < 8 or self.hidden % self.heads or type(self.spe_bases) is not int or self.spe_bases < 0:
            raise ValueError("Invalid PPGT dimensions/SPE")
        if type(self.use_phi) is not bool or self.pooling not in {"sum", "mean"}:
            raise ValueError("Explicit PPGT phi/readout required")
        if not math.isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError("Positive norm epsilon required")
        if any(type(p) not in (int, float) or not 0 <= p < 1 for p in (self.attention_dropout, self.stochastic_depth)):
            raise ValueError("Invalid dropout/stochastic-depth probability")


def stable_rms(x, epsilon):
    # Rescaling avoids FP32 squaring overflow without hidden FP64 promotion.
    scale = x.abs().amax(-1, keepdim=True).clamp_min(math.sqrt(epsilon))
    return scale * ((x / scale).square().mean(-1, keepdim=True) + (math.sqrt(epsilon) / scale).square()).sqrt()


class AdaRMSNorm(nn.Module):
    def __init__(self, dim, epsilon=1e-8):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(dim))
        self.beta = nn.Parameter(torch.ones(dim))
        self.epsilon = epsilon

    def forward(self, x):
        return (x / stable_rms(x, self.epsilon)) * stable_rms(self.alpha * x + self.beta, self.epsilon)


def sl2_attention(q, k, v, *, bias=None, multiplier=None, allowed=None, dropout=0.0):
    """Eq4/5, heads x queries/keys x head_dim. True means allowed key."""
    logits = (q @ k.transpose(-1, -2) - 0.5 * k.square().sum(-1).unsqueeze(-2)) / math.sqrt(q.shape[-1])
    if bias is not None:
        logits = logits + bias
    if allowed is not None:
        if allowed.dtype != torch.bool:
            raise TypeError("Attention allowed mask must be bool")
        logits = logits.masked_fill(~allowed, -torch.inf)
        # Empty padded queries contribute exactly zero rather than NaN.
        logits = torch.where(allowed.any(-1, keepdim=True), logits, 0)
    weights = logits.softmax(-1)
    if allowed is not None:
        weights = weights * allowed
    if multiplier is not None:
        weights = weights * multiplier
    return F.dropout(weights, p=dropout, training=dropout > 0) @ v


class SL2Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.qkv = nn.Linear(config.hidden, 3 * config.hidden)
        self.theta = nn.Linear(config.pair_dim, config.heads)
        self.phi = nn.Linear(config.pair_dim, config.heads) if config.use_phi else None
        self.proj = nn.Linear(config.hidden, config.hidden)

    def forward(self, x, pair):
        c = self.config
        q, k, v = self.qkv(x).reshape(len(x), 3, c.heads, c.hidden // c.heads).permute(1, 2, 0, 3).unbind(0)
        value = sl2_attention(q, k, v, bias=self.theta(pair).permute(2, 0, 1),
                              multiplier=self.phi(pair).permute(2, 0, 1) if self.phi is not None else None,
                              dropout=c.attention_dropout if self.training else 0)
        return self.proj(value.transpose(0, 1).reshape(len(x), c.hidden))


class PreNormFFN(nn.Module):
    def __init__(self, dim, expansion, epsilon):
        super().__init__()
        self.norm = AdaRMSNorm(dim, epsilon)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * expansion), nn.GELU(), nn.Linear(dim * expansion, dim))

    def forward(self, x):
        return x + self.mlp(self.norm(x))


class PPGTBlock(nn.Module):
    def __init__(self, config, drop_path):
        super().__init__()
        self.norm = AdaRMSNorm(config.hidden, config.epsilon)
        self.attention = SL2Attention(config)
        self.ffn = PreNormFFN(config.hidden, config.expansion, config.epsilon)
        self.drop_path = drop_path

    def residual(self, value):
        if self.training and self.drop_path:
            # A graph is the stochastic-depth sample; not individual atoms.
            keep = 1 - self.drop_path
            return value * torch.empty((), device=value.device).bernoulli_(keep) / keep
        return value

    def forward(self, x, pair):
        x = x + self.residual(self.attention(self.norm(x), pair))
        return x + self.residual(self.ffn(x) - x)


class PPGT(nn.Module):
    def __init__(self, config=PPGTConfig()):
        super().__init__()
        if type(config) is not PPGTConfig:
            raise TypeError("Frozen PPGTConfig required")
        self.config = config
        c = config
        self.atoms = nn.ModuleList(nn.Embedding(d, c.hidden) for d in ATOM_FEATURE_DIMS)
        self.bonds = nn.ModuleList(nn.Embedding(d, c.pair_dim) for d in BOND_FEATURE_DIMS)
        self.node_pe = nn.Linear(c.rrwp_steps, c.hidden)
        self.node_size = nn.Linear(2, c.hidden)
        self.pe_mlp = nn.Sequential(nn.Linear((c.rrwp_steps + 3) * (1 + 2 * c.spe_bases), c.pe_mlp_hidden),
                                    nn.GELU(), nn.Linear(c.pe_mlp_hidden, c.pair_dim))
        self.stem = nn.Sequential(*(PreNormFFN(c.pair_dim, c.expansion, c.epsilon) for _ in range(c.stem_blocks)),
                                  AdaRMSNorm(c.pair_dim, c.epsilon))
        self.blocks = nn.ModuleList(PPGTBlock(c, c.stochastic_depth * (i + 1) / c.blocks) for i in range(c.blocks))
        self.norm = AdaRMSNorm(c.hidden, c.epsilon)
        self.head = nn.Sequential(nn.Linear(c.hidden, c.hidden // 2), nn.GELU(),
                                  nn.Linear(c.hidden // 2, c.hidden // 4), nn.GELU(), nn.Linear(c.hidden // 4, 1))
        for scope in (self.blocks, self.norm, self.head):
            for module in scope.modules():
                if isinstance(module, nn.Linear):
                    nn.init.trunc_normal_(module.weight, std=0.02)
                    nn.init.zeros_(module.bias)
        for scope in (self.pe_mlp, self.stem, self.node_pe, self.node_size):
            for module in scope.modules():
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_uniform_(module.weight, a=1)
                    nn.init.zeros_(module.bias)
        # Hidden stem projections use the paper's distinct a=0 convention.
        nn.init.kaiming_uniform_(self.pe_mlp[0].weight, a=0)
        for ffn in self.stem:
            if isinstance(ffn, PreNormFFN):
                nn.init.kaiming_uniform_(ffn.mlp[0].weight, a=0)

    def graph_forward(self, graph):
        c = self.config
        walk, positional, size = positional_inputs(graph, c.rrwp_steps)
        x = sum(embedding(graph.x[:, i]) for i, embedding in enumerate(self.atoms))
        x = x + self.node_pe(walk.diagonal(dim1=0, dim2=1).t()) + self.node_size(size)
        pair = self.pe_mlp(sinusoidal_pe(positional, c.spe_bases))
        bond = sum(embedding(graph.edge_attr[:, i]) for i, embedding in enumerate(self.bonds))
        edge_values = torch.zeros_like(pair)
        edge_values[graph.edge_index[0], graph.edge_index[1]] = bond
        pair = self.stem(pair + edge_values)
        for block in self.blocks:
            x = block(x, pair)
        x = self.norm(x)
        pooled = x.sum(0) if c.pooling == "sum" else x.mean(0)
        return self.head(pooled).reshape(())

    def forward(self, batch):
        return torch.stack([self.graph_forward(g) for g in topology_views(batch)])


def make_model(arm):
    values = deepcopy(arm.get("model_config", {}))
    return PPGT(PPGTConfig(**values))


CONFIG_TYPE = PPGTConfig
