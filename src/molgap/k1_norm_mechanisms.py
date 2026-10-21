"""Frozen K1 normalization mechanisms, applied after base loading before optimizer.

Resume reconstructs the same surgery before strict candidate state loading.
Only local norm1/norm3 and their placement change; the existing conv, MLP,
edge updates, mixers and readout retain their parameter objects.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import ResGatedGraphConv
from torch_geometric.nn.norm import BatchNorm

from .model_substrate import ModelSubstrate
from .pcqm_gap_architecture import OGBEdgeStateStructuralGPSWrapper

_WIDTH = 192
VARIANTS = ("post_node_rms", "pre_node_rms", "post_graph_rms", "post_node_ada")


class NodeRMSNorm(nn.Module):
    """Per-node feature RMS with the frozen BN outer affine."""

    def __init__(self, base):
        super().__init__()
        self.eps = base.eps
        self.weight = nn.Parameter(base.weight.detach().clone(), requires_grad=base.weight.requires_grad)
        self.bias = nn.Parameter(base.bias.detach().clone(), requires_grad=base.bias.requires_grad)
        self.train(base.training)

    def forward(self, x, batch=None):
        return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + self.eps) * self.weight + self.bias


class GraphRMSNorm(NodeRMSNorm):
    """Per-graph, per-channel RMS across nodes; no centering or graph coupling."""

    def forward(self, x, batch=None):
        if batch is None:
            moment = x.square().mean(dim=0, keepdim=True)
        else:
            # Unique handles noncontiguous graph IDs without allocating max(ID).
            graph_ids, inverse = torch.unique(batch, sorted=True, return_inverse=True)
            total = x.new_zeros((graph_ids.numel(), x.shape[-1]))
            total.index_add_(0, inverse, x.square())
            count = torch.bincount(inverse, minlength=total.shape[0]).to(x.dtype).unsqueeze(-1)
            moment = (total / count)[inverse]
        return x * torch.rsqrt(moment + self.eps) * self.weight + self.bias


class NodeAdaNorm(NodeRMSNorm):
    """PPGT adaptive RMS: gamma*(x/rms(x))*rms(alpha*x+beta)+bias.

    Both RMS operations use the same epsilon so alpha=1,beta=0 restores
    amplitude exactly (up to floating arithmetic). At alpha=0,beta=1 the
    RMS multiplier is sqrt(1+eps), within eps of ordinary node RMS.
    """

    def __init__(self, base):
        super().__init__(base)
        self.alpha = nn.Parameter(torch.zeros_like(self.weight))
        self.beta = nn.Parameter(torch.ones_like(self.weight))

    def forward(self, x, batch=None):
        rms = (x.square().mean(dim=-1, keepdim=True) + self.eps).sqrt()
        adaptive = ((self.alpha * x + self.beta).square().mean(dim=-1, keepdim=True) + self.eps).sqrt()
        return self.weight * (x / rms) * adaptive + self.bias


class _NormLocalBlock(nn.Module):
    def __init__(self, base, norm1, norm3, *, prenorm):
        super().__init__()
        self.conv, self.mlp = base.conv, base.mlp
        self.norm1, self.norm3 = norm1, norm3
        self.dropout = base.dropout
        self.norm_with_batch = True
        self.prenorm = prenorm
        self.training = base.training

    def forward(self, x, edge_index, batch, *, edge_attr):
        node = self.norm1(x, batch) if self.prenorm else x
        local = self.conv(node, edge_index, edge_attr=edge_attr)
        h = x + F.dropout(local, p=self.dropout, training=self.training)
        if self.prenorm:
            return h + self.mlp(self.norm3(h, batch))
        h = self.norm1(h, batch)
        return self.norm3(h + self.mlp(h), batch)

def _validated_sites(model):
    if (not isinstance(model, OGBEdgeStateStructuralGPSWrapper)
            or type(model).__module__ != "molgap.qm9_neural_atom"
            or type(model).__qualname__ != "make_encoder.<locals>.NeuralAtomEdgeStateGPS"
            or type(getattr(model, "local_blocks", None)) is not nn.ModuleList
            or len(model.local_blocks) != 9
            or type(getattr(model, "edge_updates", None)) is not nn.ModuleList
            or len(getattr(model, "edge_updates", ())) != 9
            or model.edge_state_channels != 64 or model.rwse_dim != 16
            or hasattr(model, "convs")
            or getattr(model, "pooling", None) != "mean"):
        raise ValueError("Expected frozen nine-block K1 topology")
    mixers = getattr(model, "neural_atom_mixers", None)
    if (type(mixers) is not nn.ModuleDict or tuple(mixers) != ("3", "6", "9")
            or any(type(m).__module__ != "molgap.qm9_neural_atom"
                   or type(m).__qualname__ != "_NeuralAtomMixerFactory.make.<locals>.NeuralAtomMixer"
                   or m.active_slots != 1 or m.max_slots != 4
                   or m.latent_channels != 64 for m in mixers.values())):
        raise ValueError("Expected frozen K1 slot topology")
    head = getattr(model, "head", None)
    if (type(head) is not nn.Sequential
            or tuple(type(m) for m in head) != (
                nn.Linear, nn.SiLU, nn.Dropout, nn.Linear, nn.SiLU, nn.Linear)
            or [(m.in_features, m.out_features) for m in head if type(m) is nn.Linear]
            != [(192, 192), (192, 96), (96, 1)]):
        raise ValueError("Expected K1 readout head")
    sites = []
    expected_paths = set()
    for index, block in enumerate(model.local_blocks):
        if (type(block).__module__ != "molgap.qm9_neural_atom"
                or type(block).__qualname__ != "_LocalGPSBlockFactory.make.<locals>.LocalGPSBlock"
                or set(block._modules) != {"conv", "mlp", "norm1", "norm3"}
                or type(block.conv) is not ResGatedGraphConv
                or type(block.mlp) is not nn.Sequential
                or tuple(type(m) for m in block.mlp) != (
                    nn.Linear, nn.SiLU, nn.Dropout, nn.Linear, nn.Dropout)
                or [(m.in_features, m.out_features) for m in block.mlp if type(m) is nn.Linear]
                != [(192, 384), (384, 192)]
                or block.norm_with_batch is not False):
            raise ValueError("Unexpected local block topology or normalization placement")
        for name in ("norm1", "norm3"):
            wrapper = getattr(block, name)
            if (type(wrapper) is not BatchNorm
                    or set(wrapper._modules) != {"module"}
                    or getattr(wrapper, "allow_single_element", False) is not False
                    or type(wrapper.module) is not nn.BatchNorm1d):
                raise ValueError("Expected tracked affine BatchNorm1d sites")
            norm = wrapper.module
            if (norm.num_features != _WIDTH or not norm.affine
                    or not norm.track_running_stats or norm.eps != 1e-5
                    or norm.momentum != 0.1
                    or norm.weight is None or norm.bias is None
                    or norm.running_mean is None or norm.running_var is None
                    or norm.num_batches_tracked is None
                    or norm.weight.shape != (_WIDTH,) or norm.bias.shape != (_WIDTH,)
                    or norm.weight.dtype != norm.bias.dtype
                    or norm.weight.device != norm.bias.device):
                raise ValueError("Unexpected BatchNorm width, affine state or tracking contract")
            sites.append((block, name, wrapper, norm))
            expected_paths.add(f"local_blocks.{index}.{name}.module")
    actual_paths = {name for name, module in model.named_modules(remove_duplicate=False)
                    if isinstance(module, nn.modules.batchnorm._BatchNorm)}
    if (actual_paths != expected_paths or len(sites) != 18
            or len({id(norm) for _, _, _, norm in sites}) != 18):
        raise ValueError("Expected exactly 18 uniquely owned tracked BN sites")
    return sites


def apply_addon(model, config):
    """Validate all 18 BN sites before mutation; consume no RNG or data roles."""
    if (type(config) is not dict or set(config) != {"variant"}
            or type(config["variant"]) is not str or config["variant"] not in VARIANTS):
        raise ValueError("k1 normalization config requires one reviewed variant")
    owner = getattr(model, "_molgap_substrate", None)
    if owner is not None and (type(owner) is not ModelSubstrate
            or owner.model is not model or owner.family != "neural_atom_k1"
            or owner.sealed or owner.bindings or owner.mutations or owner._active):
        raise ValueError("Expected an unsealed, empty K1 substrate")
    if hasattr(model, ModelSubstrate.namespace):
        raise ValueError("Model already has addon ownership")
    sites = _validated_sites(model)
    variant = config["variant"]
    kind = GraphRMSNorm if variant == "post_graph_rms" else NodeAdaNorm if variant == "post_node_ada" else NodeRMSNorm
    norms = [kind(norm).train(wrapper.training) for _, _, wrapper, norm in sites]
    replacements = [_NormLocalBlock(block, norms[2*i], norms[2*i+1],
                    prenorm=variant == "pre_node_rms") for i, block in enumerate(model.local_blocks)]
    if owner is None:
        owner = ModelSubstrate(model, "neural_atom_k1")
    for i, replacement in enumerate(replacements):
        owner.replace_submodule(f"k1_norm_mechanism_block_{i}", f"local_blocks.{i}", replacement)
    from .k1_norm_observations import NormDiagnostics
    object.__setattr__(model, "_k1_norm_diagnostics", NormDiagnostics(variant))
    return model


__all__ = ["apply_addon", "NodeRMSNorm", "GraphRMSNorm", "NodeAdaNorm", "VARIANTS"]
