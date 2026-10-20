"""Frozen nine-block K1 post-residual BN -> node-wise LayerNorm surgery.

Call after loading frozen *base* weights and before optimizer creation. Resume
instead reconstructs this surgery before strictly loading candidate state. The
family kernel owns that ordering; this addon does not load weights or train.
"""
from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import ResGatedGraphConv
from torch_geometric.nn.norm import BatchNorm

from .model_substrate import ModelSubstrate, SlotBinding, SlotName, bind_substrate
from .pcqm_gap_architecture import OGBEdgeStateStructuralGPSWrapper


_MARKER = "k1_post_layernorm"
_WIDTH = 192


class _SurgeryMarker(nn.Module):
    """Parameter/buffer-free identity recording surgery at PRE_READOUT.

    This is bookkeeping, not a readout mechanism. LayerNorm parameters belong
    exclusively to local_blocks, never to the substrate addon namespace.
    """

    def forward(self, payload):
        return payload


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
    """Apply only the reviewed empty-config surgery; reject reapplication.

    The caller must have loaded the frozen BN base first. Running statistics
    are deliberately discarded; affine values/grad flags, dtype/device, epsilon
    and local normalization placement are retained. No RNG is consumed.
    """
    if type(config) is not dict or config:
        raise ValueError("k1_post_layernorm requires config={}")
    owner = getattr(model, "_molgap_substrate", None)
    if owner is not None and (type(owner) is not ModelSubstrate
            or owner.model is not model or owner.family != "neural_atom_k1"
            or owner.sealed or owner.bindings or owner._active):
        raise ValueError("Expected an unsealed, empty K1 substrate")
    if hasattr(model, ModelSubstrate.namespace):
        raise ValueError("Model already has addon ownership")
    sites = _validated_sites(model)
    replacements = []
    for block, name, wrapper, norm in sites:
        replacement = nn.LayerNorm(_WIDTH, eps=norm.eps, elementwise_affine=True,
                                   device=norm.weight.device, dtype=norm.weight.dtype)
        with torch.no_grad():
            replacement.weight.copy_(norm.weight)
            replacement.bias.copy_(norm.bias)
        replacement.weight.requires_grad_(norm.weight.requires_grad)
        replacement.bias.requires_grad_(norm.bias.requires_grad)
        replacement.train(wrapper.training)
        replacements.append((block, name, replacement))
    binding = SlotBinding("neural_atom_k1", SlotName.PRE_READOUT)
    if owner is None:
        bind_substrate(model, "neural_atom_k1", [(_MARKER, _SurgeryMarker(), binding)])
    else:
        owner.attach(_MARKER, _SurgeryMarker(), binding)
    for block, name, replacement in replacements:
        setattr(block, name, replacement)
    return model


def policy(mode):
    from .family_training_policy import FamilyModePolicy
    if mode != "post_layernorm":
        raise ValueError("Unknown post-LayerNorm mode")
    return FamilyModePolicy(objective_name="reference")


__all__ = ["apply_addon", "policy"]
