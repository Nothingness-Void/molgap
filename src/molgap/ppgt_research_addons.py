"""Opt-in compact PPGT factorial mechanisms; no trainer or execution override.

The reference arm uses ``compact_config()`` without an addon. These mechanisms
are a fresh scientific scope, not an execution-equivalent repair of PPGT/2.
"""
from dataclasses import replace

import torch
from torch import nn

from .model_substrate import ModelSubstrate, SlotBinding, SlotName
from .ppgt_author import PPGTAuthor, PPGTAuthorConfig


def compact_config() -> PPGTAuthorConfig:
    """Return the shared sum-pooling initialization for all four fresh arms."""
    return PPGTAuthorConfig(
        hidden=192, blocks=8, heads=8, pair_dim=64, pe_mlp_hidden=256,
        stem_blocks=2, rrwp_steps=8, spe_bases=5, pooling="sum",
        layer_scale=1e-4, attention_dropout=0.0, stochastic_depth=0.0,
    )


def apply_addon(model, config):
    """Apply one reviewed variant before optimizer construction/state recovery.

    The identity slot declares the mechanism to the shared kernel. Pooling is
    handled by the existing forward paths, and residual scaling happens only
    here at initialization; neither operation is repeated by the readout hook.
    The shared construction owner seals the substrate after composition.
    """
    if (type(config) is not dict or set(config) != {"variant"}
            or type(config["variant"]) is not str
            or config["variant"] not in {"mean_readout", "residual_scale", "both"}):
        raise ValueError("Expected PPGT research variant: mean_readout, residual_scale, or both")
    if not isinstance(model, PPGTAuthor):
        raise TypeError("PPGT research addons require PPGTAuthor")
    if hasattr(model, "_molgap_ppgt_research_variant"):
        raise ValueError("PPGT research mechanism is already applied")

    variant = config["variant"]
    scales = ()
    if variant in {"residual_scale", "both"}:
        scales = tuple(scale for block in model.blocks
                       for scale in (block.attention_scale, block.ffn_scale))
        if model.config.layer_scale != 1e-4 or any(
                not torch.equal(scale.detach(), torch.full_like(scale, 1e-4))
                for scale in scales):
            raise ValueError("Residual scale requires fresh 1e-4 LayerScale tensors")
    configuration = (replace(model.config, pooling="mean")
                     if variant in {"mean_readout", "both"} else model.config)
    owner = getattr(model, "_molgap_substrate", None)
    if owner is None:
        owner = ModelSubstrate(model, "ppgt_small")
    if owner.family != "ppgt_small":
        raise ValueError("PPGT research substrate family differs")
    owner.attach("ppgt_research_" + variant, nn.Identity(),
                 SlotBinding("ppgt_small", SlotName.PRE_READOUT))
    with torch.no_grad():
        for scale in scales:
            scale.fill_(1e-2)
    model.config = configuration
    # Plain metadata leaves tensor keys unchanged and prevents reinitialization.
    model._molgap_ppgt_research_variant = variant
    return model


__all__ = ["apply_addon", "compact_config"]
