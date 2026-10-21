"""Opt-in ESA encoder residual research; retained native weights and modules.

Reconstruct this addon before loading an addon checkpoint. Only the M/S encoder
residual source changes; native tokenization, padding, initial skip and invariant
PMA remain owned by ESANative. This hook supplies no training or run authority.
"""
from types import MethodType

from torch import nn

from .esa_native import ESANative, _PaddingAwareBlock
from .model_substrate import ModelSubstrate, SlotBinding, SlotName, bind_substrate


_NAME = "esa_identity_residual"
_MARKER = "_molgap_esa_identity_residual"


def _identity_residual_forward(block, x, allowed=None):
    residual = x
    normalized = block.norm(x)
    out = residual + block.attention(normalized, normalized, allowed)
    return out + block.mlp(block.norm_mlp(out))


def apply_addon(model, config):
    """Bind canonical pre-LN M/S residuals to an existing exact ESANative.

    The stateless readout identity declares the attachment to the shared family
    kernel; the scientific mechanism lives exclusively in the encoder methods.
    No constructor, initialization, parameter, buffer or config is replaced.
    """
    if (type(config) is not dict or set(config) != {"variant"}
            or type(config["variant"]) is not str
            or config["variant"] != "identity_residual"):
        raise ValueError("ESA research addon requires exactly variant='identity_residual'")
    if type(model) is not ESANative:
        raise TypeError("ESA identity residual requires an exact ESANative model")
    if hasattr(model, _MARKER):
        raise ValueError("ESA identity residual addon is already applied")
    blocks = tuple(model.encoder)
    if (not blocks or tuple(block.kind for block in blocks) != model.config.layer_types[:-1]
            or any(type(block) is not _PaddingAwareBlock or block.kind not in {"M", "S"}
                   or "forward" in block.__dict__ for block in blocks)):
        raise ValueError("ESA identity residual requires unmodified native M/S encoder blocks")
    owner = getattr(model, "_molgap_substrate", None)
    binding = SlotBinding("esa_small", SlotName.PRE_READOUT)
    if owner is None:
        bind_substrate(model, "esa_small", ((_NAME, nn.Identity(), binding),))
    else:
        if type(owner) is not ModelSubstrate or owner.model is not model or owner.family != "esa_small":
            raise ValueError("ESA research addon substrate owner differs")
        owner.attach(_NAME, nn.Identity(), binding)
    for block in blocks:
        block.forward = MethodType(_identity_residual_forward, block)
    object.__setattr__(model, _MARKER, True)
    return model


__all__ = ["apply_addon"]
