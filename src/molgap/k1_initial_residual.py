"""Gated initial residual for the K1 local blocks (GCNII-style, zero start)."""
from dataclasses import replace

import torch
from torch import nn

from .model_substrate import ModelSubstrate, SlotBinding, SlotName

FAMILY = "neural_atom_k1"


class _Tap(nn.Module):
    """One conv-input seam. Layer 1 stores h0, layers 2-9 add gain * h0.

    The holder is a plain dict on purpose: a tensor or module reference held as a
    submodule would enter state_dict. Layer 1 overwrites it on every forward and
    the last layer drops it, so eval and no_grad passes leave nothing behind.
    """

    def __init__(self, holder, channels, layer, last):
        super().__init__()
        self._holder, self._layer, self._last = holder, layer, last
        # Layer 1 only records h0, so it owns no weights and keeps the state keys minimal.
        self.gain = nn.Parameter(torch.zeros(channels)) if layer > 1 else None

    def forward(self, payload):
        if self._layer == 1:
            self._holder["h0"] = payload.node
            return payload
        h0 = self._holder.get("h0")
        if h0 is None:
            raise RuntimeError("Initial residual tap ran before the layer-1 tap of the same forward")
        node = payload.node + self.gain * h0
        if self._last:
            self._holder.clear()
        return replace(payload, node=node)


def apply_addon(model, config):
    blocks = getattr(model, "local_blocks", ())
    if len(blocks) != 9:
        raise ValueError("Initial residual requires the nine-layer K1 backbone")
    owner = getattr(model, "_molgap_substrate", None)
    if owner is None:
        owner = ModelSubstrate(model, FAMILY)
    if owner.family != FAMILY:
        raise ValueError("Initial residual requires the K1 substrate")
    channels, holder = blocks[0].conv.in_channels, {}
    for layer in range(1, 10):
        owner.attach(f"initial_residual_{layer}", _Tap(holder, channels, layer, layer == 9),
                     SlotBinding(FAMILY, SlotName.PRE_MESSAGE, layer))
    return model
