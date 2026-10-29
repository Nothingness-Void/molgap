"""Zero-start extremal/second-moment readout branch beside K1 mean pooling."""
from dataclasses import replace

import torch
from torch import nn

from .model_substrate import ModelSubstrate, SlotBinding, SlotName

FAMILY = "neural_atom_k1"
MIXER = "9"
EPS = 1e-6


class _Tap(nn.Module):
    """PRE_READOUT seam: embedding + W [max, min, std] of the final node states.

    The holder is a plain dict (a tensor held as a submodule would enter state_dict).
    The mixer-9 forward hook fills it; this tap pops it, so every forward (train,
    eval, no_grad) leaves it empty. W starts at zero, so step 0 equals the baseline.
    """

    def __init__(self, holder, channels):
        super().__init__()
        self._holder = holder
        self.projection = nn.Linear(3 * channels, channels, bias=False)
        nn.init.zeros_(self.projection.weight)

    def forward(self, payload):
        if "h" not in self._holder:
            raise RuntimeError("Extremal readout tap ran before the mixer-9 forward of the same pass")
        h, batch, count = (self._holder.pop(key) for key in ("h", "batch", "count"))
        if count is None:
            count = int(batch.max()) + 1
        index = batch.view(-1, 1).expand_as(h)
        # include_self=False keeps the zero start for a graph with no nodes (no +-inf).
        top = h.new_zeros(count, h.shape[1]).scatter_reduce(0, index, h, "amax", include_self=False)
        bottom = h.new_zeros(count, h.shape[1]).scatter_reduce(0, index, h, "amin", include_self=False)
        size = torch.bincount(batch, minlength=count).clamp_min(1).to(h.dtype).unsqueeze(1)
        mean = h.new_zeros(count, h.shape[1]).index_add(0, batch, h) / size
        variance = h.new_zeros(count, h.shape[1]).index_add(0, batch, (h - mean[batch]) ** 2) / size
        spread = torch.sqrt(variance + EPS)  # eps keeps the gradient finite at zero variance
        extra = self.projection(torch.cat([top, bottom, spread], dim=1))
        return replace(payload, embedding=payload.embedding + extra)


def apply_addon(model, config):
    mixer = getattr(model, "neural_atom_mixers", None)
    if mixer is None or MIXER not in mixer:
        raise ValueError("Extremal readout requires the K1 backbone with mixer 9")
    owner = getattr(model, "_molgap_substrate", None)
    if owner is None:
        owner = ModelSubstrate(model, FAMILY)
    if owner.family != FAMILY:
        raise ValueError("Extremal readout requires the K1 substrate")
    holder = {}

    def stash(module, args, output):
        # args = (hidden, batch); the output is the final node state before pooling.
        holder.update(h=output, batch=args[1], count=getattr(model, "_execution_graph_count", None))

    mixer[MIXER].register_forward_hook(stash)
    owner.attach("extremal_readout", _Tap(holder, output_channels(model)),
                 SlotBinding(FAMILY, SlotName.PRE_READOUT))
    return model


def output_channels(model):
    return model.head[0].in_features
