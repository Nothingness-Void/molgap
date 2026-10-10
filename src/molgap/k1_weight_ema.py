"""Parameter-only K1 EMA; buffers are copied from the live training model.

Reviewed extraction of make_ema/update_ema and DECAY from
codex/exp/k1-ema-100k-night-20261006 at
667b68dc15b9750f082e5b68564ca2972a5ee8f7 (src/molgap/k1_weight_ema.py).
The owner's 100K qualification/selection policy is deliberately not imported.
"""
from __future__ import annotations

DECAY = 0.999


def make_ema(model):
    import copy
    shadow = copy.deepcopy(model).eval()
    shadow.requires_grad_(False)
    return shadow


def update_ema(shadow, model) -> None:
    import torch
    with torch.no_grad():
        live = dict(model.named_parameters())
        for name, parameter in shadow.named_parameters():
            parameter.mul_(DECAY).add_(live[name].detach(), alpha=1.0 - DECAY)
        buffers = dict(model.named_buffers())
        for name, value in shadow.named_buffers():
            value.copy_(buffers[name].detach())
