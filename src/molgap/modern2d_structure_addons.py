"""Bounded structural falsifiers over the retained PPGT/3 and ESA/4 owners.

These are scientific interventions, not execution repairs or author replicas.
They reuse the existing addon state namespace, model factories and Sessions.
No labels, geometry, corpus cache or training code is accessed here.
"""
from dataclasses import replace
from types import MethodType

import torch
from torch import nn

from .model_substrate import ModelSubstrate, SlotBinding, SlotName


class ContentPairConditioner(nn.Module):
    """Let current node content condition a retained structural pair tensor.

    Sharing the narrow conditioner across depth tests conditional use of PE,
    not additional slots, an independent pair-memory trunk or a wider model.
    The zero return preserves the complete base function at initialization.
    """
    def __init__(self, hidden, pair_dim, rank=16):
        super().__init__()
        self.left = nn.Linear(hidden, rank, bias=False)
        self.right = nn.Linear(hidden, rank, bias=False)
        self.return_projection = nn.Linear(rank, pair_dim, bias=False)
        nn.init.zeros_(self.return_projection.weight)

    def condition(self, node, pair):
        content = torch.tanh(self.left(node).unsqueeze(-2) + self.right(node).unsqueeze(-3))
        modulation = .5 * torch.tanh(self.return_projection(content))
        return pair * (1 + modulation)

    def forward(self, payload):
        # Parameter ownership uses the existing substrate. The declared body
        # intervention runs at each AuthorAttention, not at the readout seam.
        return payload


class CountConditionedReadout(nn.Module):
    """Graph-size FiLM after invariant pooling, with an exact identity start."""
    def __init__(self, graph_dim):
        super().__init__()
        self.scale = nn.Linear(1, graph_dim, bias=False)
        self.shift = nn.Linear(1, graph_dim, bias=False)
        nn.init.zeros_(self.scale.weight)
        nn.init.zeros_(self.shift.weight)
        self._frames = []

    def begin(self, model, args, kwargs):
        batch = args[0] if args else kwargs.get("batch")
        ptr = getattr(batch, "ptr", None)
        if (not torch.is_tensor(ptr) or ptr.dtype != torch.int64 or ptr.ndim != 1
                or ptr.numel() < 2):
            # always_call cleanup must not pop another invocation's frame.
            self._frames.append(None)
            raise ValueError("ESA count conditioning requires explicit graph boundaries")
        self._frames.append((ptr[1:] - ptr[:-1]).detach())

    def finish(self, model, args, kwargs, output):
        if self._frames:
            self._frames.pop()

    def forward(self, payload):
        if not self._frames or self._frames[-1] is None:
            raise RuntimeError("ESA count conditioning requires its owning model call")
        embedding = payload.embedding
        counts = self._frames[-1]
        if embedding.ndim != 2 or embedding.shape[0] != counts.numel():
            raise ValueError("ESA pooled representations and graph boundaries differ")
        size = torch.log1p(counts.to(device=embedding.device, dtype=embedding.dtype))[:, None]
        return replace(payload, embedding=embedding * (1 + .5 * torch.tanh(self.scale(size)))
                       + self.shift(size))


def _conditioned_attention(attention, node, pair):
    return attention._molgap_unconditioned_forward(
        node, attention._molgap_pair_conditioner.condition(node, pair))


def apply_addon(model, config):
    """Reconstruct before optimizer creation and before loading addon weights."""
    from .ppgt_author import PPGTAuthor, AuthorAttention
    from .esa_native import ESANative

    if type(config) is not dict or set(config) != {"variant"}:
        raise ValueError("A single explicit Modern2D structural variant is required")
    variant = config["variant"]
    ppgt = variant == "content_pair"
    if variant not in {"content_pair", "count_readout"}:
        raise ValueError("Unsupported Modern2D structural variant")
    if type(model) is not (PPGTAuthor if ppgt else ESANative):
        raise TypeError("Structural variant differs from the retained model family")
    name = "ppgt_content_pair" if ppgt else "esa_count_readout"
    if hasattr(model, "_molgap_structural_variant"):
        raise ValueError("A structural variant is already installed")
    family = "ppgt_small" if ppgt else "esa_small"
    owner = getattr(model, "_molgap_substrate", None)
    if owner is None:
        owner = ModelSubstrate(model, family)
    if owner.family != family:
        raise ValueError("Structural addon owner differs from its family")
    if ppgt:
        attentions = [block.attention for block in model.blocks]
        if any(type(a) is not AuthorAttention or "forward" in vars(a) for a in attentions):
            raise ValueError("Content conditioning requires unchanged author attention")
        addon = ContentPairConditioner(model.config.hidden, model.config.pair_dim)
        owner.attach(name, addon, SlotBinding(family, SlotName.PRE_READOUT))
        for attention in attentions:
            # Object references avoid duplicate parameter/state namespaces.
            object.__setattr__(attention, "_molgap_pair_conditioner", addon)
            object.__setattr__(attention, "_molgap_unconditioned_forward", attention.forward)
            attention.forward = MethodType(_conditioned_attention, attention)
    else:
        addon = CountConditionedReadout(model.config.graph_dim)
        owner.attach(name, addon, SlotBinding(family, SlotName.PRE_READOUT))
        model.register_forward_pre_hook(addon.begin, with_kwargs=True)
        model.register_forward_hook(addon.finish, with_kwargs=True, always_call=True)
    object.__setattr__(model, "_molgap_structural_variant", variant)
    return model


__all__ = ["apply_addon", "ContentPairConditioner", "CountConditionedReadout"]
