"""DeeperGCN message normalization at K1's neighbor aggregation seam."""
from dataclasses import replace

import torch
from torch import nn

from .model_substrate import ModelSubstrate, SlotBinding, SlotName


class MessageNorm(nn.Module):
    """Scale the neighbor aggregate to the receiving node's input norm."""

    def __init__(self, learn_scale=False):
        super().__init__()
        if type(learn_scale) is not bool:
            raise ValueError("learn_scale must be boolean")
        scale = torch.ones(())
        if learn_scale:
            self.scale = nn.Parameter(scale)
        else:
            self.register_buffer("scale", scale)

    def forward(self, payload):
        if payload.node is None:
            raise ValueError("MessageNorm requires the receiving node input")
        message = payload.aggregated
        node_norm = torch.linalg.vector_norm(payload.node, dim=-1, keepdim=True)
        message_norm = torch.linalg.vector_norm(message, dim=-1, keepdim=True)
        normalized = message / message_norm.clamp_min(1e-7)
        return replace(payload, aggregated=self.scale * node_norm * normalized)


def apply_addon(model, config):
    """Attach one independent scalar to each of the nine retained K1 layers."""
    if type(config) is not dict or set(config) != {"learn_scale"}:
        raise ValueError("MessageNorm requires the bounded learn_scale configuration")
    if type(config["learn_scale"]) is not int or config["learn_scale"] not in (0, 1):
        raise ValueError("learn_scale must be integer 0 or 1")
    if len(model.local_blocks) != 9:
        raise ValueError("MessageNorm requires the retained nine-layer K1 backbone")
    owner = getattr(model, "_molgap_substrate", None)
    if owner is None:
        owner = ModelSubstrate(model, "neural_atom_k1")
    if owner.family != "neural_atom_k1":
        raise ValueError("MessageNorm requires a K1 substrate")
    for layer in range(1, 10):
        owner.attach("message_norm_" + str(layer), MessageNorm(bool(config["learn_scale"])),
                     SlotBinding("neural_atom_k1", SlotName.POST_AGGREGATION, layer))
    return model
