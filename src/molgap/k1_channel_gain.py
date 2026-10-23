"""Identity-initialized channel gain for the local lifecycle exercise."""
from dataclasses import replace

import torch

from .model_substrate import ModelSubstrate, SlotBinding, SlotName


class ChannelGain(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gain = torch.nn.Parameter(torch.ones(channels))

    def forward(self, payload):
        return replace(payload, aggregated=payload.aggregated * self.gain)


def apply_addon(model, config):
    owner = getattr(model, "_molgap_substrate", None)
    if owner is None:
        owner = ModelSubstrate(model, "neural_atom_k1")
    owner.attach("channel_gain", ChannelGain(config["channels"]),
                 SlotBinding("neural_atom_k1", SlotName.POST_AGGREGATION, config["layer"]))
    return model
