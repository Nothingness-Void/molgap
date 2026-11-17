"""Small registered slot example shared by the two accelerated backbones."""
from dataclasses import replace

import torch

from .model_substrate import ModelSubstrate, SlotBinding, SlotName


class SlotScale(torch.nn.Module):
    def __init__(self, field):
        super().__init__()
        self.field = field
        self.gain = torch.nn.Parameter(torch.ones(()))

    def forward(self, payload):
        return replace(payload, **{self.field: getattr(payload, self.field) * self.gain})


def apply_addon(model, config):
    """Static registration binds family/slot/name; JSON cannot name Python hooks."""
    family, slot, name = config["family"], SlotName(config["slot"]), config["name"]
    owner = getattr(model, "_molgap_substrate", None)
    if owner is None:
        owner = ModelSubstrate(model, family)
    if owner.family != family:
        raise ValueError("Addon substrate family differs")
    field = ("embedding" if slot is SlotName.PRE_READOUT else "node" if slot is SlotName.PRE_MESSAGE
             else "aggregated" if family == "neural_atom_k1" else "node_update")
    owner.attach(name, SlotScale(field), SlotBinding(family, slot,
        None if slot is SlotName.PRE_READOUT else 1))
    return model
