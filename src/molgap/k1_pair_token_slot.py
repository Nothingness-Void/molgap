"""Zero-start all-pair relation token on a K1/3 message input slot."""
from dataclasses import replace

import torch

from .k1_pair_token import MODE, _PairTokenFactory
from .model_substrate import ModelSubstrate, SlotBinding, SlotName


class PairTokenMessage(torch.nn.Module):
    """Reuse the qualified PairToken; its zero return projection makes step 0 exact."""

    def __init__(self):
        super().__init__()
        self.token = _PairTokenFactory.make(MODE)

    def forward(self, payload):
        update, _ = self.token.compute_update(payload.node, payload.graph_index)
        return replace(payload, node=payload.node + update)


def apply_addon(model, config):
    owner = getattr(model, "_molgap_substrate", None)
    if owner is None:
        owner = ModelSubstrate(model, "neural_atom_k1")
    # A private CPU stream keeps the trainer's global RNG, and so data order and
    # dropout, identical to a same-run reference arm.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(config["seed"])
        module = PairTokenMessage()
    owner.attach(config["name"], module, SlotBinding("neural_atom_k1", SlotName.PRE_MESSAGE, config["layer"]))
    return model
