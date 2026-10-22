"""Graph-wise stochastic masking of K1 neighbor aggregates, not whole blocks."""
from dataclasses import replace
import weakref

from torch import nn

from .model_substrate import ModelSubstrate, SlotBinding, SlotName


class MessageDropPath(nn.Module):
    def __init__(self, model, probability):
        super().__init__()
        if type(probability) not in (int, float) or not 0 <= probability < 1:
            raise ValueError("Message DropPath probability must satisfy 0 <= p < 1")
        self.probability = probability
        # A module reference here would register the entire backbone as a child.
        self._model_ref = weakref.ref(model)

    def forward(self, payload):
        if not self.training or self.probability == 0:
            return payload
        graphs = getattr(self._model_ref(), "_execution_graph_count", None)
        if type(graphs) is not int or graphs <= 0:
            raise ValueError("Message DropPath requires the accelerated graph count")
        keep = 1 - self.probability
        mask = payload.aggregated.new_empty((graphs, 1)).bernoulli_(keep).div_(keep)
        return replace(payload, aggregated=payload.aggregated * mask[payload.graph_index])


def apply_addon(model, config):
    """Attach independent per-layer masks at all nine pre-root aggregate seams."""
    if len(getattr(model, "local_blocks", ())) != 9:
        raise ValueError("Message DropPath requires the nine-layer K1 backbone")
    owner = getattr(model, "_molgap_substrate", None)
    if owner is None:
        owner = ModelSubstrate(model, "neural_atom_k1")
    if owner.family != "neural_atom_k1":
        raise ValueError("Message DropPath requires the K1 substrate")
    for layer in range(1, 10):
        owner.attach(f"message_droppath_{layer}", MessageDropPath(model, config["probability"]),
            SlotBinding("neural_atom_k1", SlotName.POST_AGGREGATION, layer))
    return model
