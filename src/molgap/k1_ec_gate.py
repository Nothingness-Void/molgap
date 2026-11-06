"""Compare source-degree and expansion-contribution guidance on K1 bond messages."""
import torch
from torch import nn

from .model_substrate import ModelSubstrate, SlotBinding, SlotName


def edge_statistics(edge_index, nodes):
    """CPU bitsets implement Definition 1 on within-hop receptive fields."""
    source, target = edge_index.tolist()
    neighbors = [[] for _ in range(nodes)]
    for u, v in zip(source, target):
        neighbors[v].append(u)
    previous = [1 << i for i in range(nodes)]
    expansion = []
    for depth in range(1, 4):
        current = previous.copy()
        for v, incoming in enumerate(neighbors):
            for u in incoming:
                current[v] |= previous[u]
        if depth >= 2:
            expansion.append([(previous[u] & (current[v] ^ previous[v])).bit_count()
                              for u, v in zip(source, target)])
        previous = current
    degree = [max(len(neighbors[u]) - 1, 0) for u in source]
    return torch.tensor(degree, dtype=torch.float32), torch.tensor(expansion, dtype=torch.float32)


class TopologyContext:
    def __init__(self):
        self.statistics = None


class ExpansionGate(nn.Module):
    def __init__(self, context, layer, guidance):
        super().__init__()
        self.context, self.layer, self.guidance = context, layer, guidance
        self.projection = nn.Linear(192, 192, bias=False)
        nn.init.eye_(self.projection.weight)
        self.gate = nn.Sequential(nn.Linear(1, 16), nn.SiLU(), nn.Linear(16, 8))
        with torch.no_grad():
            self.gate[-1].weight.mul_(1e-3)
            self.gate[-1].bias.zero_()
        self.features = None

    def forward(self, payload):
        if self.layer == 2:
            self.context.statistics = edge_statistics(payload.edge_index.detach().cpu(), payload.node.shape[0])
        degree, expansion = self.context.statistics
        raw = degree if self.guidance == "degree" else expansion[self.layer - 2]
        self.features = raw.log1p().to(payload.node.device).unsqueeze(-1)
        if self.layer == 3:
            self.context.statistics = None
        return payload

    def modulate(self, module, inputs, output):
        features, self.features = self.features, None
        gate = 0.1 * self.gate(features).tanh().reshape(output.shape[0], 8, 1)
        projected = self.projection(output).reshape(output.shape[0], 8, 24)
        return output + (gate * projected).reshape_as(output)


def apply_addon(model, config):
    owner = getattr(model, "_molgap_substrate", None)
    if owner is None:
        owner = ModelSubstrate(model, "neural_atom_k1")
    context = TopologyContext()
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(config["seed"])
        for layer in (2, 3):
            addon = ExpansionGate(context, layer, config["guidance"])
            owner.attach(f"ec_gate_{layer}", addon,
                         SlotBinding("neural_atom_k1", SlotName.PRE_MESSAGE, layer))
            model.local_blocks[layer - 1].conv.register_message_forward_hook(addon.modulate)
    return model
