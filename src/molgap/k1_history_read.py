"""Compare current-layer and cross-layer normalized associative reads in K1."""
from dataclasses import replace

import torch
from torch import nn
from torch.nn import functional as F

from .k1_execution_aggregation import AggregationPlan
from .model_substrate import ModelSubstrate, SlotBinding, SlotName


class AssociativeRead(nn.Module):
    def __init__(self, history):
        super().__init__()
        self.history = history
        self.norm = nn.LayerNorm(192, elementwise_affine=False)
        self.qkv = nn.Linear(192, 192, bias=False)
        self.output = nn.Linear(64, 192, bias=False)
        self.dropout = nn.Dropout(0.05)
        with torch.no_grad():
            self.output.weight.mul_(1e-3)
        self.state = None

    def read(self, payload, layer):
        if layer == 3:
            self.state = None
        q, k, v = self.qkv(self.norm(payload.node)).reshape(-1, 3, 4, 16).unbind(1)
        q, k = F.elu(q) + 1, F.elu(k) + 1
        kv = (k.unsqueeze(-1) * v.unsqueeze(-2)).flatten(1)
        membership = payload.graph_index
        cpu_membership = membership.detach().cpu()
        graphs = int(cpu_membership.max()) + 1
        plan = AggregationPlan.from_cpu(cpu_membership, graphs, payload.node.device)
        state = plan.reduce(torch.cat((kv, k.flatten(1)), dim=-1), backend="segmented")
        if self.history and self.state is not None:
            state = state + self.state
        self.state = state if self.history and layer != 9 else None
        matrix = state[:, :1024].reshape(graphs, 4, 16, 16)[membership]
        mass = state[:, 1024:].reshape(graphs, 4, 16)[membership]
        numerator = torch.einsum("nhr,nhrv->nhv", q, matrix)
        denominator = (q * mass).sum(-1, keepdim=True).clamp_min(1e-6)
        update = self.dropout(self.output((numerator / denominator).reshape(-1, 64)))
        return replace(payload, node=payload.node + update)


class ReadTap(nn.Module):
    def __init__(self, core, layer):
        super().__init__()
        self.core, self.layer = core, layer

    def forward(self, payload):
        return self.core.read(payload, self.layer)


def apply_addon(model, config):
    owner = getattr(model, "_molgap_substrate", None)
    if owner is None:
        owner = ModelSubstrate(model, "neural_atom_k1")
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(config["seed"])
        core = AssociativeRead(config["history"])
    for layer in (3, 6, 9):
        owner.attach(f"associative_read_{layer}", ReadTap(core, layer),
                     SlotBinding("neural_atom_k1", SlotName.PRE_MESSAGE, layer))
    return model
