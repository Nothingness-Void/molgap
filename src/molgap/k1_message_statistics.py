"""Compare mean and dispersion of K1 gated messages without replacing their sum."""
from dataclasses import replace
import math

import torch
from torch import nn

from .k1_execution_aggregation import AggregationPlan
from .model_substrate import ModelSubstrate, SlotBinding, SlotName


class MessageStatistic(nn.Module):
    def __init__(self, statistic):
        super().__init__()
        self.statistic = statistic
        self.project = nn.Sequential(nn.Linear(192, 64), nn.SiLU(), nn.Linear(64, 192, bias=False))
        with torch.no_grad():
            self.project[-1].weight.mul_(1e-3)
        self.messages = None

    def capture(self, module, inputs, output):
        self.messages = output

    def forward(self, payload):
        messages, self.messages = self.messages, None
        index = payload.target_index
        rows = payload.aggregated.shape[0]
        cpu_index = index.detach().cpu()
        counts = torch.bincount(cpu_index, minlength=rows).to(messages.device).clamp_min(1).unsqueeze(-1)
        mean = payload.aggregated / counts
        statistic = mean
        if self.statistic == "std":
            plan = AggregationPlan.from_cpu(cpu_index, rows, messages.device)
            variance = plan.reduce((messages - mean[index]).square(), backend="segmented") / counts
            statistic = (variance.clamp_min(0) + 1e-6).sqrt() - math.sqrt(1e-6)
        return replace(payload, aggregated=payload.aggregated + self.project(statistic))


def apply_addon(model, config):
    owner = getattr(model, "_molgap_substrate", None)
    if owner is None:
        owner = ModelSubstrate(model, "neural_atom_k1")
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(config["seed"])
        addon = MessageStatistic(config["statistic"])
    owner.attach("message_statistic", addon, SlotBinding("neural_atom_k1", SlotName.POST_AGGREGATION, 6))
    model.local_blocks[5].conv.register_message_forward_hook(addon.capture)
    return model
