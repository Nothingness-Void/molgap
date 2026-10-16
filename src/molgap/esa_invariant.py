"""ESA-small/2 CPU candidate: permutation-invariant seed readout.

The frozen author reference remains in esa_model.py. This intentionally differs
from upstream: seed queries never receive an index-paired edge-token residual.
MIT-derived attention/MLP components retain their original parameter inventory.
"""
from copy import deepcopy
from dataclasses import dataclass, fields
import torch
from torch import nn

from .esa_model import ESA, ESAConfig


@dataclass(frozen=True)
class ESAInvariantConfig(ESAConfig):
    readout_version: str = "pma-no-input-residual-v2"
    isolated_atom_policy: str = "reject-pending-authorized-census"

    def __post_init__(self):
        super().__post_init__()
        if (self.readout_version != "pma-no-input-residual-v2" or not self.include_self or
                self.isolated_atom_policy != "reject-pending-authorized-census"):
            raise ValueError("ESA/2 requires its explicit safe mask, PMA and unsupported-isolate policy")


class InvariantPMA(nn.Module):
    def __init__(self, block):
        super().__init__()
        # Reuse this model's own initialized components, not another instance
        # or a second random constructor. State names/shapes remain unchanged.
        self.norm = block.norm
        self.norm_mlp = block.norm_mlp
        self.attention = block.attention
        self.mlp = block.mlp
        self.seeds = block.seeds
        self.kind = "P"

    def forward(self, x, allowed=None):
        x = self.norm(x)
        seeds = self.seeds.expand(x.shape[0], -1, -1)
        out = self.attention(seeds, x, allowed)
        # Rows now inhabit query space. Edge-token order is never paired with
        # query indices, including when token count happens to equal seed count.
        return out + self.mlp(self.norm_mlp(out))


class ESAInvariant(ESA):
    def __init__(self, config=ESAInvariantConfig()):
        if type(config) is not ESAInvariantConfig:
            raise TypeError("Frozen ESAInvariantConfig required")
        base = {f.name:getattr(config,f.name) for f in fields(ESAConfig)}
        super().__init__(ESAConfig(**base))
        self.config = config
        self.pool = InvariantPMA(self.pool)

    def graph_forward(self, graph):
        if graph.edge_index.numel() and torch.unique(graph.edge_index).numel() != len(graph.x):
            raise ValueError("ESA/2 isolated atoms unsupported pending an authorized topology census; no members changed")
        return super().graph_forward(graph)


def make_model(arm):
    values = deepcopy(arm.get("model_config", {}))
    if "layer_types" in values:
        values["layer_types"] = tuple(values["layer_types"])
    return ESAInvariant(ESAInvariantConfig(**values))


CONFIG_TYPE = ESAInvariantConfig
