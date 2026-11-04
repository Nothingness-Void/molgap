"""Compare joint versus separate normalization of a third-pair-gated return.

Aggregation equations follow server 2fae3b6d:gptrans_triplet_communication;
the experimental change is the update boundary, not its trainer or readout.
"""
import torch
from torch import nn

from .gptrans_variants import normalize_pair
from .model_substrate import ModelSubstrate


LAYERS = (3, 6, 9, 12)


class TripletReturn(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(32)
        for direction in ("in", "out"):
            setattr(self, direction + "_value", nn.Linear(32, 16))
            setattr(self, direction + "_bias", nn.Linear(32, 2))
            setattr(self, direction + "_gate", nn.Linear(32, 2))
        self.output = nn.Linear(32, 32)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, pair, padding):
        nodes = ~padding[:, 0, 0, :]
        valid = nodes[:, :, None] & nodes[:, None, :]
        values = self.norm(pair.permute(0, 2, 3, 1)).masked_fill(~valid[..., None], 0)
        returns = []
        for direction in ("in", "out"):
            value = getattr(self, direction + "_value")(values).unflatten(-1, (2, 8))
            value = value.masked_fill(~valid[..., None, None], 0)
            bias = getattr(self, direction + "_bias")(values)
            gate = getattr(self, direction + "_gate")(values)
            weights = bias.masked_fill(~valid[..., None], torch.finfo(bias.dtype).min)
            weights = weights.softmax(dim=2 if direction == "in" else 1)
            weights = weights * gate.sigmoid() * valid[..., None]
            returned = (torch.einsum("bikh,bjkhd->bijhd", weights, value) if direction == "in"
                        else torch.einsum("bkih,bkjhd->bijhd", weights, value))
            returns.append(returned.flatten(-2))
        # Retain the virtual row/column and [0,0] route to the frozen readout.
        return self.output(torch.cat(returns, -1)).masked_fill(~valid[..., None], 0).permute(0, 3, 1, 2)


class TripletNormBlock(nn.Module):
    def __init__(self, core, order):
        super().__init__()
        self.core, self.order = core, order
        self.triplet = TripletReturn()

    def forward(self, node, pair, padding):
        core = self.core
        node_update, update = core.attention(core.node_norm1(node), pair, padding)
        returned = self.triplet(pair, padding)
        update = (normalize_pair(update + returned) if self.order == "joint"
                  else normalize_pair(update) + returned)
        pair = pair + update
        node = node + core.drop_path(core.attention_scale * node_update)
        node = node + core.drop_path(core.ffn_scale * core.ffn(core.node_norm2(node)))
        return node, pair


def apply_addon(model, config):
    """The shared loader supplies the frozen BondLocal+PairNorm parent first."""
    order = config["order"]
    if order not in {"joint", "separate"}:
        raise ValueError("Unknown triplet normalization order")
    owner = getattr(model, "_molgap_substrate", None) or ModelSubstrate(model, "gptrans_t")
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(42)
        for depth in LAYERS:
            path = f"blocks.{depth - 1}.core"
            core = model.get_submodule(path)
            if getattr(core, "variant", None) != "pair_update_norm":
                raise ValueError("Triplet normalization requires the declared frozen PairNorm parent")
            owner.replace_submodule(f"triplet_norm_{depth}", path, TripletNormBlock(core, order))
    return model


def mount(recipe, model):
    return apply_addon(model, {"order": recipe.mode.removeprefix("triplet_norm_")})


def policy(mode):
    from .family_training_policy import FamilyModePolicy
    from .gptrans_bond_pair_norm import construct_family_model
    return FamilyModePolicy(construct=construct_family_model, mount=mount,
                            objective_name="degree_bond_local_ema999")
