"""Replace only K1 node FFNs with matrix-budget-matched SwiGLU."""
import torch
from torch import nn
from torch.nn import functional as F

from .model_substrate import ModelSubstrate


class K1SwiGLUFFN(nn.Module):
    def __init__(self, hidden_dropout, output_dropout):
        super().__init__()
        # Pack the independent gate/value matrices into one projection.
        self.gate_value = nn.Linear(192, 512)
        self.hidden_dropout = nn.Dropout(hidden_dropout)
        self.output = nn.Linear(256, 192)
        self.output_dropout = nn.Dropout(output_dropout)

    def forward(self, value):
        gate, content = self.gate_value(value).chunk(2, dim=-1)
        hidden = self.hidden_dropout(F.silu(gate) * content)
        return self.output_dropout(self.output(hidden))


def apply_addon(model, config):
    owner = getattr(model, "_molgap_substrate", None)
    if owner is None:
        owner = ModelSubstrate(model, "neural_atom_k1")
    # Mount after frozen backbone loading, without advancing the training RNG.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(config["seed"])
        for index, block in enumerate(model.local_blocks):
            original = block.mlp
            owner.replace_submodule(
                f"swiglu_ffn_{index}", f"local_blocks.{index}.mlp",
                K1SwiGLUFFN(original[2].p, original[4].p),
            )
    return model
