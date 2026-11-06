"""Bound K1 SwiGLU inputs with affine-free LayerNorm before multiplication."""
import torch
from torch import nn

from .k1_swiglu_ffn import K1SwiGLUFFN
from .model_substrate import ModelSubstrate


class K1PreNormSwiGLUFFN(K1SwiGLUFFN):
    def __init__(self, hidden_dropout, output_dropout):
        super().__init__(hidden_dropout, output_dropout)
        self.input_norm = nn.LayerNorm(192, eps=1e-5, elementwise_affine=False)

    def forward(self, value):
        return super().forward(self.input_norm(value))


def apply_addon(model, config):
    owner = getattr(model, "_molgap_substrate", None)
    if owner is None:
        owner = ModelSubstrate(model, "neural_atom_k1")
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(config["seed"])
        for index, block in enumerate(model.local_blocks):
            original = block.mlp
            owner.replace_submodule(
                f"swiglu_prenorm_ffn_{index}", f"local_blocks.{index}.mlp",
                K1PreNormSwiGLUFFN(original[2].p, original[4].p),
            )
    return model
