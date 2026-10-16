"""Local K1 pre-norm variants; shared family owners retain the training loop."""
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F


class AdaRMSNorm(nn.Module):
    """Input-conditioned RMS gain, including the identity as alpha=1,beta=0."""
    def __init__(self, channels=192, eps=1e-6):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(channels))
        self.beta = nn.Parameter(torch.ones(channels))
        self.eps = eps

    def forward(self, x):
        denominator = (x.square().mean(-1, keepdim=True) + self.eps).rsqrt()
        gain = ((self.alpha * x + self.beta).square().mean(-1, keepdim=True) + self.eps).sqrt()
        return x * denominator * gain


class PreNormLocalBlock(nn.Module):
    def __init__(self, base, *, adaptive):
        super().__init__()
        self.conv, self.mlp = base.conv, base.mlp
        self.dropout = base.dropout
        self.norm1 = AdaRMSNorm() if adaptive else base.norm1
        self.norm3 = AdaRMSNorm() if adaptive else base.norm3
        self.norm_with_batch = False if adaptive else base.norm_with_batch

    def _normalize(self, norm, x, batch):
        return norm(x, batch=batch) if self.norm_with_batch else norm(x)

    def forward(self, x, edge_index, batch, *, edge_attr):
        branch = self.conv(self._normalize(self.norm1, x, batch), edge_index, edge_attr=edge_attr)
        hidden = x + F.dropout(branch, p=self.dropout, training=self.training)
        return hidden + self.mlp(self._normalize(self.norm3, hidden, batch))


def _final_node_norm(module, args, output):
    return module.output_norm(output)


def configure_model(model, mode):
    if mode not in {"prenorm_bn", "prenorm_ada"}:
        raise ValueError("Unknown K1 pre-norm variant")
    adaptive = mode == "prenorm_ada"
    final_norm = AdaRMSNorm() if adaptive else deepcopy(model.local_blocks[-1].norm3)
    model.local_blocks = nn.ModuleList(PreNormLocalBlock(block, adaptive=adaptive)
                                      for block in model.local_blocks)
    # Normalize nodes after the last mixer, before graph pooling. Keep mixer
    # attributes intact for the existing accelerated layout provider.
    last = model.neural_atom_mixers["9"]
    last.add_module("output_norm", final_norm)
    last.register_forward_hook(_final_node_norm)
    return model


def make_model(mode):
    from .qm9_neural_atom import make_encoder
    return configure_model(make_encoder("neural_atom_k1"), mode)


def transform_family_model(recipe, model):
    return configure_model(model, recipe.mode)


def policy(mode):
    from .family_training_policy import FamilyModePolicy
    if mode not in {"prenorm_bn", "prenorm_ada"}:
        raise ValueError("Unknown K1 pre-norm policy")
    return FamilyModePolicy(transform_model=transform_family_model,
                           extra_development_weights=("ema",), objective_name="reference")
