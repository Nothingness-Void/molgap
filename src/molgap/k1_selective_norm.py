"""Post-FFN-only K1 RMS; formula retained from archived 48e4c6a5.

The family owner loads the frozen base before surgery and reconstructs surgery
before strict resume loading. Message-path BN and block forwards are unchanged.
"""
import torch
from torch import nn
from torch_geometric.nn.norm import BatchNorm

from .model_substrate import ModelSubstrate


class NodeRMSNorm(nn.Module):
    """Per-node feature RMS with the frozen BN outer affine and epsilon."""

    def __init__(self, base):
        super().__init__()
        self.eps = base.eps
        self.weight = nn.Parameter(base.weight.detach().clone(), requires_grad=base.weight.requires_grad)
        self.bias = nn.Parameter(base.bias.detach().clone(), requires_grad=base.bias.requires_grad)
        self.train(base.training)

    def forward(self, x, batch=None):
        return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + self.eps) * self.weight + self.bias


def apply_addon(model, config):
    """Replace exactly nine post-FFN norms, after validating all sites."""
    if config != {}:
        raise ValueError("k1_ffn_node_rms accepts an empty config only")
    blocks = getattr(model, "local_blocks", None)
    if not isinstance(blocks, nn.ModuleList) or len(blocks) != 9:
        raise ValueError("Expected nine K1 local blocks")
    for block in blocks:
        if (block.norm_with_batch is not False
                or any(type(getattr(block, name, None)) is not BatchNorm
                       for name in ("norm1", "norm3"))
                or any(type(norm.module) is not nn.BatchNorm1d
                       or norm.module.num_features != 192 or not norm.module.affine
                       for norm in (block.norm1, block.norm3))):
            raise ValueError("Expected K1 post-BN message and FFN sites")
    owner = getattr(model, "_molgap_substrate", None)
    if owner is None:
        owner = ModelSubstrate(model, "neural_atom_k1")
    if owner.family != "neural_atom_k1":
        raise ValueError("Selective norm requires a K1 substrate")
    for index, block in enumerate(blocks):
        replacement = NodeRMSNorm(block.norm3.module).train(block.norm3.training)
        owner.replace_submodule(f"ffn_node_rms_{index}", f"local_blocks.{index}.norm3", replacement)
    return model
