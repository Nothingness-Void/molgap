"""Channel-wise local residual gains; retain K1 modules and checkpoint keys."""
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn.norm import BatchNorm

from .model_substrate import ModelSubstrate


class LocalResidualGainBlock(nn.Module):
    """The original LocalGPSBlock operations with two unit-initialized gains."""

    def __init__(self, base):
        super().__init__()
        self.dropout = base.dropout
        self.conv = base.conv
        self.mlp = base.mlp
        self.norm1 = base.norm1
        self.norm3 = base.norm3
        self.norm_with_batch = base.norm_with_batch
        self.gamma1 = nn.Parameter(torch.ones_like(base.norm1.module.weight))
        self.gamma2 = nn.Parameter(torch.ones_like(base.norm3.module.weight))
        self.train(base.training)

    def _normalize(self, normalizer, value, batch):
        if self.norm_with_batch:
            return normalizer(value, batch=batch)
        return normalizer(value)

    def forward(self, x, edge_index, batch, *, edge_attr):
        local = self.conv(x, edge_index, edge_attr=edge_attr)
        local = F.dropout(local, p=self.dropout, training=self.training)
        output = self._normalize(self.norm1, self.gamma1 * local + x, batch)
        output = output + self.gamma2 * self.mlp(output)
        return self._normalize(self.norm3, output, batch)


def apply_addon(model, config):
    """Keep every existing parameter/buffer object; add 384 scalars per block."""
    if config != {}:
        raise ValueError("k1_local_residual_gain accepts an empty config only")
    blocks = getattr(model, "local_blocks", None)
    if not isinstance(blocks, nn.ModuleList) or len(blocks) != 9:
        raise ValueError("Expected nine K1 local blocks")
    for block in blocks:
        if (isinstance(block, LocalResidualGainBlock)
                or getattr(block, "norm_with_batch", None) is not False
                or not isinstance(getattr(block, "conv", None), nn.Module)
                or not isinstance(getattr(block, "mlp", None), nn.Module)
                or any(type(getattr(block, name, None)) is not BatchNorm
                       for name in ("norm1", "norm3"))
                or any(type(norm.module) is not nn.BatchNorm1d
                       or norm.module.num_features != 192 or not norm.module.affine
                       for norm in (block.norm1, block.norm3))):
            raise ValueError("Expected original K1 192-channel post-BN local blocks")
    owner = getattr(model, "_molgap_substrate", None)
    if owner is None:
        owner = ModelSubstrate(model, "neural_atom_k1")
    if owner.family != "neural_atom_k1":
        raise ValueError("Residual gain requires a K1 substrate")
    for index, block in enumerate(blocks):
        owner.replace_submodule(f"local_residual_gain_{index}",
                                f"local_blocks.{index}", LocalResidualGainBlock(block))
    return model
