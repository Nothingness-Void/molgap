"""Depth-normalized pair updates; no extra tensor, cache, parameter or RNG."""
from __future__ import annotations

import math
import torch
from .gptrans import GPTransBlock


def pair_residual(previous, updated, depth):
    if type(depth) is not int or depth < 1:
        raise ValueError("A positive frozen depth is required")
    return previous + (updated - previous) / math.sqrt(depth)


class DepthScaledPairBlock(GPTransBlock):
    def forward(self, node, pair, key_padding_mask):
        next_node, updated = super().forward(node, pair, key_padding_mask)
        result = pair_residual(pair, updated, self._pair_depth)
        if getattr(self, "_capture_pair_diagnostics", False):
            # One already scheduled train batch/epoch; exclude padded pairs.
            valid = (~key_padding_mask) & (~key_padding_mask.transpose(-1, -2))
            count = valid.sum().clamp_min(1) * pair.shape[1]
            rms = lambda x: ((x.detach().square() * valid).sum() / count).sqrt()
            self._pair_diagnostics = torch.stack([rms(pair), rms(updated - pair), rms(result)])
            self._capture_pair_diagnostics = False
        return next_node, result


def apply_pair_depth_scale(model):
    depth = len(model.blocks)
    if depth != 12:
        raise ValueError("Only the frozen twelve-block GPTrans is qualified")
    for block in model.blocks:
        if type(block) is not GPTransBlock:
            raise ValueError("Pair scaling cannot stack with another block intervention")
        block.__class__ = DepthScaledPairBlock
        block._pair_depth = depth
    return model
