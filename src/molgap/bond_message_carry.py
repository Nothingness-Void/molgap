"""A stateless-in-batches research primitive, not a registered model/runner.

The owning forward must keep one directed-edge order and start with previous=None
for every new batch. No molecular encoder, optimizer or platform operation lives
here. A zero coefficient recovers the parent's transient message exactly.
"""
from __future__ import annotations

import torch
from torch import nn


class BondMessageCarry(nn.Module):
    def __init__(self, layers: int = 12, channels: int = 64):
        super().__init__()
        if type(layers) is not int or layers < 1 or type(channels) is not int or channels < 1:
            raise ValueError("Positive integer layers/channels required")
        self.layers, self.channels = layers, channels
        # No coefficient for layer1: multiplying its absent predecessor would
        # create a permanently disconnected parameter.
        self.logits = nn.Parameter(torch.zeros(layers - 1))

    def forward(self, messages, previous, *, depth: int):
        if type(depth) is not int or not 1 <= depth <= self.layers:
            raise ValueError("Depth is a one-based layer index")
        if messages.ndim != 2 or messages.shape[1] != self.channels or not messages.is_floating_point():
            raise ValueError("Expected floating directed-edge-by-channel messages")
        if depth == 1:
            if previous is not None:
                raise ValueError("Every batch must start without retained edge memory")
            return messages
        if (previous is None or previous.shape != messages.shape
                or previous.device != messages.device or previous.dtype != messages.dtype):
            raise ValueError("Predecessor must retain the same directed-edge layout/device/dtype")
        if self.logits.device != messages.device or self.logits.dtype != messages.dtype:
            raise ValueError("Move the primitive with its parent; never cast inside forward")
        return messages + self.logits[depth - 2].tanh() * previous
