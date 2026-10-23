"""Batch-renorm adaptation for the eighteen local K1 post-residual norms.

Correction uses the pre-update running variance and detached batch moments.
Running variance keeps PyTorch BN's unbiased EMA convention; this is not the
original paper's exact standard-deviation EMA. The existing integer BN counter
is the correction clock under the frozen single-forward training contract.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .model_substrate import ModelSubstrate


class BatchRenorm1d(nn.BatchNorm1d):
    """BN-compatible tensor state with frozen, non-learned correction bounds."""

    def __init__(self, num_features, *, correct_shift=1, warmup_updates=3905,
                 ramp_updates=3905, r_max=3.0, d_max=5.0, eps=1e-5,
                 momentum=0.1, affine=True, track_running_stats=True,
                 device=None, dtype=None):
        if type(num_features) is not int or num_features < 1:
            raise ValueError("num_features must be a positive integer")
        if type(eps) not in (int, float) or not math.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be finite and positive")
        if momentum is not None and (type(momentum) not in (int, float)
                or not math.isfinite(momentum) or not 0 <= momentum <= 1):
            raise ValueError("momentum must be None or finite in [0, 1]")
        if track_running_stats is not True:
            raise ValueError("Batch renorm requires tracked running statistics")
        if type(correct_shift) is not int or correct_shift not in (0, 1):
            raise ValueError("correct_shift must be integer zero or one")
        if type(warmup_updates) is not int or warmup_updates < 0:
            raise ValueError("warmup_updates must be a nonnegative integer")
        if type(ramp_updates) is not int or ramp_updates < 1:
            raise ValueError("ramp_updates must be a positive integer")
        if (type(r_max) not in (int, float) or not math.isfinite(r_max) or r_max < 1
                or type(d_max) not in (int, float) or not math.isfinite(d_max) or d_max < 0):
            raise ValueError("Invalid finite batch-renorm correction bounds")
        super().__init__(num_features, eps=eps, momentum=momentum, affine=affine,
                         track_running_stats=True, device=device, dtype=dtype)
        self.correct_shift = correct_shift
        self.warmup_updates, self.ramp_updates = warmup_updates, ramp_updates
        self.r_max, self.d_max = float(r_max), float(d_max)

    def forward(self, value):
        if (not torch.is_tensor(value) or value.ndim != 2
                or value.shape[1] != self.num_features or value.shape[0] < 1
                or not value.is_floating_point()):
            raise ValueError("Batch renorm requires a nonempty floating (nodes, channels) tensor")
        # The shared kernel owns finite-check cadence; do not scan every norm.
        if not self.training or self.momentum is None:
            # Train-only calibration follows native cumulative BN semantics.
            return super().forward(value)
        if value.shape[0] < 2:
            raise ValueError("Training batch renorm requires at least two nodes")
        with torch.no_grad():
            fraction = ((self.num_batches_tracked.to(dtype=value.dtype) + 1
                         - self.warmup_updates) / self.ramp_updates).clamp(0, 1)
            limit_r = 1 + fraction * (self.r_max - 1)
            limit_d = fraction * self.d_max * self.correct_shift
            batch_variance, batch_mean = torch.var_mean(value.detach(), dim=0, unbiased=False)
            running_std = (self.running_var + self.eps).sqrt()
            correction_r = ((batch_variance + self.eps).sqrt() / running_std).clamp(
                min=limit_r.reciprocal(), max=limit_r)
            correction_d = ((batch_mean - self.running_mean) / running_std).clamp(
                min=-limit_d, max=limit_d)
            self.num_batches_tracked.add_(1)
        normalized = F.batch_norm(value, self.running_mean, self.running_var,
                                  None, None, True, self.momentum, self.eps)
        output = normalized * correction_r + correction_d
        if self.affine:
            output = output * self.weight + self.bias
        return output


def apply_addon(model, config):
    """Transport the initialized local BN state before optimizer/EMA binding."""
    from .experiment_catalog import ADDONS
    from .common.spec import validate_addon_config

    validate_addon_config(ADDONS[("k1_batch_renorm", "1")], config,
                          name="k1_batch_renorm", family_version="3")
    blocks = getattr(model, "local_blocks", None)
    if not isinstance(blocks, nn.ModuleList) or len(blocks) != 9:
        raise ValueError("Batch renorm requires the nine-block K1 backbone")
    replacements = []
    from torch_geometric.nn.norm import BatchNorm

    for index, block in enumerate(blocks):
        for norm_name in ("norm1", "norm3"):
            wrapper = getattr(block, norm_name, None)
            if type(wrapper) is not BatchNorm or wrapper.allow_single_element:
                raise ValueError("Expected the original local PyG BatchNorm wrappers")
            original = wrapper.module
            if type(original) is not nn.BatchNorm1d or not original.track_running_stats:
                raise ValueError("Expected exactly eighteen unmodified tracked local BatchNorm1d modules")
            replacement = BatchRenorm1d(original.num_features, **config,
                eps=original.eps, momentum=original.momentum, affine=original.affine,
                device=original.running_mean.device, dtype=original.running_mean.dtype)
            replacement.load_state_dict(original.state_dict(), strict=True)
            for name, parameter in replacement.named_parameters():
                parameter.requires_grad_(getattr(original, name).requires_grad)
            replacements.append((f"brn_{index}_{norm_name}",
                                 f"local_blocks.{index}.{norm_name}.module", replacement))
    owner = getattr(model, "_molgap_substrate", None)
    if owner is None:
        owner = ModelSubstrate(model, "neural_atom_k1")
    if owner.family != "neural_atom_k1":
        raise ValueError("Batch renorm requires the K1 substrate")
    for name, path, replacement in replacements:
        owner.replace_submodule(name, path, replacement)
    return model
