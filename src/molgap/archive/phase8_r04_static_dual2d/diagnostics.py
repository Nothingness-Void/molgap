"""Router-collapse and expert-use diagnostics."""

from __future__ import annotations

import torch


def router_diagnostics(weights: torch.Tensor) -> dict:
    """Summarize weights shaped [molecules, targets, experts]."""
    value = weights.detach().float().cpu().clamp_min(1e-12)
    entropy = -(value * value.log()).sum(dim=-1)
    effective = entropy.exp()
    quantiles = torch.quantile(value, value.new_tensor([0.1, 0.5, 0.9]), dim=0)
    return {
        "mean_weight": value.mean(dim=0).tolist(),
        "median_weight": quantiles[1].tolist(),
        "p10_weight": quantiles[0].tolist(),
        "p90_weight": quantiles[2].tolist(),
        "mean_entropy": entropy.mean(dim=0).tolist(),
        "mean_effective_experts": effective.mean(dim=0).tolist(),
        "collapsed": bool((effective.mean(dim=0) < 1.3).any()),
    }
