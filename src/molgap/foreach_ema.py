"""Tensor-list EMA primitive, independent of any profiling/training entrypoint."""
from .pcqm_gptrans_v4 import ExponentialMovingAverage


class ForeachExponentialMovingAverage(ExponentialMovingAverage):
    """The same EMA equation issued as tensor-list kernels."""

    def update(self, model) -> None:
        import torch

        floating_targets = []
        floating_sources = []
        for name, value in model.state_dict().items():
            target = self.state[name]
            if value.is_floating_point():
                floating_targets.append(target)
                floating_sources.append(value.detach())
            else:
                target.copy_(value)
        torch._foreach_mul_(floating_targets, self.decay)
        torch._foreach_add_(
            floating_targets, floating_sources, alpha=1.0 - self.decay
        )
