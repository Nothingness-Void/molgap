"""Tensor-list EMA primitive, independent of any profiling/training entrypoint."""


class ExponentialMovingAverage:
    """Retained EMA state ABI without a dependency on a trainer or its constants."""
    def __init__(self, model, decay=0.9999):
        if decay not in (0.9999, 0.999):
            raise ValueError("EMA decay must be a reviewed frozen value")
        self.decay = decay
        self.state = {name: value.detach().clone() for name, value in model.state_dict().items()}

    def state_dict(self):
        return self.state

    def load_state_dict(self, state):
        if state.keys() != self.state.keys():
            raise RuntimeError("EMA state identity changed")
        self.state = {name: value.detach().clone() for name, value in state.items()}


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
