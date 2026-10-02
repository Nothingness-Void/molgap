"""Frozen training-only FLAG objective for the existing K1 encoder.

Only the OGB atom encoder output is perturbed, before RWSE is added.
Hooks and perturbations are temporary and add no model state or parameters.
"""
from __future__ import annotations

from contextlib import contextmanager


CONFIG = {"passes": 3, "step_size": 0.001,
          "initialization": "uniform-minus-alpha-plus-alpha",
          "ascent": "unbounded-sign", "location": "ogb-atom-embedding-before-rwse",
          "loss": "normalized-gap-l1", "gradient_reduction": "mean"}


def validate_config(config):
    if (type(config) is not dict or config != CONFIG or
            any(type(config[key]) is not type(value) for key, value in CONFIG.items())):
        raise ValueError("FLAG configuration differs from the frozen objective")


@contextmanager
def embedding_perturbation(model, delta):
    """Add an externally owned perturbation for this context only."""
    def add_delta(_module, _inputs, output):
        if output.shape != delta.shape or output.device != delta.device or output.dtype != delta.dtype:
            raise ValueError("FLAG perturbation differs from atom embedding shape/device/dtype")
        return output + delta

    handle = model.node_emb.register_forward_hook(add_delta)
    try:
        yield
    finally:
        handle.remove()


def adversarial_optimizer_step(model, optimizer, batch, mean, std, forward, *, config=CONFIG):
    """Average three adversarial/dropout losses, then clip and update once.

    The forward callable is supplied by the trusted family owner, never a
    configuration import. Returned metrics average perturbed pre-update passes.
    """
    import torch
    import torch.nn.functional as functional

    validate_config(config)
    optimizer.zero_grad(set_to_none=True)
    delta = torch.empty((batch.x.shape[0], 192), device=batch.x.device,
                        dtype=next(model.node_emb.parameters()).dtype)
    delta.uniform_(-config["step_size"], config["step_size"])
    delta.requires_grad_()
    target = (batch.y.view(-1) - mean) / std
    loss_sum = target.new_zeros(())
    absolute_sum = target.new_zeros(())
    for index in range(config["passes"]):
        with embedding_perturbation(model, delta):
            prediction = forward(model, batch)
            loss = functional.l1_loss(prediction, target)
            (loss / config["passes"]).backward()
        if delta.grad is None or not bool(torch.isfinite(delta.grad).all()):
            raise RuntimeError("FLAG perturbation gradient is missing or nonfinite")
        loss_sum += loss.detach() / config["passes"]
        absolute_sum += (prediction.detach() - target).abs().sum() / config["passes"]
        if index + 1 < config["passes"]:
            delta = (delta.detach() + config["step_size"] * delta.grad.detach().sign()).requires_grad_()
    if not bool(torch.isfinite(loss_sum)):
        raise RuntimeError("FLAG training loss is nonfinite")
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    return loss_sum, absolute_sum, int(target.numel())
