"""Two-pass K1 regression objectives; clean inference stays unchanged."""
from __future__ import annotations

MODES = ("dropout_mean2", "dropout_consistency2")


def configuration(mode: str) -> dict:
    if mode not in MODES:
        raise ValueError("Unknown two-pass K1 objective")
    return {"passes": 2, "consistency_weight": 0.1 if mode == MODES[1] else 0.0,
            "space": "normalized-gap", "disagreement": "squared-output"}


def objective_identity(mode: str) -> str:
    from .screen_policy import canonical_fingerprint
    return canonical_fingerprint({"supervised": "mean-two-independent-dropout-L1",
                                  **configuration(mode)})


def objective(first, second, target, *, mode: str):
    import torch
    import torch.nn.functional as functional
    if first.shape != second.shape or first.shape != target.shape:
        raise ValueError("Two-pass regression shapes differ")
    weight = configuration(mode)["consistency_weight"]
    supervised = 0.5 * (functional.l1_loss(first, target) + functional.l1_loss(second, target))
    disagreement = (first - second).square().mean()
    loss = supervised + weight * disagreement
    if not bool(torch.isfinite(loss)):
        raise ValueError("Nonfinite two-pass objective")
    return loss, disagreement


def optimizer_step(model, optimizer, batch, mean, std, *, mode: str):
    import torch
    from .k1_screen_training import _forward
    optimizer.zero_grad(set_to_none=True)
    target = (batch.y.view(-1) - mean) / std
    first, second = _forward(model, batch), _forward(model, batch)
    loss, _ = objective(first, second, target, mode=mode)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    absolute = 0.5 * ((first.detach() - target).abs() + (second.detach() - target).abs()).sum()
    return loss.detach(), absolute, int(target.numel())


def qualify_signal(model, batch, *, mode: str) -> dict:
    """Bounded train-fixture falsifier, before the all-arm training barrier."""
    import torch
    from .k1_screen_training import _forward
    model.train()
    first, second = _forward(model, batch), _forward(model, batch)
    disagreement = (first - second).square().mean()
    gradients = torch.autograd.grad(disagreement, tuple(model.parameters()), allow_unused=True)
    norm = sum(float(value.detach().square().sum()) for value in gradients if value is not None)
    value = float(disagreement.detach())
    if not (0 < value < float("inf") and 0 < norm < float("inf")):
        raise ValueError("No finite nonzero dropout disagreement/gradient; stop before training")
    return {"accepted": True, "normalized_disagreement": value,
            "disagreement_gradient_squared_norm": norm, "fixture_role": "training-only",
            "formal_sample_presentations": 0, "config": configuration(mode)}
