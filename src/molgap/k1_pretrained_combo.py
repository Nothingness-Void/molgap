"""Frozen pretrained K1 two-pass objectives, with optional mean-output teacher."""
from __future__ import annotations

MODES = ("pretrained_consistency", "pretrained_consistency_teacher")
ADDON_MODES = {"k1_pretrained_consistency": MODES[0], "k1_pretrained_consistency_teacher": MODES[1]}


def validate_config(config, mode):
    from .experiment_spec import _digest
    fields = {"initialization_sha256", "pretrained_source_sha256", "head_reset_sha256",
              "consistency_weight", "teacher_cache"}
    if mode not in MODES or type(config) is not dict or set(config) != fields:
        raise ValueError("Expected exact pretrained K1 objective configuration")
    for key in ("initialization_sha256", "pretrained_source_sha256", "head_reset_sha256"):
        _digest(config[key], key)
    if type(config["consistency_weight"]) not in (int, float) or config["consistency_weight"] != 0.1:
        raise ValueError("Pretrained K1 consistency weight must be 0.1")
    if mode == MODES[0]:
        if config["teacher_cache"] is not None:
            raise ValueError("Arm A must not consume teacher predictions")
    else:
        from .k1_teacher_cache import validate_teacher_config
        validate_teacher_config(config["teacher_cache"], "distill_strong")
    return dict(config)


def objective_name(mode):
    if mode not in MODES:
        raise ValueError("Unknown pretrained K1 mode")
    return "normalized-gap-two-pass-consistency" + ("-teacher-mean-mse" if mode == MODES[1] else "")


def objective_identity(mode, config):
    from .screen_policy import canonical_fingerprint
    return canonical_fingerprint({"name": objective_name(mode), "config": validate_config(config, mode),
                                  "teacher_location": "mean-two-dropout-predictions"})


def objective(first, second, target, *, mode, teacher=None):
    import torch
    import torch.nn.functional as F
    if mode not in MODES or first.shape != second.shape or first.shape != target.shape:
        raise ValueError("Invalid two-pass K1 objective shapes/mode")
    supervised = 0.5 * (F.l1_loss(first, target) + F.l1_loss(second, target))
    disagreement = (first - second).square().mean()
    teacher_mse = None
    if mode == MODES[1]:
        if teacher is None or teacher.shape != target.shape:
            raise ValueError("Arm B requires aligned normalized teacher")
        teacher_mse = F.mse_loss((first + second) / 2, teacher.detach())
    elif teacher is not None:
        raise ValueError("Arm A must not consume teacher")
    loss = supervised + 0.1 * disagreement
    if teacher_mse is not None:
        loss = loss + teacher_mse
    if not bool(torch.isfinite(loss)):
        raise ValueError("Nonfinite pretrained K1 objective")
    return loss, {"supervised_l1": supervised, "disagreement": disagreement,
                  "teacher_mse": teacher_mse, "combined_loss": loss}


def optimizer_step(model, optimizer, batch, mean, std, *, mode):
    import torch
    from .k1_screen_training import _forward
    optimizer.zero_grad(set_to_none=True)
    target = (batch.y.view(-1) - mean) / std
    first, second = _forward(model, batch), _forward(model, batch)
    teacher = (batch.teacher_eV.view(-1).detach() - mean) / std if mode == MODES[1] else None
    loss, components = objective(first, second, target, mode=mode, teacher=teacher)
    loss.backward()
    if not any(p.grad is not None for p in model.parameters()):
        raise ValueError("Missing pretrained K1 gradients")
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    model._combo_components = {k: None if v is None else v.detach() for k, v in components.items()}
    absolute = 0.5 * ((first.detach() - target).abs() + (second.detach() - target).abs()).sum()
    return loss.detach(), absolute, int(target.numel())


def qualify_signal(model, batch):
    import torch
    from .k1_screen_training import _forward
    model.train()
    first, second = _forward(model, batch), _forward(model, batch)
    disagreement = (first - second).square().mean()
    gradients = torch.autograd.grad(disagreement, tuple(model.parameters()), allow_unused=True)
    norm = sum(float(g.detach().square().sum()) for g in gradients if g is not None)
    value = float(disagreement.detach())
    if not (0 < value < float("inf") and 0 < norm < float("inf")):
        raise ValueError("Missing finite nonzero dropout disagreement/gradient")
    return {"accepted": True, "disagreement": value, "gradient_squared_norm": norm}


def validate_initial_state(state, config):
    from .v4_runtime import state_dict_sha256
    if state_dict_sha256(state) != config["initialization_sha256"]:
        raise ValueError("Pretrained K1 tensor identity mismatch")
    head = {k: v for k, v in state.items() if k.startswith("head.")}
    if not head or state_dict_sha256(head) != config["head_reset_sha256"]:
        raise ValueError("Pretrained K1 reset-head tensor identity mismatch")
