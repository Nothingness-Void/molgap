"""Parameter EMA for K1; live buffers preserve explicit normalization semantics."""
from __future__ import annotations

DECAY = 0.999


def configuration() -> dict:
    return {"decay": DECAY, "start": "initial-state", "update": "after-every-step",
            "buffers": "copy-live", "selection": "best-development-ema-only"}


def make_ema(model):
    import copy
    shadow = copy.deepcopy(model).eval()
    shadow.requires_grad_(False)
    return shadow


def update_ema(shadow, model) -> None:
    import torch
    with torch.no_grad():
        live = dict(model.named_parameters())
        for name, parameter in shadow.named_parameters():
            parameter.mul_(DECAY).add_(live[name].detach(), alpha=1.0 - DECAY)
        buffers = dict(model.named_buffers())
        for name, value in shadow.named_buffers():
            value.copy_(buffers[name].detach())


def qualification(model, batch, mean, std, output) -> dict:
    """Assigned train fixture: EMA update and atomic shadow resume equivalence."""
    import torch
    from .k1_screen_training import _optimizer_step, _forward, LEARNING_RATE, WEIGHT_DECAY
    from .training_reproducibility import atomic_torch_save, capture_rng_state, restore_rng_state
    from .v4_runtime import state_dict_sha256
    shadow = make_ema(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    _optimizer_step(model, optimizer, batch, mean, std)
    update_ema(shadow, model)
    if any(not torch.equal(value, dict(model.named_buffers())[name])
           for name, value in shadow.named_buffers()):
        raise ValueError("EMA buffers were not copied from live")
    if state_dict_sha256(shadow.state_dict()) == state_dict_sha256(model.state_dict()):
        raise ValueError("EMA has no distinct parameter state")
    path = output / "diagnostic_ema_resume.pt"
    atomic_torch_save(path, {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                            "ema": shadow.state_dict(), "rng_state": capture_rng_state()})
    _optimizer_step(model, optimizer, batch, mean, std)
    update_ema(shadow, model)
    continuous = state_dict_sha256(shadow.state_dict())
    saved = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(saved["model"], strict=True)
    shadow.load_state_dict(saved["ema"], strict=True)
    optimizer.load_state_dict(saved["optimizer"])
    restore_rng_state(saved["rng_state"])
    _optimizer_step(model, optimizer, batch, mean, std)
    update_ema(shadow, model)
    if state_dict_sha256(shadow.state_dict()) != continuous:
        raise ValueError("EMA atomic resume diverges")
    selected_path = output / "diagnostic_ema_selected.pt"
    atomic_torch_save(selected_path, shadow.state_dict())
    selected = make_ema(model)
    selected.load_state_dict(torch.load(selected_path, map_location="cpu", weights_only=True), strict=True)
    with torch.no_grad():
        delta = float((_forward(selected, batch) - _forward(shadow, batch)).abs().max())
    if delta != 0.0:
        raise ValueError("Selected EMA clone changes inference")
    return {"accepted": True, "configuration": configuration(), "ema_state_sha256": continuous,
            "ema_resume_exact": True, "selected_ema_roundtrip_delta": delta,
            "formal_sample_presentations": 0, "fixture_role": "training-only"}
