"""Frozen baseline checks from4517a5b0; diagnostic-only, never live dispatch."""
import math


def finite_tree(value, label):
    import torch
    if isinstance(value, torch.Tensor):
        if not torch.isfinite(value).all():
            raise ValueError("Nonfinite checkpoint state: " + label)
    elif isinstance(value, dict):
        for key, child in value.items():
            finite_tree(child, label + "." + str(key))
    elif isinstance(value, (list, tuple)):
        for child in value:
            finite_tree(child, label)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Nonfinite checkpoint state: " + label)


def finite_state_dict(state_dict, *, label):
    import torch
    failures, device_flags = [], {}
    for name, value in state_dict.items():
        if not torch.is_tensor(value):
            continue
        finite = torch.isfinite(value).all()
        if value.device.type == "cpu":
            if not bool(finite):
                failures.append(name)
        else:
            device_flags.setdefault(value.device, []).append((name, finite))
    for flags in device_flags.values():
        observed = torch.stack([flag for _, flag in flags]).cpu().tolist()
        failures.extend(name for (name, _), finite in zip(flags, observed) if not finite)
    if failures:
        failed = set(failures)
        ordered = [name for name in state_dict if name in failed]
        raise RuntimeError(f"{label} contains non-finite tensors: {ordered[:8]}")
