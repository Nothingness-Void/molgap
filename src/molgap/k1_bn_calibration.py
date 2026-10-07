"""Frozen K1 BatchNorm-buffer calibration; no optimizer or parameter updates."""
import time
from contextlib import contextmanager
from .k1_screen_training import FORBIDDEN_MODEL_FIELDS, _forward


@contextmanager
def recalibrated_batch_norm(model, loader, *, source_idx, source_bounds,
                            device: str, deadline: float,
                            dropout_enabled: bool = False, passes_per_batch: int = 1):
    """Temporarily fit only BN buffers on explicitly bound training members.

    Optional K1 dropout includes LocalGPSBlock functional dropout and attention
    dropout, as well as Dropout children. Cumulative minibatch
    estimates follow PyTorch's update_bn convention; they are not exact
    node-weighted population moments. Restore original buffers even on failure.
    """
    import torch
    from .v4_runtime import state_dict_sha256

    if type(dropout_enabled) is not bool or type(passes_per_batch) is not int or passes_per_batch < 1:
        raise ValueError("Expected boolean dropout and positive integer passes_per_batch")
    if model.training or any(module.training for module in model.modules()):
        raise ValueError("BN recalibration requires clean eval mode")
    if any(parameter.requires_grad or parameter.grad is not None for parameter in model.parameters()):
        raise ValueError("BN recalibration requires frozen gradient-free parameters")
    expected = torch.as_tensor(source_idx, dtype=torch.long).view(-1).cpu()
    lower, upper = source_bounds
    if len(expected) == 0 or len(expected.unique()) != len(expected) or \
            not bool(((expected >= lower) & (expected < upper)).all()):
        raise ValueError("Invalid training-only calibration membership")
    bn = {name: module for name, module in model.named_modules()
          if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)}
    if not bn or any(not module.track_running_stats for module in bn.values()):
        raise ValueError("Expected tracked BatchNorm buffers")
    model.to(device)
    original = {name: value.detach().clone() for name, value in model.named_buffers()}
    bn_keys = {f"{name}.{key}" if name else key for name, module in bn.items()
               for key, _ in module.named_buffers(recurse=False)}
    momentum = {name: module.momentum for name, module in bn.items()}
    modes = {module: module.training for module in model.modules()}
    parameters = state_dict_sha256(dict(model.named_parameters()))
    tick, cpu_tick = time.perf_counter(), time.process_time()
    cursor, batches = 0, 0
    report = {}
    try:
        model.eval()
        for module in bn.values():
            module.reset_running_stats()
            module.momentum = None
            module.train()
        if dropout_enabled:
            for module in model.modules():
                if isinstance(module, (torch.nn.modules.dropout._DropoutNd, torch.nn.MultiheadAttention)) or \
                        (module.__class__.__name__ == "LocalGPSBlock" and isinstance(getattr(module, "dropout", None), float)):
                    # Assign directly: train() recursively enables unrelated children.
                    module.training = True
        with torch.no_grad():
            for batch in loader:
                if time.perf_counter() >= deadline:
                    raise TimeoutError("BN recalibration wall budget exhausted")
                observed = batch.source_idx.view(-1).long().cpu()
                if not torch.equal(observed, expected[cursor:cursor + len(observed)]):
                    raise ValueError("Calibration row order or membership differs")
                if any(field in batch for field in FORBIDDEN_MODEL_FIELDS):
                    raise ValueError("Geometry reached pure-2D BN recalibration")
                batch = batch.to(device)
                for _ in range(passes_per_batch):
                    if time.perf_counter() >= deadline:
                        raise TimeoutError("BN recalibration wall budget exhausted")
                    prediction = _forward(model, batch)
                    if not torch.isfinite(prediction).all():
                        raise ValueError("Nonfinite calibration output")
                cursor += len(observed)
                batches += passes_per_batch
        if cursor != len(expected):
            raise ValueError("Incomplete calibration membership")
        model.eval()
        current = dict(model.named_buffers())
        if any(parameter.requires_grad or parameter.grad is not None for parameter in model.parameters()) or \
                parameters != state_dict_sha256(dict(model.named_parameters())) or any(
                not torch.equal(current[key], value) for key, value in original.items() if key not in bn_keys):
            raise ValueError("Calibration changed parameters or non-BN buffers")
        if any(not torch.isfinite(current[key]).all() for key in bn_keys) or any(
                int(module.num_batches_tracked) != batches for module in bn.values()):
            raise ValueError("Invalid calibrated BatchNorm state")
        report.update({"rows": cursor, "batches": batches, "bn_modules": len(bn),
            "num_batches_tracked": {name: int(module.num_batches_tracked) for name, module in bn.items()},
            "wall_seconds": time.perf_counter() - tick,
            "process_cpu_seconds": time.process_time() - cpu_tick,
            "parameters_unchanged": True, "non_bn_buffers_unchanged": True,
            "dropout_disabled": not dropout_enabled, "passes_per_batch": passes_per_batch,
            "labels_used_for_calibration": False,
            "buffer_sha256_before": state_dict_sha256(original),
            "buffer_sha256_calibrated": state_dict_sha256(current),
            "bn_changes": {name: {
                "running_mean_rms_change": float((module.running_mean - original[f"{name}.running_mean" if name else "running_mean"]).square().mean().sqrt()),
                "running_var_mean_before": float(original[f"{name}.running_var" if name else "running_var"].mean()),
                "running_var_mean_after": float(module.running_var.mean())} for name, module in bn.items()}})
        yield report
    finally:
        with torch.no_grad():
            for name, value in model.named_buffers():
                value.copy_(original[name])
        for name, module in bn.items():
            module.momentum = momentum[name]
        for module, training in modes.items():
            module.training = training
        report["buffers_restored"] = state_dict_sha256(dict(model.named_buffers())) == state_dict_sha256(original)
