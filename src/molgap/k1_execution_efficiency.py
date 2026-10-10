"""Bounded scratch execution probes, not a trainer or promotion authority.

Factories/callbacks are trusted caller code. Inputs are already ordered CPU
batches; no loader, data access, checkpoint publication or runtime reconfiguration
is provided. CUDA sync removal assumes exclusively current-stream work: blocking
loss extraction already waits for that stream. Other-stream work is unsupported.
"""
from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
import math
import statistics
import time

import numpy as np
import torch

from .training_reproducibility import capture_rng_state, restore_rng_state


def _snapshot(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: _snapshot(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_snapshot(v) for v in value)
    return deepcopy(value)


def _json_state(value):
    if torch.is_tensor(value) or isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: _json_state(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_state(v) for v in value]
    return value


def compare_states(reference, candidate, *, atol=0.0, rtol=0.0):
    """Full recursive comparison with explicit key/dtype/shape and error reports."""
    failures, tolerance_failures = [], []
    maximum = 0.0

    def walk(a, b, path):
        nonlocal maximum
        exact = close = False
        if torch.is_tensor(a) and torch.is_tensor(b):
            if a.dtype == b.dtype and a.shape == b.shape:
                exact = torch.equal(a, b)
                if a.is_floating_point() or a.is_complex():
                    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
                    error = float((a - b).abs().max()) if a.numel() and finite else 0.0
                    maximum = max(maximum, error)
                    close = finite and torch.allclose(a, b, atol=atol, rtol=rtol)
                    exact = exact and finite
                else:
                    close = exact
        elif isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
            exact = close = a.dtype == b.dtype and np.array_equal(a, b)
        elif isinstance(a, dict) and isinstance(b, dict):
            if a.keys() != b.keys():
                failures.append(path + ".keys")
                tolerance_failures.append(path + ".keys")
            for key in a.keys() & b.keys():
                walk(a[key], b[key], f"{path}.{key}")
            return
        elif isinstance(a, (list, tuple)) and type(a) is type(b):
            if len(a) != len(b):
                failures.append(path + ".length")
                tolerance_failures.append(path + ".length")
            for i, (x, y) in enumerate(zip(a, b)):
                walk(x, y, f"{path}.{i}")
            return
        elif type(a) is type(b):
            exact = close = a == b
            if isinstance(a, float):
                exact = exact and math.isfinite(a) and math.isfinite(b)
                close = exact or (math.isfinite(a) and math.isfinite(b)
                                  and math.isclose(a, b, abs_tol=atol, rel_tol=rtol))
                if math.isfinite(a) and math.isfinite(b):
                    maximum = max(maximum, abs(a - b))
        if not exact:
            failures.append(path)
        if not close:
            tolerance_failures.append(path)

    walk(reference, candidate, "state")
    return {"exact": not failures, "within_tolerance": not tolerance_failures,
            "atol": atol, "rtol": rtol, "max_abs_error": maximum,
            "differences": failures, "tolerance_failures": tolerance_failures}


def _finite_output(_module, _inputs, output, *, expected_predictions=None):
    if expected_predictions is not None:
        prediction = output[0] if isinstance(output, (tuple, list)) and output else output
        if (not torch.is_tensor(prediction) or prediction.numel() != expected_predictions
                or prediction.dtype != torch.float32):
            raise RuntimeError("Forward prediction must contain exactly 128 FP32 values")
    def check(value):
        if torch.is_tensor(value) and not bool(torch.isfinite(value).all()):
            raise RuntimeError("Non-finite forward output")
        if isinstance(value, dict):
            for child in value.values():
                check(child)
        elif isinstance(value, (tuple, list)):
            for child in value:
                check(child)
    check(output)


def probe_execution_efficiency(
    model_factory, ordered_batches, mean, std, *, optimizer_factory, step_fn,
    device="cpu", warmup=2, steps=3, repeats=2, wall_seconds=60.0,
    optimizer_cases=(), extra_cases=None, atol=0.0, rtol=0.0,
    real_run=False, single_stream=True, equivalence_steps=3,
    return_checkpoint_snapshots=False,
):
    """Probe independent scratch states and return timing plus numerical gates.

    ``model_factory()`` creates a fresh module; ``optimizer_factory(model)``
    creates the owner's optimizer; ``step_fn(model,opt,batch,mean,std)`` performs
    the unchanged owner step and returns a detached scalar loss. Extra cases map
    names to ``batch_context_factory(model,cpu_batch,device)->contextmanager``.
    Context entry/exit and H2D are measured. No context is installed by default.
    Equivalence uses only ``equivalence_steps`` updates, without timing warmup.
    Optional final checkpoint snapshots contain CPU tensors (not JSON values).
    ``real_run=True`` enforces FP32/BS128 and existing strict determinism settings;
    generic CPU infrastructure fixtures may use smaller batches.

    Bounds are cooperative between operations, not an interruptible hard timeout.
    Equivalence has full host snapshots outside timing; timing omits comparison
    hooks but retains finite guards, float(loss), and capture_rng_state per step.
    """
    for name, value, minimum, maximum in (("warmup", warmup, 0, 64),
            ("steps", steps, 2, 64), ("repeats", repeats, 2, 16),
            ("equivalence_steps", equivalence_steps, 2, 8)):
        if type(value) is not int or not minimum <= value <= maximum:
            raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    if not math.isfinite(wall_seconds) or not 0 < wall_seconds <= 3600:
        raise ValueError("wall_seconds must be in (0, 3600]")
    if repeats % 2:
        raise ValueError("repeats must be even for balanced timing order")
    if not all(math.isfinite(v) for v in (mean, std, atol, rtol)) or std <= 0 or min(atol, rtol) < 0:
        raise ValueError("Invalid normalization or tolerances")
    if not single_stream:
        raise ValueError("Only current-stream execution is supported")
    batch_count = max(warmup + steps, equivalence_steps)
    if not isinstance(ordered_batches, (list, tuple)) or len(ordered_batches) < batch_count:
        raise ValueError("Supply enough ordered CPU batches for timing and equivalence")
    device = torch.device(device)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Only CPU and CUDA are supported")
    def validate_cpu(value):
        if torch.is_tensor(value):
            if value.device.type != "cpu":
                raise ValueError("Retained inputs must be CPU tensors")
            if value.is_floating_point() and value.dtype != torch.float32:
                raise ValueError("Floating inputs must be FP32")
        elif isinstance(value, dict):
            for child in value.values():
                validate_cpu(child)
        elif isinstance(value, (tuple, list)):
            for child in value:
                validate_cpu(child)
    for batch in ordered_batches[:batch_count]:
        validate_cpu(batch.to_dict() if hasattr(batch, "to_dict") else vars(batch))
    if torch.is_autocast_enabled(device.type):
        raise ValueError("Autocast must be disabled")
    if real_run:
        if not torch.are_deterministic_algorithms_enabled() or torch.is_deterministic_algorithms_warn_only_enabled():
            raise ValueError("Strict deterministic algorithms must already be configured")
        if (torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32
                or torch.backends.cudnn.benchmark or not torch.backends.cudnn.deterministic
                or torch.get_float32_matmul_precision() != "highest"):
            raise ValueError("Strict FP32/no-TF32 settings must already be configured")
        if any(getattr(b, "num_graphs", None) != 128 for b in ordered_batches[:batch_count]):
            raise ValueError("Real-run API requires physical BS128")
    extra_cases = dict(extra_cases or {})
    reserved = {"baseline", "non_blocking", "no_redundant_sync", "transfer_and_sync", "foreach", "fused"}
    if reserved & extra_cases.keys() or any(not isinstance(k, str) or not k for k in extra_cases):
        raise ValueError("Extra case names must be nonempty and not reserved")
    if set(optimizer_cases) - {"foreach", "fused"} or len(set(optimizer_cases)) != len(optimizer_cases):
        raise ValueError("Optimizer cases must be unique foreach/fused flags")
    cases = ["baseline", "non_blocking", "no_redundant_sync", "transfer_and_sync", *optimizer_cases, *extra_cases]
    started = time.perf_counter()
    deadline = started + wall_seconds
    entry_rng = capture_rng_state()
    report = {"status": "complete", "cases": {case: {"status": "not_run",
              "optimizer_contract_change": case in optimizer_cases, "timings": [],
              "execution_only_admissible": False, "speed_candidate": False} for case in cases},
              "device": str(device), "timing_order": [], "equivalence_updates": equivalence_steps,
              "compared": ["loss", "full_gradients_before_clip", "full_gradients_after_clip",
                           "parameters_and_buffers", "optimizer_tensors", "rng", "checkpoint_keys"],
              "compile": {"status": "deferred", "reason": "No budgeted subprocess/dropout qualification"},
              "contract_promotion": False, "real_run": real_run,
              "limits": ["Cooperative wall bound", "Current stream only",
                         "Guard/loss scalar synchronization remains; speedup may be negligible",
                         "No loader/full-epoch or scientific quality qualification",
                         "First scratch step is not process/device cold-start evidence"]}

    def guard():
        if time.perf_counter() >= deadline:
            raise TimeoutError("Cooperative wall budget exhausted")

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def execute(case, *, evidence):
        guard()
        restore_rng_state(entry_rng)
        setup_start = time.perf_counter()
        model = model_factory().to(device).train()
        if any(t.is_floating_point() and t.dtype != torch.float32 for t in model.state_dict().values()):
            raise ValueError("Model must be FP32")
        model.load_state_dict(deepcopy(initial_state), strict=True)
        optimizer = optimizer_factory(model)
        if case in optimizer_cases:
            if not isinstance(optimizer, torch.optim.AdamW):
                raise ValueError("foreach/fused probe requires owner's AdamW")
            for group in optimizer.param_groups:
                group.update(foreach=case == "foreach", fused=case == "fused")
            optimizer.defaults.update(foreach=case == "foreach", fused=case == "fused")
        restore_rng_state(step_rng)
        sync()
        setup_seconds = time.perf_counter() - setup_start
        def forward_guard(module, inputs, output):
            _finite_output(module, inputs, output,
                           expected_predictions=128 if real_run else None)
        handles = [model.register_forward_hook(forward_guard)]
        raw, clipped = {}, {}
        if evidence:
            def save_raw(name):
                def hook(gradient):
                    raw[name] = _snapshot(gradient)
                return hook
            handles.extend(p.register_hook(save_raw(n)) for n, p in model.named_parameters() if p.requires_grad)
            def pre_step(_opt, _args, _kwargs):
                clipped.update({n: _snapshot(p.grad) for n, p in model.named_parameters()})
            handles.append(optimizer.register_step_pre_hook(pre_step))
        snapshots, durations = [], []
        try:
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            sync()
            interval_start = time.perf_counter()
            warm_seconds = 0.0
            run_warmup = 0 if evidence else warmup
            run_steps = equivalence_steps if evidence else steps
            for i in range(run_warmup + run_steps):
                guard()
                raw.clear()
                if evidence:
                    raw.update({n: None for n, _ in model.named_parameters()})
                clipped.clear()
                if i == run_warmup:
                    sync()
                    interval_start = time.perf_counter()
                tick = time.perf_counter()
                cpu_batch = deepcopy(ordered_batches[i])
                context = extra_cases[case](model, cpu_batch, device) if case in extra_cases else nullcontext()
                with context:
                    batch = cpu_batch.to(device, non_blocking=case in {"non_blocking", "transfer_and_sync"})
                    loss = step_fn(model, optimizer, batch, mean, std)
                if not bool(torch.isfinite(loss).all()):
                    raise RuntimeError("Non-finite loss")
                if case in {"baseline", "non_blocking"}:
                    sync()
                scalar = float(loss)
                rng = capture_rng_state()
                durations.append(time.perf_counter() - tick)
                if i < run_warmup:
                    warm_seconds += durations[-1]
                if evidence:
                    snapshots.append(_snapshot({"loss": scalar, "gradients_before_clip": raw,
                        "gradients_after_clip": clipped, "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(), "rng": rng,
                        "checkpoint_keys": ("model", "optimizer", "rng")}))
            sync()
            measured = time.perf_counter() - interval_start
            return snapshots, {"seconds": measured, "seconds_per_step": measured / run_steps,
                "first_step_seconds": durations[0], "warmup_seconds": warm_seconds,
                "setup_seconds": setup_seconds, "warm_start": "fresh scratch state each repeat",
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None}
        finally:
            for handle in handles:
                handle.remove()

    try:
        guard()
        seed_model = model_factory()
        initial_state = _snapshot(seed_model.state_dict())
        del seed_model
        step_rng = capture_rng_state()
        baseline, _ = execute("baseline", evidence=True)
        for case in cases:
            guard()
            row = report["cases"][case]
            try:
                snapshots = baseline if case == "baseline" else execute(case, evidence=True)[0]
                if case in optimizer_cases:
                    # Implementation flags are the declared contract delta, not numerical drift.
                    # Preserve all other hyperparameters and optimizer state in the comparison.
                    for ref, actual in zip(baseline, snapshots):
                        for ref_group, group in zip(ref["optimizer"]["param_groups"],
                                                    actual["optimizer"]["param_groups"]):
                            for flag in ("foreach", "fused"):
                                group[flag] = ref_group[flag]
                    row["declared_optimizer_flags"] = {"foreach": case == "foreach", "fused": case == "fused"}
                row["equivalence"] = compare_states(baseline, snapshots, atol=atol, rtol=rtol)
                if case in optimizer_cases:
                    for actual in snapshots:
                        for group in actual["optimizer"]["param_groups"]:
                            group.update(row["declared_optimizer_flags"])
                last = snapshots[-1]
                row["end_rng"] = _json_state(last["rng"])
                row["checkpoint_structure"] = {
                    "keys": list(last["checkpoint_keys"]),
                    "model_keys": list(last["model"]),
                    "optimizer_keys": list(last["optimizer"]),
                    "optimizer_state_parameter_ids": list(last["optimizer"]["state"]),
                    "gradient_keys": list(last["gradients_after_clip"]),
                    "scope": "Scratch model/optimizer/RNG only; not full trainer checkpoint metadata"}
                if return_checkpoint_snapshots:
                    row["final_checkpoint"] = {k: _snapshot(last[k]) for k in last["checkpoint_keys"]}
                row["execution_only_admissible"] = row["equivalence"]["exact"] and case not in optimizer_cases
                row["status"] = "qualified" if row["equivalence"]["exact"] else "numerical_mismatch"
                del last, snapshots
            except (RuntimeError, ValueError, NotImplementedError) as error:
                row.update(status="failed_or_unsupported", reason=f"{type(error).__name__}: {error}")
        # Alternating forward/reverse orders balance pair positions.
        for repeat in range(repeats):
            order = cases if repeat % 2 == 0 else list(reversed(cases))
            report["timing_order"].append(list(order))
            for case in order:
                row = report["cases"][case]
                if row["status"] == "failed_or_unsupported":
                    continue
                try:
                    _, timing = execute(case, evidence=False)
                    row["timings"].append(timing)
                except (RuntimeError, ValueError, NotImplementedError) as error:
                    row.update(status="timing_failed", reason=f"{type(error).__name__}: {error}",
                               execution_only_admissible=False)
        baseline_timings = report["cases"]["baseline"]["timings"]
        baseline_seconds = statistics.median(t["seconds_per_step"] for t in baseline_timings) if baseline_timings else None
        for row in report["cases"].values():
            if row["timings"]:
                median = statistics.median(t["seconds_per_step"] for t in row["timings"])
                row.update(median_seconds_per_step=median,
                           speedup=baseline_seconds / median if baseline_seconds else None,
                           min_seconds_per_step=min(t["seconds_per_step"] for t in row["timings"]),
                           max_seconds_per_step=max(t["seconds_per_step"] for t in row["timings"]),
                           speed_candidate=bool(baseline_seconds and row["execution_only_admissible"]
                                              and len(row["timings"]) == repeats and median < baseline_seconds))
                row["paired_repeat_savings_fraction"] = [
                    1 - t["seconds_per_step"] / b["seconds_per_step"]
                    for b, t in zip(baseline_timings, row["timings"])]
                row["speed_acceptance"] = "parent_owned; median gain is descriptive, not a noise-qualified gate"
    except TimeoutError as error:
        report.update(status="budget_exhausted", reason=str(error))
        for row in report["cases"].values():
            row["speed_candidate"] = False
    finally:
        restore_rng_state(entry_rng)
    report["wall_seconds"] = time.perf_counter() - started
    return report


def probe_k1_execution_efficiency(model_factory, ordered_batches, mean, std, **kwargs):
    """Strict real-run adapter; caller supplies the owner-built K1 factory."""
    from .pcqm_500k_v4_evidence import optimizer_for, step
    return probe_execution_efficiency(model_factory, ordered_batches, mean, std,
        optimizer_factory=optimizer_for, step_fn=step, real_run=True, **kwargs)
