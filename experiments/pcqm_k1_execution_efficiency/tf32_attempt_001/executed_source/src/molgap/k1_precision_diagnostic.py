"""Explicit precision-contract diagnostics; never change family defaults."""
from contextlib import contextmanager, ExitStack
import math
import time

import torch

from .k1_execution_layout import cpu_layout_context
from .training_reproducibility import capture_rng_state, restore_rng_state


@contextmanager
def matmul_tf32(enabled):
    """Temporarily allow TF32 matmuls, retaining dtype, cuDNN and RNG policies.

    Allowing TF32 does not prove Tensor Core kernel dispatch. Restores settings
    on all exits. No autocast, BF16 decomposition or cuDNN precision change.
    """
    if type(enabled) is not bool:
        raise ValueError("TF32 permission must be an explicit boolean")
    precision = torch.get_float32_matmul_precision()
    matmul, cudnn = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    try:
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = enabled
        yield
    finally:
        torch.set_float32_matmul_precision(precision)
        torch.backends.cuda.matmul.allow_tf32 = matmul
        torch.backends.cudnn.allow_tf32 = cudnn


def precision_case(*, tf32=False, layout=False):
    @contextmanager
    def context(model, batch, device):
        with ExitStack() as stack:
            stack.enter_context(matmul_tf32(tf32))
            if layout:
                stack.enter_context(cpu_layout_context(model, batch, device))
            yield
    return context


def prediction_drift(reference, candidate):
    """Aligned absolute prediction perturbation, not independent quality evidence."""
    for name in ("source_idx", "target_eV", "prediction_eV"):
        a, b = reference[name], candidate[name]
        if a.shape != b.shape or a.ndim != 1 or not a.numel():
            raise ValueError("Aligned nonempty vector predictions required")
        if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
            raise ValueError("Nonfinite prediction artifact")
    if (not torch.equal(reference["source_idx"], candidate["source_idx"])
            or not torch.equal(reference["target_eV"], candidate["target_eV"])):
        raise ValueError("Prediction members/targets differ")
    if reference["source_idx"].unique().numel() != reference["source_idx"].numel():
        raise ValueError("Duplicated prediction members")
    difference = (candidate["prediction_eV"] - reference["prediction_eV"]).double()
    absolute = difference.abs()
    a = (reference["prediction_eV"].double() - reference["target_eV"].double()).abs().mean()
    b = (candidate["prediction_eV"].double() - candidate["target_eV"].double()).abs().mean()
    return dict(rows=difference.numel(), exact=bool(torch.equal(reference["prediction_eV"], candidate["prediction_eV"])),
        mean_abs_prediction_delta_eV=float(absolute.mean()), p50_abs_prediction_delta_eV=float(absolute.quantile(.5)),
        p95_abs_prediction_delta_eV=float(absolute.quantile(.95)), max_abs_prediction_delta_eV=float(absolute.max()),
        signed_mean_prediction_delta_eV=float(difference.mean()), reference_train_sample_mae_eV=float(a),
        candidate_train_sample_mae_eV=float(b), train_sample_mae_delta_eV=float(b-a),
        generalization_accuracy_qualified=False, precision_contract_change=True)


def probe_k1_precision(factory, batches, mean, std, *, deadline, warmup=3, steps=12, repeats=4):
    """Reuse the strict scratch probe; classify TF32 as a separate contract."""
    from .k1_execution_efficiency import probe_k1_execution_efficiency
    report = probe_k1_execution_efficiency(factory, batches, mean, std, device="cuda",
        wall_seconds=deadline-time.perf_counter(), warmup=warmup, steps=steps, repeats=repeats,
        equivalence_steps=3, atol=1e-6, rtol=1e-5,
        extra_cases={"cpu_layout": cpu_layout_context, "tf32": precision_case(tf32=True),
                     "tf32_layout": precision_case(tf32=True, layout=True)})
    for name, row in report["cases"].items():
        row["precision_contract_change"] = name in {"tf32", "tf32_layout"}
        if row["precision_contract_change"]:
            row.update(execution_only_admissible=False, speed_candidate=False,
                       disposition="Separate precision contract; no strict-FP32 admission")
    report["limits"].append("TF32 permission is not proof of Tensor Core dispatch or final training MAE")
    return report


def profile_k1_step(factory, batch, mean, std, *, tf32, layout, deadline):
    """Bounded attribution, separated from throughput; retain actual kernel names."""
    from .pcqm_500k_v4_evidence import optimizer_for, step
    from .pcqm_gptrans_v4 import _forward
    import torch.nn.functional as F

    if not math.isfinite(deadline):
        raise ValueError("Finite deadline required")
    def guard():
        if time.perf_counter() >= deadline:
            raise TimeoutError("Precision profiler budget exhausted")
    guard()
    rng = capture_rng_state()
    model = factory().cuda().train()
    optimizer = optimizer_for(model)
    rows = []
    try:
        # Warm initialization is excluded from synchronized phase measurements.
        warm_batch = batch.clone()
        with precision_case(tf32=tf32, layout=layout)(model, warm_batch, "cuda"):
            step(model, optimizer, warm_batch.cuda(), mean, std)
        for _ in range(3):
            guard()
            cpu_batch = batch.clone()
            torch.cuda.synchronize()
            stamps = [time.perf_counter()]
            with precision_case(tf32=tf32, layout=layout)(model, cpu_batch, "cuda"):
                data = cpu_batch.cuda()
                torch.cuda.synchronize(); stamps.append(time.perf_counter())
                optimizer.zero_grad(set_to_none=True)
                prediction = _forward(model, data)
                loss = F.l1_loss(prediction.view(-1), (data.y.view(-1)-mean)/std)
                if not bool(torch.isfinite(loss)):
                    raise ValueError("Nonfinite profile loss")
                torch.cuda.synchronize(); stamps.append(time.perf_counter())
                loss.backward()
                torch.cuda.synchronize(); stamps.append(time.perf_counter())
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                torch.cuda.synchronize(); stamps.append(time.perf_counter())
                optimizer.step()
                torch.cuda.synchronize(); stamps.append(time.perf_counter())
            rows.append(dict(zip(("layout_h2d", "forward_loss", "backward", "clip", "optimizer"),
                                 (b-a for a,b in zip(stamps, stamps[1:])))))
        guard()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA], record_shapes=True) as profiler:
            for _ in range(2):
                guard()
                data = batch.clone()
                with precision_case(tf32=tf32, layout=layout)(model, data, "cuda"):
                    with torch.profiler.record_function("K1_owner_step"):
                        step(model, optimizer, data.cuda(), mean, std)
                torch.cuda.synchronize()
                profiler.step()
        operators = [dict(name=e.key, calls=e.count, cpu_self_us=e.self_cpu_time_total,
                          device_self_us=e.self_device_time_total) for e in profiler.key_averages()]
        kernels = sorted({e.name for e in profiler.events() if e.device_type == torch.autograd.DeviceType.CUDA})
        return dict(synchronized_phase_samples=rows, operators=sorted(operators, key=lambda e:e["device_self_us"], reverse=True),
                    kernel_names=kernels, tf32_allowed=tf32, cpu_layout=layout,
                    throughput_measurement=False, tensor_core_dispatch="inspect retained kernel names; permission alone is insufficient")
    finally:
        restore_rng_state(rng)
