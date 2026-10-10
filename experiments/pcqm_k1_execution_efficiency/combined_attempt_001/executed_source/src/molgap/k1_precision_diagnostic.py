"""Explicit precision-contract diagnostics; never change family defaults."""
from contextlib import contextmanager, ExitStack
import math
import statistics
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
    expected_shape = reference["source_idx"].shape
    for name in ("source_idx", "target_eV", "prediction_eV"):
        a, b = reference[name], candidate[name]
        if a.shape != b.shape or a.shape != expected_shape or a.ndim != 1 or not a.numel():
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


def qualify_precision_report(report, *, repeats=4):
    """Reject missing/failed cases; TF32 need not be exact to be measured."""
    expected = {"baseline", "non_blocking", "no_redundant_sync", "transfer_and_sync",
                "cpu_layout", "tf32", "tf32_layout"}
    if report.get("status") != "complete" or set(report.get("cases", {})) != expected:
        raise ValueError("Incomplete precision case inventory")
    for name, row in report["cases"].items():
        if row.get("status") not in {"qualified", "numerical_mismatch"} or not isinstance(row.get("equivalence"), dict):
            raise ValueError(f"Missing/failed numerical comparison: {name}")
        timings = row.get("timings", [])
        if len(timings) != repeats or any(not math.isfinite(t.get("seconds_per_step", float("nan")))
                                        or t["seconds_per_step"] <= 0 for t in timings):
            raise ValueError(f"Missing/invalid timing windows: {name}")
        if name not in {"tf32", "tf32_layout"} and not row["equivalence"].get("exact"):
            raise ValueError(f"Nonexact unchanged-FP32 control: {name}")
    base, tf32 = report["cases"]["cpu_layout"], report["cases"]["tf32_layout"]
    return dict(complete_cases=True, exact_fp32_controls=True, precision_contract_change=True,
        precision_isolated_comparison="tf32_layout versus cpu_layout; both extra contexts without explicit barrier",
        tf32_incremental_time_reduction=1-tf32["median_seconds_per_step"]/base["median_seconds_per_step"],
        paired_tf32_incremental_savings=[1-t["seconds_per_step"]/b["seconds_per_step"]
                                        for b,t in zip(base["timings"],tf32["timings"])],
        strict_fp32_admission=False, generalization_accuracy_qualified=False)


def probe_k1_precision(factory, batches, mean, std, *, deadline, warmup=3, steps=12, repeats=4):
    """Reuse the strict scratch probe; classify TF32 as a separate contract."""
    from .k1_execution_efficiency import probe_k1_execution_efficiency
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8:
        raise ValueError("TF32 requires an Ampere-or-later CUDA device; T4 is unsupported")
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


def qualify_combined_report(report):
    """Matched joint cost contrasts; optimizer/precision quality stays unknown."""
    controls = {"baseline", "non_blocking", "no_redundant_sync", "transfer_and_sync", "cpu_layout"}
    variants = {"fused_layout", "fused_tf32_layout"}
    if (report.get("status") != "complete" or set(report.get("cases", {})) != controls | variants
            or report.get("equivalence_updates") != 3 or report.get("contract_promotion") is not False):
        raise ValueError("Incomplete joint diagnostic")
    medians = {}
    for name, row in report["cases"].items():
        values = [t.get("seconds_per_step", float("nan")) for t in row.get("timings", [])]
        if (row.get("status") not in {"qualified", "numerical_mismatch"} or len(values) != 4
                or any(not math.isfinite(v) or v <= 0 for v in values)):
            raise ValueError("Missing/failed joint timing window")
        if name in controls and (not row.get("equivalence", {}).get("exact") or row.get("optimizer_contract_change")):
            raise ValueError("Unchanged FP32 control failed")
        if name in variants and (row.get("optimizer_contract_change") is not True
                or row.get("execution_only_admissible") is not False
                or row.get("declared_optimizer_flags") != {"foreach": False, "fused": True}):
            raise ValueError("Optimizer change must remain explicitly non-admissible")
        medians[name] = statistics.median(values)
    contrasts = {}
    for name, reference, candidate in (
            ("fused_increment", "cpu_layout", "fused_layout"),
            ("tf32_on_fused_increment", "fused_layout", "fused_tf32_layout"),
            ("combined_total_not_single_mechanism", "baseline", "fused_tf32_layout")):
        ref, actual = report["cases"][reference], report["cases"][candidate]
        contrasts[name] = dict(reference=reference, candidate=candidate,
            median_time_reduction=1-medians[candidate]/medians[reference],
            paired_window_reductions=[1-a["seconds_per_step"]/b["seconds_per_step"]
                                     for b,a in zip(ref["timings"],actual["timings"])])
    return dict(contrasts=contrasts, median_seconds_per_step=medians,
                strict_fp32_admission=False, generalization_accuracy_qualified=False,
                full_training_saving_qualified=False)


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
