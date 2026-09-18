"""Train-only V5 execution profile for the frozen GPTrans-T reference."""
from __future__ import annotations

import gc
import hashlib
import json
import math
import random
import time
from pathlib import Path

from .pcqm_gptrans_prenorm_500k import (
    BATCH_SIZE,
    PARAMETERS,
    SEED,
    make_model,
    optimizer_for,
)
from .pcqm_gptrans_v4 import EMA_DECAY, ExponentialMovingAverage, _forward, _state_sha256
from .pcqm_k1_pair_token_batch_profile import (
    _cohorts,
    _graph_sizes,
    _load_train_only,
    _loader,
    _normalizer,
)
from .pcqm_k1_scale import FIXED_500K_MANIFEST_SHA256
from .pcqm_k1_scale_runner import find_cache
from .training_reproducibility import (
    atomic_json,
    atomic_torch_save,
    build_runtime_manifest,
    configure_fp32_determinism,
    sha256_file,
)


FORMAT = "molgap-gptrans-step-profile-v1"
COHORTS = ("representative", "graph_size_tail")
VARIANTS = (
    "baseline",
    "sparse_finite_checks",
    "foreach_adamw",
    "foreach_ema",
    "combined",
)
WARMUP_STEPS = 2
MEASURED_STEPS = 8
REPEATS = 3
FINITE_CHECK_INTERVAL = 50
MAX_EQUIVALENCE_DELTA = 1.0e-7
MIN_REPRESENTATIVE_SPEEDUP = 1.05
MIN_TAIL_SPEED_RATIO = 0.98
TRAIN_ROWS_PER_EPOCH = (500_000 // BATCH_SIZE) * BATCH_SIZE
DEVELOPMENT_PROXY_ROWS = 50_000


def _hash_indices(indices: list[int]) -> str:
    return hashlib.sha256(
        ",".join(str(int(value)) for value in indices).encode("ascii")
    ).hexdigest()


def _cpu_state(state: dict) -> dict:
    return {
        name: value.detach().cpu().clone() if hasattr(value, "detach") else value
        for name, value in state.items()
    }


def _flatten_optimizer_state(optimizer) -> tuple[dict[str, object], dict]:
    import torch

    state = optimizer.state_dict()
    tensors: dict[str, object] = {}
    # ``foreach`` selects an execution backend rather than optimizer math. Keep
    # all mathematical fields in the comparison while normalizing this one
    # intentional implementation difference.
    param_groups = []
    for group in state["param_groups"]:
        normalized = dict(group)
        normalized.pop("foreach", None)
        param_groups.append(normalized)
    scalars: dict[str, object] = {"param_groups": param_groups}
    for parameter_id, values in sorted(state["state"].items()):
        for name, value in sorted(values.items()):
            key = f"state.{parameter_id}.{name}"
            if isinstance(value, torch.Tensor):
                tensors[key] = value.detach().cpu().clone()
            else:
                scalars[key] = value
    return tensors, scalars


def _state_digest(state: dict[str, object]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode("utf-8") + b"\0")
        if hasattr(value, "detach"):
            tensor = value.detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode("ascii") + b"\0")
            digest.update(tensor.numpy().tobytes())
        else:
            digest.update(json.dumps(value, sort_keys=True).encode("utf-8"))
    return digest.hexdigest()


def _max_state_delta(left: dict, right: dict) -> tuple[float, str]:
    maximum, maximum_name = 0.0, ""
    if left.keys() != right.keys():
        raise RuntimeError("Execution variants changed state keys")
    for name in left:
        a, b = left[name], right[name]
        if a.shape != b.shape or a.dtype != b.dtype:
            raise RuntimeError(f"State tensor identity changed: {name}")
        if not a.is_floating_point():
            if not bool(a.equal(b)):
                raise RuntimeError(f"Integer state changed: {name}")
            continue
        delta = float((a - b).abs().max())
        if delta > maximum:
            maximum, maximum_name = delta, name
    return maximum, maximum_name


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
        torch._foreach_mul_(floating_targets, EMA_DECAY)
        torch._foreach_add_(
            floating_targets, floating_sources, alpha=1.0 - EMA_DECAY
        )


def _variant_flags(variant: str) -> dict[str, bool]:
    if variant not in VARIANTS:
        raise ValueError(variant)
    return {
        "sparse_checks": variant in ("sparse_finite_checks", "combined"),
        "foreach_optimizer": variant in ("foreach_adamw", "combined"),
        "foreach_ema": variant in ("foreach_ema", "combined"),
    }


def _make_state(variant: str):
    import torch

    flags = _variant_flags(variant)
    configure_fp32_determinism(SEED)
    model = make_model("reference").to("cuda").train()
    if flags["foreach_optimizer"]:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=2.5e-4, weight_decay=0.05, foreach=True
        )
    else:
        optimizer = optimizer_for(model)
    ema_type = ForeachExponentialMovingAverage if flags["foreach_ema"] else ExponentialMovingAverage
    return model, optimizer, ema_type(model), flags


def _step(
    model,
    optimizer,
    ema,
    batch,
    mean,
    std,
    *,
    step_index: int,
    sparse_checks: bool,
):
    import torch

    optimizer.zero_grad(set_to_none=True)
    prediction = _forward(model, batch)
    target = (batch.y.view(-1) - mean) / std
    loss = torch.nn.functional.l1_loss(prediction, target)
    check = not sparse_checks or step_index % FINITE_CHECK_INTERVAL == 0
    if check and not bool(torch.isfinite(loss)):
        raise RuntimeError("Non-finite profiling loss")
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), 1.0, error_if_nonfinite=check
    )
    if check and not bool(torch.isfinite(gradient_norm)):
        raise RuntimeError("Non-finite profiling gradient norm")
    optimizer.step()
    ema.update(model)
    return loss.detach()


def _run_variant(batches, *, variant: str, mean, std) -> tuple[dict, dict]:
    import torch

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        model, optimizer, ema, flags = _make_state(variant)
    except (RuntimeError, TypeError) as caught:
        if variant == "baseline":
            raise
        return {
            "variant": variant,
            "status": "unsupported",
            "error": str(caught).splitlines()[0][:500],
            "initial_state_sha256": None,
            "final_state_sha256": None,
            "final_ema_sha256": None,
            "final_optimizer_tensor_sha256": None,
            "losses": [],
            "measured_graphs": 0,
            "h2d_seconds": 0.0,
            "optimizer_step_seconds": 0.0,
            "end_to_end_seconds": 0.0,
            "graphs_per_second": None,
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
            "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
        }, {}
    initial_sha256 = _state_sha256(model)
    losses = []
    h2d_seconds = 0.0
    step_seconds = 0.0
    measured_graphs = 0
    status, error = "complete", None
    try:
        for step_index, cpu_batch in enumerate(batches):
            started = time.perf_counter()
            batch = cpu_batch.clone().to("cuda", non_blocking=True)
            torch.cuda.synchronize()
            transfer = time.perf_counter() - started
            started = time.perf_counter()
            loss = _step(
                model,
                optimizer,
                ema,
                batch,
                mean,
                std,
                step_index=step_index,
                sparse_checks=flags["sparse_checks"],
            )
            torch.cuda.synchronize()
            compute = time.perf_counter() - started
            losses.append(float(loss))
            if step_index >= WARMUP_STEPS:
                h2d_seconds += transfer
                step_seconds += compute
                measured_graphs += int(batch.num_graphs)
            del batch, loss
    except (RuntimeError, TypeError) as caught:
        if variant == "baseline":
            raise
        status = "unsupported" if variant != "baseline" else "failed"
        error = str(caught).splitlines()[0][:500]
        torch.cuda.synchronize()

    model_state = _cpu_state(model.state_dict())
    ema_state = _cpu_state(ema.state_dict())
    optimizer_tensors, optimizer_scalars = _flatten_optimizer_state(optimizer)
    total_seconds = h2d_seconds + step_seconds
    result = {
        "variant": variant,
        "status": status,
        "error": error,
        "initial_state_sha256": initial_sha256,
        "final_state_sha256": _state_digest(model_state),
        "final_ema_sha256": _state_digest(ema_state),
        "final_optimizer_tensor_sha256": _state_digest(optimizer_tensors),
        "losses": losses,
        "measured_graphs": measured_graphs,
        "h2d_seconds": h2d_seconds,
        "optimizer_step_seconds": step_seconds,
        "end_to_end_seconds": total_seconds,
        "graphs_per_second": (
            measured_graphs / total_seconds if total_seconds > 0 else None
        ),
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
    }
    states = {
        "model": model_state,
        "ema": ema_state,
        "optimizer_tensors": optimizer_tensors,
        "optimizer_scalars": optimizer_scalars,
    }
    del optimizer, ema, model
    gc.collect()
    torch.cuda.empty_cache()
    return result, states


def _compare_states(baseline: tuple[dict, dict], candidate: tuple[dict, dict]) -> dict:
    base_result, base_state = baseline
    result, state = candidate
    if result["status"] != "complete":
        return {"passed": False, "reason": result["error"]}
    loss_delta = max(
        abs(left - right)
        for left, right in zip(
            base_result["losses"], result["losses"], strict=True
        )
    )
    model_delta, model_name = _max_state_delta(base_state["model"], state["model"])
    ema_delta, ema_name = _max_state_delta(base_state["ema"], state["ema"])
    optimizer_delta, optimizer_name = _max_state_delta(
        base_state["optimizer_tensors"], state["optimizer_tensors"]
    )
    scalar_equal = base_state["optimizer_scalars"] == state["optimizer_scalars"]
    maximum = max(loss_delta, model_delta, ema_delta, optimizer_delta)
    return {
        "passed": maximum <= MAX_EQUIVALENCE_DELTA and scalar_equal,
        "maximum_allowed_delta": MAX_EQUIVALENCE_DELTA,
        "maximum_loss_delta": loss_delta,
        "maximum_model_delta": model_delta,
        "maximum_model_delta_name": model_name,
        "maximum_ema_delta": ema_delta,
        "maximum_ema_delta_name": ema_name,
        "maximum_optimizer_delta": optimizer_delta,
        "maximum_optimizer_delta_name": optimizer_name,
        "optimizer_scalar_state_equal": scalar_equal,
    }


def _timed_stage(rows: dict[str, list[float]], name: str, function):
    import torch

    torch.cuda.synchronize()
    started = time.perf_counter()
    value = function()
    torch.cuda.synchronize()
    rows.setdefault(name, []).append(time.perf_counter() - started)
    return value


def _detailed_baseline(batches, mean, std) -> dict:
    import torch

    torch.cuda.empty_cache()
    model, optimizer, ema, _ = _make_state("baseline")
    rows: dict[str, list[float]] = {}
    for step_index, cpu_batch in enumerate(batches):
        batch = _timed_stage(
            rows,
            "h2d",
            lambda cpu_batch=cpu_batch: cpu_batch.clone().to(
                "cuda", non_blocking=True
            ),
        )
        _timed_stage(rows, "zero_grad", lambda: optimizer.zero_grad(set_to_none=True))
        prediction = _timed_stage(rows, "forward", lambda: _forward(model, batch))
        loss = _timed_stage(
            rows,
            "target_and_loss",
            lambda: torch.nn.functional.l1_loss(
                prediction, (batch.y.view(-1) - mean) / std
            ),
        )
        finite_loss = _timed_stage(
            rows, "loss_finite_sync", lambda: bool(torch.isfinite(loss))
        )
        if not finite_loss:
            raise RuntimeError("Non-finite detailed-profile loss")
        _timed_stage(rows, "backward", loss.backward)
        gradient_norm = _timed_stage(
            rows,
            "gradient_clip",
            lambda: torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=False
            ),
        )
        finite_gradient = _timed_stage(
            rows,
            "gradient_finite_sync",
            lambda: bool(torch.isfinite(gradient_norm)),
        )
        if not finite_gradient:
            raise RuntimeError("Non-finite detailed-profile gradient")
        _timed_stage(rows, "adamw", optimizer.step)
        _timed_stage(rows, "ema", lambda: ema.update(model))
        if step_index < WARMUP_STEPS:
            for values in rows.values():
                values.clear()
        del batch, prediction, loss, gradient_norm

    totals = {name: sum(values) for name, values in rows.items()}
    total = sum(totals.values())
    stages = {
        name: {
            "total_seconds": value,
            "mean_seconds": value / MEASURED_STEPS,
            "fraction": value / total if total else None,
        }
        for name, value in totals.items()
    }
    result = {
        "measured_steps": MEASURED_STEPS,
        "total_seconds": total,
        "stages": stages,
    }
    del optimizer, ema, model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _fixed_overhead_proxy(batches, mean, std, output_root: Path) -> dict:
    import torch

    model, optimizer, ema, _ = _make_state("baseline")
    for step_index, cpu_batch in enumerate(batches[:WARMUP_STEPS]):
        batch = cpu_batch.clone().to("cuda", non_blocking=True)
        _step(
            model,
            optimizer,
            ema,
            batch,
            mean,
            std,
            step_index=step_index,
            sparse_checks=False,
        )
    torch.cuda.synchronize()

    live = _cpu_state(model.state_dict())
    swap_started = time.perf_counter()
    model.load_state_dict(ema.state_dict(), strict=True)
    torch.cuda.synchronize()
    ema_swap_in_seconds = time.perf_counter() - swap_started
    eval_graphs = 0
    started = time.perf_counter()
    model.eval()
    with torch.no_grad():
        for cpu_batch in batches:
            batch = cpu_batch.clone().to("cuda", non_blocking=True)
            _forward(model, batch)
            eval_graphs += int(batch.num_graphs)
    torch.cuda.synchronize()
    evaluation_seconds = time.perf_counter() - started
    restore_started = time.perf_counter()
    model.load_state_dict(live, strict=True)
    torch.cuda.synchronize()
    ema_swap_out_seconds = time.perf_counter() - restore_started

    temporary = output_root / ".checkpoint_profile.tmp.pt"
    payload = {
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "profiling_only": True,
    }
    started = time.perf_counter()
    atomic_torch_save(temporary, payload)
    checkpoint_sha256 = sha256_file(temporary)
    checkpoint_bytes = temporary.stat().st_size
    checkpoint_seconds = time.perf_counter() - started
    temporary.unlink()
    result = {
        "train_role_only": True,
        "evaluation_proxy_graphs": eval_graphs,
        "evaluation_proxy_seconds": evaluation_seconds,
        "evaluation_proxy_graphs_per_second": eval_graphs / evaluation_seconds,
        "projected_50k_evaluation_seconds": (
            DEVELOPMENT_PROXY_ROWS / eval_graphs * evaluation_seconds
        ),
        "ema_swap_in_seconds": ema_swap_in_seconds,
        "ema_swap_out_seconds": ema_swap_out_seconds,
        "checkpoint_write_hash_seconds": checkpoint_seconds,
        "checkpoint_bytes": checkpoint_bytes,
        "temporary_checkpoint_sha256": checkpoint_sha256,
        "temporary_checkpoint_retained": False,
    }
    del payload, optimizer, ema, model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _load_batches(graphs, indices) -> tuple[list, dict]:
    selected = indices[: (WARMUP_STEPS + MEASURED_STEPS) * BATCH_SIZE]
    loader = _loader(graphs, selected, BATCH_SIZE)
    iterator = iter(loader)
    batches = []
    durations = []
    for _ in range(WARMUP_STEPS + MEASURED_STEPS):
        started = time.perf_counter()
        batches.append(next(iterator))
        durations.append(time.perf_counter() - started)
    del iterator, loader
    measured = durations[WARMUP_STEPS:]
    return batches, {
        "selected_rows": len(selected),
        "indices_sha256": _hash_indices(selected),
        "measured_steps": MEASURED_STEPS,
        "total_seconds": sum(measured),
        "mean_seconds": sum(measured) / len(measured),
    }


def _profile_cohort(graphs, indices, mean, std, cohort: str, output_root: Path) -> dict:
    batches, loader_profile = _load_batches(graphs, indices)
    detailed = _detailed_baseline(batches, mean, std)
    rng = random.Random(SEED + (0 if cohort == "representative" else 1))
    repeat_rows = []
    for repeat in range(REPEATS):
        order = list(VARIANTS)
        rng.shuffle(order)
        outputs = {}
        states = {}
        for variant in order:
            outputs[variant], states[variant] = _run_variant(
                batches, variant=variant, mean=mean, std=std
            )
        baseline = (outputs["baseline"], states["baseline"])
        comparisons = {
            variant: _compare_states(
                baseline, (outputs[variant], states[variant])
            )
            for variant in VARIANTS
            if variant != "baseline"
        }
        speedups = {
            variant: (
                outputs["baseline"]["end_to_end_seconds"]
                / outputs[variant]["end_to_end_seconds"]
                if outputs[variant]["status"] == "complete"
                else None
            )
            for variant in VARIANTS
            if variant != "baseline"
        }
        repeat_rows.append(
            {
                "repeat": repeat,
                "execution_order": order,
                "variants": outputs,
                "equivalence": comparisons,
                "speedup_vs_baseline": speedups,
            }
        )
        del states

    medians = {}
    for variant in VARIANTS:
        throughputs = sorted(
            row["variants"][variant]["graphs_per_second"]
            for row in repeat_rows
            if row["variants"][variant]["status"] == "complete"
        )
        medians[variant] = (
            throughputs[len(throughputs) // 2] if throughputs else None
        )
    baseline_throughput = medians["baseline"]
    median_speedups = {
        variant: (
            medians[variant] / baseline_throughput
            if medians[variant] is not None and baseline_throughput
            else None
        )
        for variant in VARIANTS
        if variant != "baseline"
    }
    all_equivalent = {
        variant: all(
            row["equivalence"][variant].get("passed") is True
            for row in repeat_rows
        )
        for variant in VARIANTS
        if variant != "baseline"
    }
    overhead = (
        _fixed_overhead_proxy(batches, mean, std, output_root)
        if cohort == "representative"
        else None
    )
    return {
        "cohort": cohort,
        "status": "complete",
        "physical_batch_per_device": BATCH_SIZE,
        "loader_collate": loader_profile,
        "synchronized_baseline_decomposition": detailed,
        "fixed_epoch_overhead_proxy": overhead,
        "repeats": repeat_rows,
        "median_graphs_per_second": medians,
        "median_speedup_vs_baseline": median_speedups,
        "all_equivalence_checks_passed": all_equivalent,
    }


def _followup_eligibility(results: list[dict]) -> dict:
    representative = next(row for row in results if row["cohort"] == "representative")
    tail = next(row for row in results if row["cohort"] == "graph_size_tail")
    eligibility = {}
    for variant in VARIANTS[1:]:
        representative_speedup = representative["median_speedup_vs_baseline"][variant]
        tail_speedup = tail["median_speedup_vs_baseline"][variant]
        individual = variant != "combined"
        eligibility[variant] = {
            "eligible": bool(
                individual
                and representative["all_equivalence_checks_passed"][variant]
                and tail["all_equivalence_checks_passed"][variant]
                and representative_speedup is not None
                and representative_speedup >= MIN_REPRESENTATIVE_SPEEDUP
                and tail_speedup is not None
                and tail_speedup >= MIN_TAIL_SPEED_RATIO
            ),
            "individual_variant": individual,
            "representative_speedup": representative_speedup,
            "tail_speedup": tail_speedup,
            "minimum_representative_speedup": MIN_REPRESENTATIVE_SPEEDUP,
            "minimum_tail_speed_ratio": MIN_TAIL_SPEED_RATIO,
        }
    return eligibility


def run(
    output_root: Path,
    *,
    source_commit: str,
    source_archive_sha256: str,
    allocation_seconds: float,
    slurm_job_id: str,
) -> dict:
    import torch

    output_root.mkdir(parents=True, exist_ok=True)
    settings = configure_fp32_determinism(SEED)
    runtime = build_runtime_manifest(settings)
    root, manifest = find_cache(FIXED_500K_MANIFEST_SHA256)
    graphs, shards, shard_records = _load_train_only(root, manifest)
    node_counts, edge_counts = _graph_sizes(shards)
    cohorts = _cohorts(node_counts, edge_counts)
    mean_value, std_value = _normalizer(shards)
    mean = torch.tensor(mean_value, device="cuda")
    std = torch.tensor(std_value, device="cuda")

    results = []
    for cohort in COHORTS:
        row = _profile_cohort(
            graphs, cohorts[cohort], mean, std, cohort, output_root
        )
        results.append(row)
        atomic_json(output_root / f"{cohort}.json", row)
        print(
            f"PROFILE cohort={cohort} baseline_gps="
            f"{row['median_graphs_per_second']['baseline']:.3f}",
            flush=True,
        )

    eligibility = _followup_eligibility(results)
    representative = results[0]
    baseline_gps = representative["median_graphs_per_second"]["baseline"]
    overhead = representative["fixed_epoch_overhead_proxy"]
    projection = {
        "training_rows_per_epoch": TRAIN_ROWS_PER_EPOCH,
        "training_seconds_from_representative_throughput": (
            TRAIN_ROWS_PER_EPOCH / baseline_gps
        ),
        "projected_training_loader_seconds": (
            representative["loader_collate"]["mean_seconds"]
            * (TRAIN_ROWS_PER_EPOCH // BATCH_SIZE)
        ),
        "projected_50k_evaluation_seconds": overhead[
            "projected_50k_evaluation_seconds"
        ],
        "checkpoint_write_hash_seconds": overhead[
            "checkpoint_write_hash_seconds"
        ],
    }
    projection["projected_epoch_seconds"] = (
        projection["training_seconds_from_representative_throughput"]
        + projection["projected_training_loader_seconds"]
        + projection["projected_50k_evaluation_seconds"]
        + projection["checkpoint_write_hash_seconds"]
    )
    summary = {
        "format": FORMAT,
        "status": "complete",
        "profiling_only": True,
        "scientific_result_produced": False,
        "training_checkpoint_produced": False,
        "model_bundle_produced": False,
        "scientific_contract_changed": False,
        "development_role_read": False,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
        "test_challenge_role_read": False,
        "train_role_read": True,
        "source_commit": source_commit,
        "source_archive_sha256": source_archive_sha256,
        "slurm_job_id": slurm_job_id,
        "allocation_seconds": float(allocation_seconds),
        "cache_manifest_sha256": FIXED_500K_MANIFEST_SHA256,
        "train_rows": len(graphs),
        "parameter_count": PARAMETERS,
        "precision": "fp32",
        "physical_batch_per_device": BATCH_SIZE,
        "runtime_manifest": runtime,
        "train_shards": shard_records,
        "node_count_quantiles": {
            str(q): float(torch.quantile(node_counts.float(), q))
            for q in (0.5, 0.9, 0.95, 0.99, 1.0)
        },
        "results": results,
        "projected_epoch_breakdown": projection,
        "exact_implementation_followup_eligibility": eligibility,
        "followup_note": (
            "Eligibility permits a separately reviewed execution-only repair; "
            "it does not alter an accepted scientific result or contract."
        ),
    }
    atomic_json(output_root / "profile_summary.json", summary)
    artifacts = {
        path.name: sha256_file(path)
        for path in sorted(output_root.glob("*.json"))
        if path.name != "completion_manifest.json"
    }
    completion = {**summary, "artifact_sha256": artifacts}
    atomic_json(output_root / "completion_manifest.json", completion)
    return completion
