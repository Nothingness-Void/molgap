"""Bitwise replay checks for accelerator reduction kernels.

This module deliberately imports Torch only inside the runtime entry point so
the caller can set deterministic-runtime environment variables first.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import time
from pathlib import Path
from typing import Callable


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _tensor_sha256(tensor) -> str:
    contiguous = tensor.detach().cpu().contiguous()
    return hashlib.sha256(contiguous.numpy().tobytes()).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize_repetitions(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("at least one repetition is required")
    successful = [row for row in rows if row.get("error") is None]
    if not successful:
        return {
            "successful_repetitions": 0,
            "all_forward_bitwise_equal": False,
            "all_gradient_bitwise_equal": False,
            "max_forward_abs_diff": None,
            "max_gradient_abs_diff": None,
            "errors": [row["error"] for row in rows],
        }
    return {
        "successful_repetitions": len(successful),
        "all_forward_bitwise_equal": len({row["forward_sha256"] for row in successful}) == 1,
        "all_gradient_bitwise_equal": len({row["gradient_sha256"] for row in successful}) == 1,
        "max_forward_abs_diff": max(row["forward_max_abs_diff"] for row in successful),
        "max_gradient_abs_diff": max(row["gradient_max_abs_diff"] for row in successful),
        "errors": [row["error"] for row in rows if row.get("error") is not None],
    }


def configure_determinism(strict: bool) -> dict:
    import torch

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = strict
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(strict)
    enabled = (
        bool(torch.are_deterministic_algorithms_enabled())
        if hasattr(torch, "are_deterministic_algorithms_enabled")
        else strict
    )
    return {
        "requested_strict": strict,
        "deterministic_algorithms_enabled": enabled,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "rocblas_default_atomics_mode": os.environ.get(
            "ROCBLAS_DEFAULT_ATOMICS_MODE"
        ),
        "miopen_find_mode": os.environ.get("MIOPEN_FIND_MODE"),
        "cudnn_deterministic": bool(getattr(torch.backends.cudnn, "deterministic", False)),
        "cudnn_benchmark": bool(getattr(torch.backends.cudnn, "benchmark", False)),
    }


def install_sorted_segment_scatter() -> dict:
    """Replace index-based reductions with sorted CSR segment reductions.

    The Xi'an Torch 1.10 PyG path resolves reductions through the mutable
    ``torch_scatter`` module object, so installing this before model imports
    also covers MessagePassing and global pooling.
    """
    import torch
    import torch_scatter

    original_scatter = torch_scatter.scatter

    def deterministic_scatter(
        src,
        index,
        dim=-1,
        out=None,
        dim_size=None,
        reduce="sum",
    ):
        normalized_dim = src.dim() + dim if dim < 0 else dim
        normalized_reduce = "sum" if reduce == "add" else reduce
        if out is not None:
            raise ValueError("deterministic scatter does not support out=")
        if normalized_reduce not in {"sum", "mean", "min", "max"}:
            return original_scatter(
                src,
                index,
                dim=dim,
                out=out,
                dim_size=dim_size,
                reduce=reduce,
            )
        if index.dim() != 1:
            raise ValueError("deterministic scatter requires one-dimensional index")
        if normalized_dim < 0 or normalized_dim >= src.dim():
            raise ValueError("scatter dimension is outside the source tensor")
        if src.shape[normalized_dim] != index.numel():
            raise ValueError("scatter index does not align with source dimension")
        if dim_size is None:
            dim_size = int(index.max().item()) + 1 if index.numel() else 0
        moved = src.movedim(normalized_dim, 0)
        # Equal destination indices leave their reduction order unspecified on
        # the Xi'an DTK sort kernel. Add the original position as a unique
        # secondary key so each segment is accumulated in a fixed order.
        positions = torch.arange(index.numel(), device=index.device, dtype=torch.long)
        grouped_key = index.to(torch.long) * (index.numel() + 1) + positions
        order = torch.argsort(grouped_key)
        sorted_index = index[order]
        boundaries = torch.arange(
            int(dim_size) + 1,
            device=index.device,
            dtype=index.dtype,
        )
        ptr = torch.searchsorted(sorted_index, boundaries)
        reduced = torch_scatter.segment_csr(
            moved[order],
            ptr,
            reduce=normalized_reduce,
        )
        return reduced.movedim(0, normalized_dim)

    def deterministic_scatter_add(src, index, dim=-1, out=None, dim_size=None):
        return deterministic_scatter(src, index, dim, out, dim_size, "sum")

    def deterministic_scatter_mean(src, index, dim=-1, out=None, dim_size=None):
        return deterministic_scatter(src, index, dim, out, dim_size, "mean")

    torch_scatter.scatter = deterministic_scatter
    torch_scatter.scatter_add = deterministic_scatter_add
    torch_scatter.scatter_mean = deterministic_scatter_mean
    return {
        "installed": True,
        "supported_reductions": ["sum", "mean", "min", "max"],
        "implementation": "unique-group-key+argsort+searchsorted+segment_csr",
    }


def _operators(groups: int, index, ptr) -> dict[str, Callable]:
    import torch
    import torch_scatter

    def native_index_add(values):
        output = values.new_zeros((groups, values.shape[-1]))
        return output.index_add_(0, index, values)

    def scatter_add(values):
        return torch_scatter.scatter_add(values, index, dim=0, dim_size=groups)

    def scatter_mean(values):
        return torch_scatter.scatter_mean(values, index, dim=0, dim_size=groups)

    positions = torch.arange(index.numel(), device=index.device, dtype=torch.long)
    grouped_key = index.to(torch.long) * (index.numel() + 1) + positions
    order = torch.argsort(grouped_key)

    def segment_sum(values):
        return torch_scatter.segment_csr(values[order], ptr, reduce="sum")

    def segment_mean(values):
        return torch_scatter.segment_csr(values[order], ptr, reduce="mean")

    return {
        "native_index_add_sum": native_index_add,
        "torch_scatter_add": scatter_add,
        "torch_scatter_mean": scatter_mean,
        "sorted_segment_csr_sum": segment_sum,
        "sorted_segment_csr_mean": segment_mean,
    }


def run_operator_audit(
    *,
    output: Path,
    strict: bool,
    repetitions: int = 12,
    rows: int = 262_144,
    width: int = 32,
    groups: int = 4_096,
    seed: int = 42,
) -> dict:
    import numpy as np
    import torch
    import torch_scatter

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one visible DCU is required")
    if repetitions < 2 or rows < groups or width < 1:
        raise ValueError("invalid audit shape")

    settings = configure_determinism(strict)
    random = np.random.RandomState(seed)
    base_cpu = torch.from_numpy(random.standard_normal((rows, width)).astype("float32"))
    projection_cpu = torch.from_numpy(random.standard_normal((width, width)).astype("float32"))
    readout_cpu = torch.from_numpy(random.standard_normal((groups, width)).astype("float32"))
    index_cpu = torch.from_numpy(random.randint(0, groups, size=rows).astype("int64"))
    counts_cpu = torch.bincount(index_cpu, minlength=groups)
    ptr_cpu = torch.cat([torch.zeros(1, dtype=torch.long), counts_cpu.cumsum(0)])

    device = torch.device("cuda:0")
    base = base_cpu.to(device)
    projection = projection_cpu.to(device)
    readout = readout_cpu.to(device)
    index = index_cpu.to(device)
    ptr = ptr_cpu.to(device)
    operators = _operators(groups, index, ptr)
    results = {}

    for name, operation in operators.items():
        repetitions_out = []
        reference_forward = None
        reference_gradient = None
        started = time.perf_counter()
        for repetition in range(repetitions):
            try:
                source = base.detach().clone().requires_grad_(True)
                hidden = torch.tanh(source.matmul(projection))
                reduced = operation(hidden)
                loss = (reduced * readout).sum()
                loss.backward()
                torch.cuda.synchronize(device)
                gradient = source.grad
                if gradient is None:
                    raise RuntimeError("source gradient is missing")
                if not bool(torch.isfinite(reduced).all() and torch.isfinite(gradient).all()):
                    raise RuntimeError("non-finite output or gradient")
                if reference_forward is None:
                    reference_forward = reduced.detach().clone()
                    reference_gradient = gradient.detach().clone()
                row = {
                    "repetition": repetition,
                    "error": None,
                    "forward_sha256": _tensor_sha256(reduced),
                    "gradient_sha256": _tensor_sha256(gradient),
                    "forward_max_abs_diff": float((reduced - reference_forward).abs().max().item()),
                    "gradient_max_abs_diff": float((gradient - reference_gradient).abs().max().item()),
                }
            except Exception as error:
                row = {
                    "repetition": repetition,
                    "error": f"{type(error).__name__}: {error}",
                }
            repetitions_out.append(row)
        result = summarize_repetitions(repetitions_out)
        result["elapsed_s"] = time.perf_counter() - started
        result["repetitions"] = repetitions_out
        results[name] = result

    payload = {
        "format": "molgap-xian-reduction-determinism-audit-v1",
        "complete": True,
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "node": platform.node(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_hip": getattr(torch.version, "hip", None),
        "torch_scatter": getattr(torch_scatter, "__version__", None),
        "device_name": torch.cuda.get_device_name(0),
        "settings": settings,
        "shape": {"rows": rows, "width": width, "groups": groups},
        "seed": seed,
        "results": results,
        "dataset_read": False,
        "model_executed": False,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
    }
    atomic_json(output, payload)
    return payload


def _model_forward(model, batch):
    return model(
        batch.x,
        batch.edge_index,
        batch.edge_attr,
        batch.batch,
        batch.random_walk_pe,
        batch.wedge_edge_ids,
        batch.edge_distance,
        batch.wedge_angle_cos,
        batch.geometry_valid,
    ).view(-1)


def _state_sha256(model) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _state_dict_sha256(state: dict) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _gradient_fingerprints(model) -> tuple[str, dict[str, dict]]:
    digest = hashlib.sha256()
    fingerprints = {}
    for name, parameter in sorted(model.named_parameters()):
        if parameter.grad is None:
            raise RuntimeError(f"missing gradient: {name}")
        gradient = parameter.grad.detach().cpu().contiguous()
        value_hash = hashlib.sha256(gradient.numpy().tobytes()).hexdigest()
        digest.update(name.encode("utf-8"))
        digest.update(gradient.numpy().tobytes())
        fingerprints[name] = {
            "sha256": value_hash,
            "l2_norm": float(gradient.float().norm().item()),
            "max_abs": float(gradient.float().abs().max().item()),
        }
    return digest.hexdigest(), fingerprints


def run_model_replay_audit(
    *,
    model_source: Path,
    cache_root: Path,
    output: Path,
    repetitions: int = 3,
    batch_size: int = 128,
    seed: int = 42,
) -> dict:
    import random
    import sys

    import numpy as np
    import torch

    expected_cache = "3e4206fd239942ab79f9c4978cd4334f6025ba14e8f1cd1df78c20060a0d1d22"
    expected_parameters = 3_665_809
    candidate = "ogb_distance_angle_triangle_edge_state_graph_state9"
    if repetitions < 2:
        raise ValueError("at least two model repetitions are required")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one visible DCU is required")

    settings = configure_determinism(True)
    patch = install_sorted_segment_scatter()
    import molgap

    source_package = str(model_source / "src" / "molgap")
    if source_package not in molgap.__path__:
        molgap.__path__.append(source_package)

    from torch_geometric.loader import DataLoader
    from molgap.pcqm_gap_architecture import make_pcqm_gap_encoder

    manifest = json.loads((cache_root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("aggregate_sha256") != expected_cache:
        raise RuntimeError("cache aggregate identity changed")
    first_train = next(row for row in manifest["shards"] if row["role"] == "train")
    shard_path = cache_root / first_train["file"]
    if _file_sha256(shard_path) != first_train["sha256"]:
        raise RuntimeError("first train shard hash changed")
    try:
        graphs = torch.load(shard_path, map_location="cpu", weights_only=False)
    except TypeError:
        graphs = torch.load(shard_path, map_location="cpu")
    graphs = graphs[:batch_size]
    if len(graphs) != batch_size:
        raise RuntimeError("first train shard cannot fill the replay batch")
    batch = next(iter(DataLoader(graphs, batch_size=batch_size, shuffle=False)))
    device = torch.device("cuda:0")

    repetition_rows = []
    reference_prediction = None
    for repetition in range(repetitions):
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model = make_pcqm_gap_encoder(candidate).to(device)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        if parameter_count != expected_parameters:
            raise RuntimeError(f"parameter count changed: {parameter_count}")
        initial_hash = _state_sha256(model)
        model.train()
        model.zero_grad(set_to_none=True)
        torch.manual_seed(seed + 1)
        torch.cuda.manual_seed_all(seed + 1)
        prediction = _model_forward(model, batch.to(device))
        target = batch.y.view(-1).float().to(device)
        loss = torch.nn.functional.l1_loss(prediction, target)
        loss.backward()
        torch.cuda.synchronize(device)
        if not bool(torch.isfinite(prediction).all() and torch.isfinite(loss)):
            raise RuntimeError("non-finite model replay")
        if reference_prediction is None:
            reference_prediction = prediction.detach().clone()
        gradient_hash, gradient_fingerprints = _gradient_fingerprints(model)
        repetition_rows.append(
            {
                "repetition": repetition,
                "initial_model_sha256": initial_hash,
                "prediction_sha256": _tensor_sha256(prediction),
                "gradient_sha256": gradient_hash,
                "gradient_fingerprints": gradient_fingerprints,
                "loss": float(loss.item()),
                "prediction_max_abs_diff": float(
                    (prediction - reference_prediction).abs().max().item()
                ),
            }
        )
        del model, prediction, loss
        torch.cuda.empty_cache()

    payload = {
        "format": "molgap-xian-model-determinism-audit-v1",
        "complete": True,
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "node": platform.node(),
        "torch": torch.__version__,
        "torch_hip": getattr(torch.version, "hip", None),
        "device_name": torch.cuda.get_device_name(0),
        "settings": settings,
        "scatter_patch": patch,
        "candidate": candidate,
        "parameter_count": expected_parameters,
        "cache_aggregate_sha256": expected_cache,
        "train_shard": first_train["file"],
        "train_shard_sha256": first_train["sha256"],
        "model_source_hashes": {
            "pcqm_gap_architecture.py": _file_sha256(
                model_source / "src" / "molgap" / "pcqm_gap_architecture.py"
            ),
            "gps.py": _file_sha256(model_source / "src" / "molgap" / "gps.py"),
        },
        "batch_size": batch_size,
        "seed": seed,
        "repetitions": repetition_rows,
        "within_process_initial_model_identity": len(
            {row["initial_model_sha256"] for row in repetition_rows}
        ) == 1,
        "within_process_prediction_identity": len(
            {row["prediction_sha256"] for row in repetition_rows}
        ) == 1,
        "within_process_gradient_identity": len(
            {row["gradient_sha256"] for row in repetition_rows}
        ) == 1,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
    }
    atomic_json(output, payload)
    return payload


def accept_model_replays(first_path: Path, second_path: Path, output: Path) -> dict:
    first = json.loads(first_path.read_text(encoding="utf-8"))
    second = json.loads(second_path.read_text(encoding="utf-8"))
    if first.get("format") != "molgap-xian-model-determinism-audit-v1":
        raise ValueError("unexpected first model audit format")
    if second.get("format") != first["format"]:
        raise ValueError("unexpected second model audit format")
    identity_fields = (
        "candidate",
        "parameter_count",
        "cache_aggregate_sha256",
        "train_shard_sha256",
        "model_source_hashes",
        "batch_size",
        "seed",
        "torch",
        "torch_hip",
        "device_name",
    )
    same_contract = all(first.get(key) == second.get(key) for key in identity_fields)
    hashes = lambda payload, key: {row[key] for row in payload["repetitions"]}
    model_identity = hashes(first, "initial_model_sha256") == hashes(
        second, "initial_model_sha256"
    )
    prediction_identity = hashes(first, "prediction_sha256") == hashes(
        second, "prediction_sha256"
    )
    gradient_identity = hashes(first, "gradient_sha256") == hashes(
        second, "gradient_sha256"
    )
    first_gradients = first["repetitions"][0]["gradient_fingerprints"]
    second_gradients = second["repetitions"][0]["gradient_fingerprints"]
    differing_gradient_parameters = sorted(
        name
        for name in set(first_gradients) | set(second_gradients)
        if first_gradients.get(name, {}).get("sha256")
        != second_gradients.get(name, {}).get("sha256")
    )
    accepted = bool(
        same_contract
        and model_identity
        and prediction_identity
        and gradient_identity
        and all(
            payload[field]
            for payload in (first, second)
            for field in (
                "within_process_initial_model_identity",
                "within_process_prediction_identity",
                "within_process_gradient_identity",
            )
        )
    )
    payload = {
        "format": "molgap-xian-model-determinism-acceptance-v1",
        "complete": True,
        "same_contract": same_contract,
        "initial_model_bitwise_replay": model_identity,
        "prediction_bitwise_replay": prediction_identity,
        "gradient_bitwise_replay": gradient_identity,
        "differing_gradient_parameter_count": len(differing_gradient_parameters),
        "differing_gradient_parameters": differing_gradient_parameters,
        "accepted": accepted,
        "short_training_retest_authorized": accepted,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
    }
    atomic_json(output, payload)
    return payload


def accept_short_training_retests(
    first_root: Path,
    second_root: Path,
    output: Path,
) -> dict:
    import torch

    first = json.loads((first_root / "metrics.json").read_text(encoding="utf-8"))
    second = json.loads((second_root / "metrics.json").read_text(encoding="utf-8"))
    first_runtime = json.loads(
        (first_root / "deterministic_runtime.json").read_text(encoding="utf-8")
    )
    second_runtime = json.loads(
        (second_root / "deterministic_runtime.json").read_text(encoding="utf-8")
    )
    identity_fields = (
        "candidate",
        "model_source_commit",
        "cache_aggregate_sha256",
        "seed",
        "precision",
        "batch_size",
        "optimizer",
        "learning_rate",
        "weight_decay",
        "target",
        "parameter_count",
        "device",
    )
    same_contract = all(first.get(key) == second.get(key) for key in identity_fields)
    metric_keys = ("epoch", "train_mae_eV", "validation_mae_eV")
    first_metrics = [tuple(row[key] for key in metric_keys) for row in first["epochs"]]
    second_metrics = [tuple(row[key] for key in metric_keys) for row in second["epochs"]]
    epoch_metrics_exact = first_metrics == second_metrics
    initialization_exact = (
        first_runtime["initialization"] == second_runtime["initialization"]
    )
    first_batch_exact = first_runtime["first_batch"] == second_runtime["first_batch"]
    runtime_contract_valid = all(
        runtime.get("complete") is True
        and runtime.get("rocblas_default_atomics_mode") == "0"
        and runtime.get("deterministic_algorithms_enabled") is True
        and runtime.get("scatter_implementation")
        == "unique-group-key+argsort+searchsorted+segment_csr"
        and runtime.get("data_loader_generator_seed") == 42
        for runtime in (first_runtime, second_runtime)
    )

    def load(path: Path):
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(path, map_location="cpu")

    first_best_hash = _state_dict_sha256(load(first_root / "best_model.pt"))
    second_best_hash = _state_dict_sha256(load(second_root / "best_model.pt"))
    first_last = load(first_root / "last_checkpoint.pt")
    second_last = load(second_root / "last_checkpoint.pt")
    first_last_hash = _state_dict_sha256(first_last["model_state"])
    second_last_hash = _state_dict_sha256(second_last["model_state"])
    accepted = bool(
        same_contract
        and runtime_contract_valid
        and initialization_exact
        and first_batch_exact
        and epoch_metrics_exact
        and first_best_hash == second_best_hash
        and first_last_hash == second_last_hash
    )
    payload = {
        "format": "molgap-xian-short-training-repeatability-v1",
        "complete": True,
        "same_contract": same_contract,
        "runtime_contract_valid": runtime_contract_valid,
        "initialization_bitwise_equal": initialization_exact,
        "first_batch_bitwise_equal": first_batch_exact,
        "epoch_metrics_bitwise_equal": epoch_metrics_exact,
        "best_model_state_bitwise_equal": first_best_hash == second_best_hash,
        "last_model_state_bitwise_equal": first_last_hash == second_last_hash,
        "first_best_model_state_sha256": first_best_hash,
        "second_best_model_state_sha256": second_best_hash,
        "first_last_model_state_sha256": first_last_hash,
        "second_last_model_state_sha256": second_last_hash,
        "first_epoch_metrics": first_metrics,
        "second_epoch_metrics": second_metrics,
        "accepted": accepted,
        "ranking_threshold_may_be_reestimated": accepted,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
    }
    atomic_json(output, payload)
    return payload


def accept_operator_audits(
    default_path: Path,
    strict_path: Path,
    strict_repeat_path: Path,
    output: Path,
) -> dict:
    default = json.loads(default_path.read_text(encoding="utf-8"))
    strict = json.loads(strict_path.read_text(encoding="utf-8"))
    strict_repeat = json.loads(strict_repeat_path.read_text(encoding="utf-8"))
    if default.get("format") != "molgap-xian-reduction-determinism-audit-v1":
        raise ValueError("unexpected default audit format")
    if strict.get("format") != default["format"]:
        raise ValueError("unexpected strict audit format")
    if strict_repeat.get("format") != default["format"]:
        raise ValueError("unexpected repeated strict audit format")
    if not (
        default.get("shape") == strict.get("shape") == strict_repeat.get("shape")
        and default.get("seed") == strict.get("seed") == strict_repeat.get("seed")
    ):
        raise ValueError("audit inputs differ")
    if not all(
        payload["settings"].get("deterministic_algorithms_enabled") is True
        for payload in (strict, strict_repeat)
    ):
        raise ValueError("strict audit did not enable deterministic algorithms")

    strict_results = strict["results"]
    repeated_results = strict_repeat["results"]

    def exact(name: str) -> bool:
        row = strict_results[name]
        repeated = repeated_results[name]
        first_hashes = {
            (item["forward_sha256"], item["gradient_sha256"])
            for item in row["repetitions"]
            if item.get("error") is None
        }
        repeated_hashes = {
            (item["forward_sha256"], item["gradient_sha256"])
            for item in repeated["repetitions"]
            if item.get("error") is None
        }
        return bool(
            row["successful_repetitions"] >= 2
            and repeated["successful_repetitions"] >= 2
            and row["all_forward_bitwise_equal"]
            and row["all_gradient_bitwise_equal"]
            and repeated["all_forward_bitwise_equal"]
            and repeated["all_gradient_bitwise_equal"]
            and not row["errors"]
            and not repeated["errors"]
            and first_hashes == repeated_hashes
        )

    direct_exact = exact("torch_scatter_add") and exact("torch_scatter_mean")
    csr_exact = exact("sorted_segment_csr_sum") and exact("sorted_segment_csr_mean")
    payload = {
        "format": "molgap-xian-reduction-determinism-acceptance-v1",
        "complete": True,
        "default_audit": str(default_path),
        "strict_audit": str(strict_path),
        "strict_repeat_audit": str(strict_repeat_path),
        "same_runtime": all(
            default.get(key) == strict.get(key) == strict_repeat.get(key)
            for key in ("torch", "torch_hip", "torch_scatter", "device_name")
        ),
        "cross_process_replay_required": True,
        "strict_scatter_bitwise_replay": direct_exact,
        "strict_sorted_csr_bitwise_replay": csr_exact,
        "operator_fix_candidate": "strict_scatter" if direct_exact else ("sorted_segment_csr" if csr_exact else None),
        "model_replay_required": True,
        "training_retest_authorized": False,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
    }
    atomic_json(output, payload)
    return payload
