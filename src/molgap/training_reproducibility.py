"""Reproducibility and durable-checkpoint primitives for remote training."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import sys
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path

import numpy as np


RUNTIME_MANIFEST_FORMAT = "molgap-runtime-manifest-v1"
CHECKPOINT_FORMAT = "molgap-resumable-checkpoint-v1"
MODEL_BUNDLE_FORMAT = "molgap-self-contained-model-v1"


@contextmanager
def deterministic_memory_fill_context(*, enabled: bool):
    """Scope the process-global safety fill; never disable deterministic algorithms.

    Exclusive execution workers only. Turning the fill off requires separate
    qualification that the executed graph never reads uninitialized storage;
    this helper supplies neither that evidence nor training authorization.
    """
    import torch

    if type(enabled) is not bool:
        raise ValueError("Memory fill selection must be an exact bool")
    if (not torch.are_deterministic_algorithms_enabled()
            or torch.is_deterministic_algorithms_warn_only_enabled()):
        raise ValueError("Fail-closed deterministic algorithms are required")
    previous = torch.utils.deterministic.fill_uninitialized_memory
    try:
        torch.utils.deterministic.fill_uninitialized_memory = enabled
        yield
    finally:
        torch.utils.deterministic.fill_uninitialized_memory = previous


def canonical_fingerprint(value: Mapping) -> str:
    payload = json.dumps(
        dict(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def configure_fp32_determinism(seed: int) -> dict:
    """Configure fail-closed FP32 execution before model construction."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True, warn_only=False)
    return {
        "seed": int(seed),
        "precision": "fp32",
        "tf32_enabled": False,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "deterministic_algorithms": True,
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "float32_matmul_precision": (
            torch.get_float32_matmul_precision()
            if hasattr(torch, "get_float32_matmul_precision")
            else "tf32-disabled-by-backend-flags"
        ),
    }


def capture_rng_state(*, loader_generator=None) -> dict:
    import torch

    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    if loader_generator is not None:
        state["loader_generator"] = loader_generator.get_state()
    return state


def validate_accelerator_request(requested: str, observed: str, *, hip_runtime: bool = False) -> None:
    """Match resource metadata, never select a different execution algorithm."""
    import re
    if (type(requested) is not str or not requested.strip() or type(observed) is not str
            or not observed.strip() or type(hip_runtime) is not bool):
        raise ValueError("Explicit declared and observed accelerator names are required")
    def compact(value):
        return re.sub(r"[^A-Z0-9]", "", value.upper())
    def model_name(value):
        return re.sub(r"^(?:(?:NVIDIA|AMD|TESLA|GEFORCE|QUADRO|INSTINCT))+", "", compact(value)).removesuffix("GPU")
    declared, actual = model_name(requested), model_name(observed)
    # Vendor/suffix stripping must not turn two unspecified names into a match.
    unspecified = {"", "GPU", "CUDA", "ACCELERATOR"}
    if declared in unspecified or actual in unspecified:
        matches = False
    elif compact(requested) == "DCU":
        matches = hip_runtime
    elif declared == "A100":
        matches = re.fullmatch(r"A100(?:(?:SXM[24]|PCIE|40GB|80GB))*", actual) is not None
    else:
        matches = declared == actual
    if not matches:
        raise ValueError("Observed accelerator differs from frozen request: " + requested + " / " + observed)


def restore_rng_state(state: Mapping, *, loader_generator=None) -> None:
    import torch

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])
    if loader_generator is not None:
        loader_generator.set_state(state["loader_generator"].cpu())


def _installed_distributions() -> list[str]:
    values = []
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            values.append(f"{name.lower()}=={distribution.version}")
    return sorted(set(values))


def build_runtime_manifest(determinism: Mapping) -> dict:
    """Capture the complete Python package set and accelerator runtime."""
    import torch

    accelerator = None
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        accelerator = {
            "name": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "total_memory_bytes": int(properties.total_memory),
            "device_count_visible": int(torch.cuda.device_count()),
        }
    packages = _installed_distributions()
    package_fingerprint = hashlib.sha256("\n".join(packages).encode()).hexdigest()
    external_runtime_artifacts = {}
    try:
        import ogb

        external_runtime_artifacts["ogb"] = {
            "version": getattr(ogb, "__version__", "unknown"),
            "source_archive_sha256": os.environ.get(
                "MOLGAP_OGB_SOURCE_SHA256"
            ),
        }
    except ImportError:
        pass
    payload = {
        "format": RUNTIME_MANIFEST_FORMAT,
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "torch_hip": torch.version.hip,
        "cudnn": torch.backends.cudnn.version(),
        "accelerator": accelerator,
        "determinism": dict(determinism),
        "installed_distributions": packages,
        "installed_distributions_sha256": package_fingerprint,
        "external_runtime_artifacts": external_runtime_artifacts,
    }
    payload["runtime_fingerprint"] = canonical_fingerprint(payload)
    return payload


def nonfinite_tensor_keys(state_dict: Mapping) -> list:
    """Check every tensor, collecting accelerator flags with one copy per device."""
    import torch

    failures, device_flags, cpu_scalars = [], {}, {}
    for name, value in state_dict.items():
        if not torch.is_tensor(value):
            continue
        if (type(value) is torch.Tensor and value.device.type == "cpu"
                and value.ndim == 0 and value.layout == torch.strided and not value.is_quantized):
            cpu_scalars.setdefault(value.dtype, []).append((name, value.detach()))
            continue
        finite = torch.isfinite(value).all()
        if value.device.type == "cpu":
            if not bool(finite):
                failures.append(name)
        else:
            device_flags.setdefault(value.device, []).append((name, finite))
    # Partial-pass metrics and AdamW counters need no individual scalar kernels.
    # Group by dtype so checking cannot introduce a narrowing conversion.
    for values in cpu_scalars.values():
        observed = torch.isfinite(torch.stack([value for _, value in values])).tolist()
        failures.extend(name for (name, _), finite in zip(values, observed) if not finite)
    # Keep each exact per-tensor check, but avoid a host/device barrier per key.
    for flags in device_flags.values():
        observed = torch.stack([flag for _, flag in flags]).cpu().tolist()
        failures.extend(name for (name, _), finite in zip(flags, observed) if not finite)
    failed = set(failures)
    return [name for name in state_dict if name in failed]


def assert_finite_state_dict(state_dict: Mapping, *, label: str) -> None:
    failures = nonfinite_tensor_keys(state_dict)
    if failures:
        raise RuntimeError(f"{label} contains non-finite tensors: {failures[:8]}")


def atomic_torch_save(path: Path, value) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def sum_scalar_metrics(values) -> float:
    """Transfer detached batch scalars once, preserving Python addition order.

    Do not replace this with a tensor sum: that changes the retained metric's
    reduction precision/order. Values have no connection to backward or RNG.
    """
    import torch

    values = list(values)
    if not values:
        return 0.0
    if isinstance(values[0], torch.Tensor):
        first = values[0]
        if any(not isinstance(v, torch.Tensor) or v.ndim != 0
               or v.dtype != first.dtype or v.device != first.device
               or not v.is_floating_point() for v in values):
            raise ValueError("Expected matching floating-point scalar metrics")
        values = torch.stack([v.detach() for v in values]).cpu().tolist()
    elif any(isinstance(v, (bool, torch.Tensor)) or not isinstance(v, (int, float))
             for v in values):
        raise ValueError("Expected scalar metrics, not mixed tensor/host values")
    total = 0.0
    for value in values:
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("Nonfinite scalar metric")
        total += value
    if not math.isfinite(total):
        raise ValueError("Nonfinite scalar metric total")
    return total


def atomic_json(path: Path, value: Mapping | list) -> None:
    """Write JSON objects or row arrays without exposing a partial destination."""
    if isinstance(value, Mapping):
        value = dict(value)
    elif not isinstance(value, list):
        raise TypeError("Expected a JSON object or array")
    payload = json.dumps(value, indent=2) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(payload, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
