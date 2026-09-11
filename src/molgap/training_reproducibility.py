"""Reproducibility and durable-checkpoint primitives for remote training."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import random
import sys
from collections.abc import Mapping
from pathlib import Path

import numpy as np


RUNTIME_MANIFEST_FORMAT = "molgap-runtime-manifest-v1"
CHECKPOINT_FORMAT = "molgap-resumable-checkpoint-v1"
MODEL_BUNDLE_FORMAT = "molgap-self-contained-model-v1"


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
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
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
    }
    payload["runtime_fingerprint"] = canonical_fingerprint(payload)
    return payload


def assert_finite_state_dict(state_dict: Mapping, *, label: str) -> None:
    import torch

    failures = [
        name
        for name, value in state_dict.items()
        if torch.is_tensor(value) and not bool(torch.isfinite(value).all())
    ]
    if failures:
        raise RuntimeError(f"{label} contains non-finite tensors: {failures[:8]}")


def atomic_torch_save(path: Path, value) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def atomic_json(path: Path, value: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(dict(value), indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
