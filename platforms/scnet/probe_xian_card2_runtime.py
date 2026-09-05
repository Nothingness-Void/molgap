"""Compute-node-only SCNet Xi'an card2 runtime probe."""
from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import time
from pathlib import Path


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def module_status(name: str) -> dict:
    try:
        module = importlib.import_module(name)
        return {
            "available": True,
            "version": getattr(module, "__version__", None),
        }
    except Exception as error:
        return {
            "available": False,
            "error": f"{type(error).__name__}: {error}",
        }


def run() -> dict:
    import torch

    started = time.perf_counter()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one visible DCU is required")
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.reset_peak_memory_stats(0)

    left = torch.randn((1024, 1024), device=device, requires_grad=True)
    right = torch.randn((1024, 1024), device=device, requires_grad=True)
    loss = (left @ right).square().mean()
    loss.backward()
    torch.cuda.synchronize(0)
    gradients_finite = bool(
        left.grad is not None
        and right.grad is not None
        and torch.isfinite(left.grad).all()
        and torch.isfinite(right.grad).all()
    )
    if not bool(torch.isfinite(loss)) or not gradients_finite:
        raise RuntimeError("non-finite accelerator forward/backward")

    properties = torch.cuda.get_device_properties(0)
    dependencies = {
        name: module_status(name)
        for name in ("torch_geometric", "torch_scatter", "torch_sparse", "ogb")
    }
    return {
        "format": "molgap-scnet-xian-card2-runtime-v1",
        "accepted_accelerator_runtime": True,
        "training_stack_ready": all(
            dependencies[name]["available"]
            for name in ("torch_geometric", "ogb")
        ),
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "node": platform.node(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "hip": getattr(torch.version, "hip", None),
        "device_count": torch.cuda.device_count(),
        "device_name": torch.cuda.get_device_name(0),
        "device_total_memory_bytes": int(properties.total_memory),
        "loss_finite": True,
        "gradients_finite": gradients_finite,
        "peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
        "dependencies": dependencies,
        "elapsed_s": time.perf_counter() - started,
        "model_executed": False,
        "dataset_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run()
    except Exception as error:
        result = {
            "format": "molgap-scnet-xian-card2-runtime-v1",
            "accepted_accelerator_runtime": False,
            "training_stack_ready": False,
            "error": f"{type(error).__name__}: {error}",
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "model_executed": False,
            "dataset_read": False,
        }
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2), flush=True)
    if result.get("accepted_accelerator_runtime") is not True:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
