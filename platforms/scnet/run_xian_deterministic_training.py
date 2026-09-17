"""Run the frozen historical Xi'an trainer with deterministic reductions."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import runpy
import sys
from pathlib import Path


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _tensor_sha256(tensor) -> str:
    value = tensor.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(value).hexdigest()


def _state_sha256(model) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-source", type=Path, required=True)
    parser.add_argument("--benchmark-script", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    if os.environ.get("ROCBLAS_DEFAULT_ATOMICS_MODE") != "0":
        raise RuntimeError("ROCBLAS_DEFAULT_ATOMICS_MODE=0 is required")

    from molgap.reproducibility_audit import (
        configure_determinism,
        install_sorted_segment_scatter,
    )

    settings = configure_determinism(True)
    if settings["deterministic_algorithms_enabled"] is not True:
        raise RuntimeError("strict deterministic algorithms were not enabled")
    install_sorted_segment_scatter()

    import molgap

    model_package = str(args.model_source / "src" / "molgap")
    if model_package not in molgap.__path__:
        molgap.__path__.append(model_package)

    import torch
    import torch.nn.functional as functional
    import torch_geometric.loader as loader_module
    from molgap import pcqm_gap_architecture

    original_factory = pcqm_gap_architecture.make_pcqm_gap_encoder
    original_l1_loss = functional.l1_loss
    original_loader = loader_module.DataLoader
    initialization = {}
    first_batch = {}

    def audited_factory(*factory_args, **factory_kwargs):
        model = original_factory(*factory_args, **factory_kwargs)
        initialization.update(
            {
                "model_state_sha256": _state_sha256(model),
                "parameter_count": sum(
                    parameter.numel() for parameter in model.parameters()
                ),
            }
        )
        _atomic_json(args.output_root / "deterministic_initialization.json", initialization)
        return model

    def deterministic_loader(*loader_args, **loader_kwargs):
        if loader_kwargs.get("shuffle"):
            loader_kwargs["generator"] = torch.Generator().manual_seed(42)
        return original_loader(*loader_args, **loader_kwargs)

    def audited_l1_loss(input_tensor, target_tensor, *loss_args, **loss_kwargs):
        loss = original_l1_loss(
            input_tensor,
            target_tensor,
            *loss_args,
            **loss_kwargs,
        )
        if not first_batch:
            first_batch.update(
                {
                    "prediction_sha256": _tensor_sha256(input_tensor),
                    "target_sha256": _tensor_sha256(target_tensor),
                    "loss": float(loss.detach().item()),
                    "batch_rows": int(target_tensor.numel()),
                }
            )
            _atomic_json(args.output_root / "deterministic_first_batch.json", first_batch)
        return loss

    pcqm_gap_architecture.make_pcqm_gap_encoder = audited_factory
    loader_module.DataLoader = deterministic_loader
    functional.l1_loss = audited_l1_loss

    sys.argv = [
        str(args.benchmark_script),
        "--cache-root",
        str(args.cache_root),
        "--output-root",
        str(args.output_root),
        "--model-source-commit",
        "9068ddb82e6bdf16b841570abbff023b90c07f07",
        "--seed",
        "42",
        "--epochs",
        "3",
        "--batch-size",
        "48",
    ]
    runpy.run_path(str(args.benchmark_script), run_name="__main__")
    metrics_path = args.output_root / "metrics.json"
    if not metrics_path.is_file() or not initialization or not first_batch:
        raise RuntimeError("deterministic training audit artifacts are incomplete")
    _atomic_json(
        args.output_root / "deterministic_runtime.json",
        {
            "format": "molgap-xian-deterministic-training-runtime-v1",
            "complete": True,
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "rocblas_default_atomics_mode": os.environ.get(
                "ROCBLAS_DEFAULT_ATOMICS_MODE"
            ),
            "deterministic_algorithms_enabled": bool(
                torch.are_deterministic_algorithms_enabled()
            ),
            "scatter_implementation": (
                "unique-group-key+argsort+searchsorted+segment_csr"
            ),
            "data_loader_generator_seed": 42,
            "initialization": initialization,
            "first_batch": first_batch,
            "official_validation_role_read": False,
            "test_dev_role_read": False,
        },
    )


if __name__ == "__main__":
    main()
