"""Reproducible latency measurements for registered inference pipelines."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import time
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch

from .constants import MODEL_REGISTRY


# These ordinary CHONSFCl molecules exercise aromatic, aliphatic, charged-free,
# and heteroatom paths without turning a latency benchmark into a data split.
DEFAULT_SMILES = (
    "CCO",
    "c1ccccc1",
    "CC(=O)Oc1ccccc1C(=O)O",
    "CCN(CC)CC",
    "O=C(Nc1ccccc1)C",
    "c1ccc2ccccc2c1",
    "COc1ccc(C=O)cc1",
    "CCOC(=O)c1ccccc1",
    "O=S(=O)(N)c1ccccc1",
    "CC(C)(C)c1ccc(O)cc1",
    "C1CCCCC1",
    "CC(C)OC(=O)N1CCCCC1",
    "FC1=CC=C(C=C1)C(=O)N",
    "Clc1ccccc1",
    "c1ncccc1",
    "CS(=O)(=O)c1ccccc1",
)


def benchmark_smiles(source: Iterable[str], rows: int) -> list[str]:
    """Repeat a non-empty source suite to create an exact benchmark batch."""
    values = [value.strip() for value in source if value.strip()]
    if rows < 1:
        raise ValueError("benchmark rows must be positive")
    if not values:
        raise ValueError("benchmark SMILES suite is empty")
    return [values[index % len(values)] for index in range(rows)]


def _sha256_text(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def routed_v4_artifacts() -> dict[str, dict[str, str]]:
    """Return the exact checkpoint identity for the registered routed v4 path."""
    routed = MODEL_REGISTRY["phase8_routed_dualgps_hybrid"]
    base = MODEL_REGISTRY[routed["base_hybrid"]]
    extra = MODEL_REGISTRY[routed["extra_gps"]]
    paths = {
        "base_gps": Path(base["components"] and MODEL_REGISTRY[base["components"][0]]["checkpoint"]),
        "base_schnet": Path(base["components"] and MODEL_REGISTRY[base["components"][1]]["checkpoint"]),
        "base_fusion": Path(base["checkpoint"]),
        "extra_gps": Path(extra["checkpoint"]),
        "dual_gps_fusion": Path(routed["checkpoint"]),
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"routed v4 checkpoint assets missing: {missing}")
    return {
        name: {"path": str(path), "sha256": _sha256_file(path)}
        for name, path in paths.items()
    }


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _hardware(device: torch.device) -> dict[str, object]:
    result: dict[str, object] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(device),
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        result["cuda"] = {
            "name": properties.name,
            "total_memory_bytes": int(properties.total_memory),
            "cuda_runtime": torch.version.cuda,
        }
    return result


def _summary(samples_s: list[float], rows: int) -> dict[str, float | list[float]]:
    values = np.asarray(samples_s, dtype=np.float64)
    median_s = float(np.median(values))
    return {
        "samples_s": [float(value) for value in values],
        "median_batch_s": median_s,
        "p95_batch_s": float(np.percentile(values, 95)),
        "median_ms_per_molecule": median_s * 1000.0 / rows,
        "median_molecules_per_s": rows / median_s,
    }


def benchmark_routed_v4(
    *,
    device: str | torch.device | None = None,
    batch_sizes: Iterable[int] = (1, 16, 64),
    repeats: int = 3,
    warmups: int = 1,
    batch_2d: int = 256,
    batch_3d: int = 128,
    smiles_source: Iterable[str] = DEFAULT_SMILES,
) -> dict[str, object]:
    """Measure warm end-to-end routed-v4 new-SMILES inference.

    The timed path intentionally includes RDKit parsing, ETKDG, PyG graph
    creation, GPS/SchNet encoder passes, and routing. It does not represent a
    lookup against a precomputed molecular database.
    """
    if repeats < 1 or warmups < 0:
        raise ValueError("repeats must be positive and warmups cannot be negative")
    sizes = [int(value) for value in batch_sizes]
    if not sizes or any(value < 1 for value in sizes):
        raise ValueError("at least one positive batch size is required")

    from .inference import load_routed_dual_gps_hybrid, predict_smiles_batch_routed_dual_gps

    resolved_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    _synchronize(resolved_device)
    load_start = time.perf_counter()
    models = load_routed_dual_gps_hybrid(resolved_device)
    _synchronize(resolved_device)
    load_s = time.perf_counter() - load_start

    results: list[dict[str, object]] = []
    source_values = [value.strip() for value in smiles_source if value.strip()]
    for rows in sizes:
        smiles = benchmark_smiles(source_values, rows)

        def run_once():
            return predict_smiles_batch_routed_dual_gps(
                smiles,
                models=models,
                bs_2d=batch_2d,
                bs_3d=batch_3d,
                device=resolved_device,
            )

        for _ in range(warmups):
            run_once()
            _synchronize(resolved_device)

        if resolved_device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(resolved_device)
        samples_s: list[float] = []
        routed_fractions: list[float] = []
        for _ in range(repeats):
            _synchronize(resolved_device)
            start = time.perf_counter()
            valid_idx, predictions, routed = run_once()
            _synchronize(resolved_device)
            samples_s.append(time.perf_counter() - start)
            if len(valid_idx) != rows or not np.isfinite(predictions).all():
                raise RuntimeError("benchmark inference did not produce finite predictions for every input")
            routed_fractions.append(float(np.mean(routed)))
        record: dict[str, object] = {
            "input_rows": rows,
            "valid_rows": rows,
            "routed_fraction": float(np.mean(routed_fractions)),
            **_summary(samples_s, rows),
        }
        if resolved_device.type == "cuda":
            record["peak_memory_bytes"] = int(torch.cuda.max_memory_allocated(resolved_device))
        results.append(record)

    return {
        "schema_version": 1,
        "status": "complete",
        "model": "routed_gps7_gps9_schnet_500k_v4",
        "registry_key": "phase8_routed_dualgps_hybrid",
        "measurement": {
            "scope": "warm_end_to_end_new_smiles_inference",
            "included": [
                "RDKit_SMILES_parsing",
                "ETKDG_3D_conformer_construction",
                "2D_and_3D_PyG_graph_construction",
                "GPS7_and_SchNet_base_encoder_forward",
                "conditional_GPS9_and_dual_fusion_forward",
            ],
            "excluded": ["model_checkpoint_load", "precomputed_catalog_lookup"],
            "model_load_s": float(load_s),
            "warmup_batches_per_size": warmups,
            "timed_repeats_per_size": repeats,
            "batch_2d": batch_2d,
            "batch_3d": batch_3d,
        },
        "hardware": _hardware(resolved_device),
        "input_suite": {
            "source_unique_smiles": len(source_values),
            "source_sha256": _sha256_text(source_values),
        },
        "artifacts": routed_v4_artifacts(),
        "results": results,
    }


def benchmark_markdown(result: dict[str, object]) -> str:
    """Render a compact human-readable companion to the machine record."""
    measurement = result["measurement"]
    hardware = result["hardware"]
    lines = [
        "# Routed V4 Local Inference Latency",
        "",
        "This measures warm end-to-end inference for new SMILES, including "
        "ETKDG and graph construction. It is not a precomputed-catalog lookup benchmark.",
        "",
        f"- Model: `{result['model']}`",
        f"- Device: `{hardware['device']}`",
        f"- Model load: `{measurement['model_load_s']:.3f} s` (excluded from warm timings)",
        f"- Timed repeats per batch: `{measurement['timed_repeats_per_size']}`",
        "",
        "| Inputs | Median batch s | P95 batch s | Median ms/mol | Molecules/s | Routed fraction | Peak GPU MiB |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["results"]:
        memory = row.get("peak_memory_bytes")
        memory_mib = "" if memory is None else f"{memory / 1024**2:.1f}"
        lines.append(
            "| {input_rows} | {median_batch_s:.4f} | {p95_batch_s:.4f} | "
            "{median_ms_per_molecule:.2f} | {median_molecules_per_s:.2f} | "
            "{routed_fraction:.3f} | {memory} |".format(**row, memory=memory_mib)
        )
    lines.extend(
        [
            "",
            "The selected repaired-2M dense/equal pure-2D checkpoints are not "
            "included until their local inventory and public inference loader are accepted.",
            "",
        ]
    )
    return "\n".join(lines)


def write_benchmark_artifacts(result: dict[str, object], output_json: Path) -> tuple[Path, Path]:
    """Atomically persist JSON evidence and its Markdown companion."""
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_markdown = output_json.with_suffix(".md")
    for path, value in (
        (output_json, json.dumps(result, indent=2) + "\n"),
        (output_markdown, benchmark_markdown(result)),
    ):
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(value, encoding="utf-8")
        os.replace(temporary, path)
    return output_json, output_markdown
