"""Run a reproducible SchNet hidden-dimension screening experiment.

The runner samples one fixed graph subset, then delegates both arms to
``train_encoder.py``. Production checkpoints are never used as output paths.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from molgap.constants import RESULTS_DIR, SEED
from molgap.utils import ensure_dirs


DEFAULT_GRAPHS = RESULTS_DIR / "phase8" / "pyg_3d_graphs_etkdg_expansion_500k.pt"
DEFAULT_OUT = RESULTS_DIR / "phase8" / "experiments" / "schnet_dim_ab_50k"


def atomic_torch_save(value: object, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def atomic_json_write(value: object, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def build_subset(source: Path, subset_path: Path, manifest_path: Path,
                 sample_size: int, seed: int) -> None:
    if subset_path.is_file() and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = {
            "source": str(source),
            "sample_size": sample_size,
            "seed": seed,
        }
        if all(manifest.get(key) == value for key, value in expected.items()):
            print(f"Reuse fixed subset: {subset_path}", flush=True)
            return
        raise ValueError(f"Existing subset manifest does not match this run: {manifest_path}")

    print(f"Loading source graph cache: {source}", flush=True)
    graphs = torch.load(source, weights_only=False)
    if sample_size > len(graphs):
        raise ValueError(f"Requested {sample_size:,} graphs from a cache of {len(graphs):,}")
    selected = np.random.RandomState(seed).choice(len(graphs), sample_size, replace=False)
    subset = [graphs[int(index)] for index in selected]
    source_indices = np.asarray([
        int(graph.source_idx.view(-1)[0]) for graph in subset
    ], dtype=np.int64)
    selection_sha256 = hashlib.sha256(source_indices.tobytes()).hexdigest()
    atomic_torch_save(subset, subset_path)
    atomic_json_write({
        "source": str(source),
        "source_count": len(graphs),
        "sample_size": sample_size,
        "seed": seed,
        "source_idx_sha256": selection_sha256,
        "source_idx_min": int(source_indices.min()),
        "source_idx_max": int(source_indices.max()),
    }, manifest_path)
    print(f"Fixed random subset -> {subset_path}", flush=True)


def run_arm(args: argparse.Namespace, subset_path: Path, dimension: int) -> dict:
    arm = args.out_dir / f"hidden_{dimension}"
    ensure_dirs(arm)
    metrics_path = arm / "metrics.json"
    if metrics_path.is_file():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics.get("complete", True):
            print(f"Reuse completed arm: hidden={dimension}", flush=True)
            return metrics

    command = [
        sys.executable,
        str(Path(__file__).with_name("train_encoder.py")),
        "--kind", "schnet",
        "--graphs", str(subset_path),
        "--hidden-channels", str(dimension),
        "--epochs", str(args.epochs),
        "--patience", str(args.patience),
        "--batch-size", str(args.batch_size),
        "--eval-batch-size", str(args.batch_size),
        "--model-out", str(arm / "model.pt"),
        "--metrics-out", str(metrics_path),
        "--embeddings-out", str(arm / "embeddings.pt"),
        "--checkpoint-out", str(arm / "checkpoint.pt"),
        "--checkpoint-every", "1",
        "--no-embeddings",
    ]
    print(f"Training SchNet hidden={dimension}", flush=True)
    subprocess.run(command, check=True)
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def summarize(results: dict[int, dict], baseline_dimension: int, out_path: Path) -> dict:
    arms = {}
    for dimension, metrics in sorted(results.items()):
        test = metrics["test_metrics"]
        arms[str(dimension)] = {
            "n_params": metrics["n_params"],
            "training_time_s": metrics["training_time_s"],
            "best_val_mae": metrics["best_val_mae"],
            "test_average_mae": test["average"]["mae"],
            "test_homo_mae": test["HOMO"]["mae"],
            "test_lumo_mae": test["LUMO"]["mae"],
            "test_gap_mae": test["Gap"]["mae"],
        }
    baseline = arms[str(baseline_dimension)]
    comparisons = {}
    for dimension, candidate in arms.items():
        if int(dimension) == baseline_dimension:
            continue
        delta = {
            "test_average_mae": candidate["test_average_mae"] - baseline["test_average_mae"],
            "test_gap_mae": candidate["test_gap_mae"] - baseline["test_gap_mae"],
            "parameter_ratio": candidate["n_params"] / baseline["n_params"],
            "training_time_ratio": candidate["training_time_s"] / baseline["training_time_s"],
        }
        delta["screening_pass"] = (
            delta["test_average_mae"] <= 0.005
            and (delta["parameter_ratio"] <= 0.90 or delta["training_time_ratio"] <= 0.85)
        )
        comparisons[dimension] = delta
    passing = [
        int(dimension) for dimension, comparison in comparisons.items()
        if comparison["screening_pass"]
    ]
    summary = {
        "arms": arms,
        "baseline_dimension": baseline_dimension,
        "candidate_comparisons": comparisons,
        "passing_dimensions": passing,
        "recommended_dimension": min(passing) if passing else None,
        "screening_rule": "average MAE delta <= 0.005 eV and params <= 90% or time <= 85%",
    }
    atomic_json_write(summary, out_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="SchNet 192D vs 128D on one fixed 50K subset")
    parser.add_argument("--source-graphs", type=Path, default=DEFAULT_GRAPHS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--sample-size", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--dimensions", type=int, nargs="+", default=[192, 160, 128])
    parser.add_argument("--baseline-dimension", type=int, default=192)
    args = parser.parse_args()

    ensure_dirs(args.out_dir)
    subset_path = args.out_dir / "graphs_50k.pt"
    manifest_path = args.out_dir / "graphs_50k_manifest.json"
    build_subset(
        args.source_graphs, subset_path, manifest_path,
        args.sample_size, args.seed,
    )
    if args.baseline_dimension not in args.dimensions:
        parser.error("--baseline-dimension must be included in --dimensions")
    results = {
        dimension: run_arm(args, subset_path, dimension)
        for dimension in args.dimensions
    }
    summary = summarize(results, args.baseline_dimension, args.out_dir / "summary.json")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
