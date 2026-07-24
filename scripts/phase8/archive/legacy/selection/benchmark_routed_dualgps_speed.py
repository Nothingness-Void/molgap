"""Benchmark the routed dual-GPS candidate against the v3 single-GPS hybrid."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from molgap.constants import RESULTS_DIR, SEED
from molgap.inference import (
    load_routed_dual_gps_hybrid,
    predict_smiles_batch_hybrid,
    predict_smiles_batch_routed_dual_gps,
)


PHASE8 = RESULTS_DIR / "phase8"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--common-csv",
        type=Path,
        default=PHASE8 / "archive" / "legacy" / "pilots_30k" / "common_eval_30k_predictions.csv",
    )
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--json-out", type=Path, default=PHASE8 / "gps_arch_routed_speed.json"
    )
    parser.add_argument(
        "--md-out", type=Path, default=PHASE8 / "gps_arch_routed_speed.md"
    )
    return parser.parse_args()


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.common_csv)
    sample = df.sample(min(args.n, len(df)), random_state=args.seed).reset_index(drop=True)
    smiles = sample["smiles"].tolist()

    routed_models = load_routed_dual_gps_hybrid()
    base_models = (
        routed_models["base_gps"],
        routed_models["encoder_3d"],
        routed_models["base_fusion"],
        routed_models["device"],
    )

    # Exclude model loading and one-time CUDA kernel initialization.
    predict_smiles_batch_hybrid(smiles[:1], models=base_models)
    predict_smiles_batch_routed_dual_gps(smiles[:1], models=routed_models)

    def time_base():
        sync()
        start = time.perf_counter()
        output = predict_smiles_batch_hybrid(smiles, models=base_models)
        sync()
        return time.perf_counter() - start, output

    def time_routed():
        sync()
        start = time.perf_counter()
        output = predict_smiles_batch_routed_dual_gps(smiles, models=routed_models)
        sync()
        return time.perf_counter() - start, output

    base_times, routed_times = [], []
    base_idx = routed_idx = routed_mask = None
    for repeat in range(args.repeats):
        if repeat % 2 == 0:
            base_seconds, (base_idx, _) = time_base()
            routed_seconds, (routed_idx, _, routed_mask) = time_routed()
        else:
            routed_seconds, (routed_idx, _, routed_mask) = time_routed()
            base_seconds, (base_idx, _) = time_base()
        base_times.append(base_seconds)
        routed_times.append(routed_seconds)

    base_seconds = float(np.median(base_times))
    routed_seconds = float(np.median(routed_times))

    if base_idx.tolist() != routed_idx.tolist():
        raise RuntimeError("Base and routed APIs produced different valid indices")
    n_valid = len(base_idx)
    result = {
        "n_requested": int(len(smiles)),
        "n_valid": int(n_valid),
        "n_routed": int(routed_mask.sum()),
        "route_fraction": float(routed_mask.mean()) if n_valid else 0.0,
        "repeats": int(args.repeats),
        "base_seconds_all": [float(value) for value in base_times],
        "routed_seconds_all": [float(value) for value in routed_times],
        "base_seconds": float(base_seconds),
        "routed_seconds": float(routed_seconds),
        "base_seconds_per_valid": float(base_seconds / n_valid),
        "routed_seconds_per_valid": float(routed_seconds / n_valid),
        "slowdown": float(routed_seconds / base_seconds),
        "model_load_excluded": True,
        "sample_seed": int(args.seed),
    }
    args.json_out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    args.md_out.write_text(
        "# Phase 8 Routed Dual-GPS Speed\n\n"
        "Model loading is excluded. Times include ETKDG graph construction; "
        f"reported times are medians of {args.repeats} alternating-order repeats.\n\n"
        "| mode | valid | routed | seconds | seconds/valid |\n"
        "|---|---:|---:|---:|---:|\n"
        f"| v3 base | {n_valid} | 0 | {base_seconds:.3f} | {base_seconds/n_valid:.5f} |\n"
        f"| routed dual-GPS | {n_valid} | {int(routed_mask.sum())} | "
        f"{routed_seconds:.3f} | {routed_seconds/n_valid:.5f} |\n\n"
        f"Slowdown: **{routed_seconds/base_seconds:.2f}x**; "
        f"route fraction: **{routed_mask.mean():.1%}**.\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
