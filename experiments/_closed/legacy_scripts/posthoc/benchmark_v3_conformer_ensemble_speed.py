"""Benchmark v3 single-conformer vs k-conformer Hybrid inference speed."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from molgap.constants import RESULTS_DIR
from molgap.inference import (
    load_hybrid,
    predict_smiles_batch_hybrid,
    predict_smiles_batch_hybrid_conformer_ensemble,
)

PHASE8 = RESULTS_DIR / "phase8"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--common-csv", type=Path, default=PHASE8 / "full_expansion500k_common_eval_predictions.csv")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--k", type=int, default=8)
    archive = PHASE8 / "archive" / "legacy" / "conformer_ensemble"
    parser.add_argument("--out-json", type=Path, default=archive / "v3_conformer_ensemble_speed.json")
    parser.add_argument("--out-md", type=Path, default=archive / "v3_conformer_ensemble_speed.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.common_csv).head(args.n)
    smiles = df["smiles"].astype(str).tolist()

    print("Loading v3 hybrid once", flush=True)
    models = load_hybrid(key="phase8_expansion_hybrid")

    print(f"Benchmark single conformer on n={len(smiles)}", flush=True)
    t0 = time.perf_counter()
    vi_single, pred_single = predict_smiles_batch_hybrid(smiles, models=models)
    t_single = time.perf_counter() - t0

    print(f"Benchmark k={args.k} conformer ensemble on n={len(smiles)}", flush=True)
    t0 = time.perf_counter()
    vi_ens, pred_ens, pred_std, n_confs = predict_smiles_batch_hybrid_conformer_ensemble(
        smiles, models=models, k=args.k
    )
    t_ens = time.perf_counter() - t0

    result = {
        "hybrid_key": "phase8_expansion_hybrid",
        "n_input": int(len(smiles)),
        "k": int(args.k),
        "single": {
            "n_valid": int(len(vi_single)),
            "wall_s": float(t_single),
            "s_per_input_mol": float(t_single / max(len(smiles), 1)),
            "s_per_valid_mol": float(t_single / max(len(vi_single), 1)),
        },
        "ensemble": {
            "n_valid": int(len(vi_ens)),
            "wall_s": float(t_ens),
            "s_per_input_mol": float(t_ens / max(len(smiles), 1)),
            "s_per_valid_mol": float(t_ens / max(len(vi_ens), 1)),
            "mean_conformers": float(n_confs.mean()) if len(n_confs) else 0.0,
        },
    }
    result["speed_factor_valid_mol"] = (
        result["ensemble"]["s_per_valid_mol"] / max(result["single"]["s_per_valid_mol"], 1e-9)
    )

    args.out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = [
        "# Phase 8 v3 Conformer Ensemble Speed",
        "",
        "Date: 2026-07-06",
        "",
        f"- Input molecules: `{result['n_input']}`",
        f"- Ensemble k: `{result['k']}`",
        f"- Single conformer: `{result['single']['wall_s']:.2f}` s total, `{result['single']['s_per_valid_mol']:.3f}` s/valid mol",
        f"- k-conformer ensemble: `{result['ensemble']['wall_s']:.2f}` s total, `{result['ensemble']['s_per_valid_mol']:.3f}` s/valid mol",
        f"- Slowdown: `{result['speed_factor_valid_mol']:.1f}x` per valid molecule",
        "",
        "Decision: keep conformer ensemble as opt-in small/medium-batch inference, not the database-scale default.",
    ]
    args.out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
