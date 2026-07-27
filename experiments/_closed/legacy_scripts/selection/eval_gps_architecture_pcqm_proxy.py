"""Evaluate the fixed-data dual-GPS candidate on the saved PCQM4Mv2 proxy.

The input is the existing leakage-filtered v1/v2/v3 prediction table. This keeps
the molecule sample and v3 control predictions fixed and computes only the new
architecture candidate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from molgap.constants import MODELS_DIR, PARAMS_GPS_2D, PARAMS_SCHNET_300K, RESULTS_DIR
from molgap.fusion import FusionHead
from molgap.gps import GPSWrapper
from molgap.inference import predict_smiles_batch_hybrid
from molgap.schnet import SchNetWrapper


PHASE8 = RESULTS_DIR / "phase8"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-predictions",
        type=Path,
        default=PHASE8 / "pcqm4mv2_proxy_p7_v2_v3_predictions.csv",
    )
    parser.add_argument(
        "--gps-base", type=Path, default=MODELS_DIR / "phase8_gps_expansion_500k.pt"
    )
    parser.add_argument(
        "--gps-extra",
        type=Path,
        default=MODELS_DIR / "phase8_gps_expansion_500k_depth9.pt",
    )
    parser.add_argument("--extra-layers", type=int, default=9)
    parser.add_argument(
        "--schnet", type=Path, default=MODELS_DIR / "phase8_schnet_expansion_500k.pt"
    )
    parser.add_argument(
        "--fusion",
        type=Path,
        default=MODELS_DIR / "phase8_hybrid_fusion_expansion_500k_dualgps.pt",
    )
    parser.add_argument(
        "--metrics-out", type=Path, default=PHASE8 / "gps_arch_dualgps_pcqm_proxy_metrics.json"
    )
    parser.add_argument(
        "--predictions-out",
        type=Path,
        default=PHASE8 / "gps_arch_dualgps_pcqm_proxy_predictions.csv",
    )
    return parser.parse_args()


def load_models(args: argparse.Namespace, device: torch.device):
    gps_base = GPSWrapper(**PARAMS_GPS_2D).to(device)
    gps_base.load_state_dict(torch.load(args.gps_base, weights_only=True, map_location=device))
    gps_base.eval()

    extra_params = dict(PARAMS_GPS_2D)
    extra_params["num_layers"] = args.extra_layers
    gps_extra = GPSWrapper(**extra_params).to(device)
    gps_extra.load_state_dict(torch.load(args.gps_extra, weights_only=True, map_location=device))
    gps_extra.eval()

    schnet = SchNetWrapper(**PARAMS_SCHNET_300K, use_charges=True).to(device)
    schnet.load_state_dict(torch.load(args.schnet, weights_only=True, map_location=device))
    schnet.eval()

    fusion = FusionHead("gate", 192, 0.0, dim_2d=384, dim_3d=192).to(device)
    fusion.load_state_dict(torch.load(args.fusion, weights_only=True, map_location=device))
    fusion.eval()
    return [gps_base, gps_extra], schnet, fusion, device


def paired_bootstrap(base_err: np.ndarray, candidate_err: np.ndarray) -> dict:
    rng = np.random.default_rng(42)
    n = len(base_err)
    draws = np.empty(10000, dtype=np.float64)
    for i in range(len(draws)):
        idx = rng.integers(0, n, n)
        draws[i] = float((candidate_err[idx] - base_err[idx]).mean())
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return {
        "delta_mae": float((candidate_err - base_err).mean()),
        "ci95": [float(lo), float(hi)],
        "probability_candidate_better": float((draws < 0).mean()),
        "n_bootstrap": int(len(draws)),
        "seed": 42,
    }


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = pd.read_csv(args.base_predictions)
    required = {"pcqm_idx", "smiles", "gap_true", "v3_gap_pred"}
    missing = sorted(required - set(base.columns))
    if missing:
        raise ValueError(f"Missing columns in {args.base_predictions}: {missing}")

    models = load_models(args, device)
    valid_idx, pred = predict_smiles_batch_hybrid(base["smiles"].tolist(), models=models)
    candidate = base.iloc[valid_idx].copy().reset_index(drop=True)
    candidate["dualgps_homo_pred"] = pred[:, 0]
    candidate["dualgps_lumo_pred"] = pred[:, 1]
    candidate["dualgps_gap_pred"] = pred[:, 2]
    candidate["dualgps_gap_abs_err"] = np.abs(
        candidate["dualgps_gap_pred"] - candidate["gap_true"]
    )
    candidate["v3_gap_abs_err"] = np.abs(candidate["v3_gap_pred"] - candidate["gap_true"])

    base_err = candidate["v3_gap_abs_err"].to_numpy(dtype=np.float64)
    dual_err = candidate["dualgps_gap_abs_err"].to_numpy(dtype=np.float64)
    metrics = {
        "note": "Fixed leakage-filtered PCQM4Mv2 proxy; not an OGB submission.",
        "n_input": int(len(base)),
        "n_valid": int(len(candidate)),
        "v3_gap_mae": float(base_err.mean()),
        "dualgps_gap_mae": float(dual_err.mean()),
        "paired_bootstrap": paired_bootstrap(base_err, dual_err),
    }
    args.metrics_out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    candidate.to_csv(args.predictions_out, index=False)
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"Metrics -> {args.metrics_out}", flush=True)
    print(f"Predictions -> {args.predictions_out}", flush=True)


if __name__ == "__main__":
    main()
