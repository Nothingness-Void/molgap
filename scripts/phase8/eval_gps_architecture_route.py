"""Validate the fixed Gap<4 eV routed dual-GPS architecture on held-out data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from molgap.constants import MODELS_DIR, RESULTS_DIR, SEED
from molgap.fusion import FusionHead
from molgap.utils import load_aligned_encoder_embeddings


PHASE8 = RESULTS_DIR / "phase8"
TARGETS = ["homo", "lumo", "gap"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=float, default=4.0)
    parser.add_argument(
        "--emb-base", type=Path, default=PHASE8 / "gps_expansion_500k_embeddings.pt"
    )
    parser.add_argument(
        "--emb-extra", type=Path, default=PHASE8 / "gps_arch_depth9_embeddings.pt"
    )
    parser.add_argument(
        "--emb-3d", type=Path, default=PHASE8 / "schnet_expansion_500k_embeddings.pt"
    )
    parser.add_argument(
        "--graphs-3d", type=Path,
        default=PHASE8 / "pyg_3d_graphs_etkdg_expansion_500k.pt",
    )
    parser.add_argument(
        "--fusion-base", type=Path,
        default=MODELS_DIR / "phase8_hybrid_fusion_expansion_500k.pt",
    )
    parser.add_argument(
        "--fusion-dual", type=Path,
        default=MODELS_DIR / "phase8_hybrid_fusion_expansion_500k_dualgps.pt",
    )
    parser.add_argument(
        "--common-predictions", type=Path,
        default=PHASE8 / "gps_arch_dualgps_common_eval_predictions.csv",
    )
    parser.add_argument(
        "--pcqm-predictions", type=Path,
        default=PHASE8 / "gps_arch_dualgps_pcqm_proxy_predictions.csv",
    )
    parser.add_argument(
        "--metrics-out", type=Path, default=PHASE8 / "gps_arch_routed_metrics.json"
    )
    parser.add_argument(
        "--decision-out", type=Path, default=PHASE8 / "gps_arch_routed_decision.md"
    )
    return parser.parse_args()


def bootstrap(delta: np.ndarray, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    n = len(delta)
    draws = np.empty(10000, dtype=np.float64)
    for i in range(len(draws)):
        draws[i] = float(delta[rng.integers(0, n, n)].mean())
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return {
        "delta": float(delta.mean()),
        "ci95": [float(lo), float(hi)],
        "probability_better": float((draws < 0).mean()),
    }


def metric_block(y: np.ndarray, base: np.ndarray, routed: np.ndarray) -> dict:
    base_err = np.abs(base - y)
    routed_err = np.abs(routed - y)
    result = {"n": int(len(y))}
    for i, target in enumerate(TARGETS):
        result[target] = {
            "base_mae": float(base_err[:, i].mean()),
            "routed_mae": float(routed_err[:, i].mean()),
            **bootstrap(routed_err[:, i] - base_err[:, i]),
        }
    result["average"] = {
        "base_mae": float(base_err.mean()),
        "routed_mae": float(routed_err.mean()),
        **bootstrap(routed_err.mean(axis=1) - base_err.mean(axis=1)),
    }
    return result


@torch.no_grad()
def internal_test(args: argparse.Namespace, device: torch.device) -> dict:
    h2, h3, y, _ = load_aligned_encoder_embeddings(
        [args.emb_base, args.emb_extra], args.emb_3d, args.graphs_3d
    )
    idx = np.random.RandomState(SEED).permutation(len(y))
    test_idx = idx[int(0.9 * len(idx)):]
    h2, h3, y = h2[test_idx], h3[test_idx], y[test_idx]

    base = FusionHead("gate", 192, 0.0, dim_2d=192, dim_3d=192).to(device)
    base.load_state_dict(torch.load(args.fusion_base, weights_only=True, map_location=device))
    base.eval()
    dual = FusionHead("gate", 192, 0.0, dim_2d=384, dim_3d=192).to(device)
    dual.load_state_dict(torch.load(args.fusion_dual, weights_only=True, map_location=device))
    dual.eval()

    base_pred, dual_pred = [], []
    loader = DataLoader(TensorDataset(h2, h3), batch_size=2048, shuffle=False)
    for batch_2d, batch_3d in loader:
        batch_2d, batch_3d = batch_2d.to(device), batch_3d.to(device)
        base_pred.append(base(batch_2d[:, :192], batch_3d).cpu())
        dual_pred.append(dual(batch_2d, batch_3d).cpu())
    base_pred = torch.cat(base_pred).numpy()
    dual_pred = torch.cat(dual_pred).numpy()
    y = y.numpy()
    route = base_pred[:, 2] < args.threshold
    routed = base_pred.copy()
    routed[route] = dual_pred[route]
    return {"route_n": int(route.sum()), "metrics": metric_block(y, base_pred, routed)}


def external_common(args: argparse.Namespace) -> dict:
    df = pd.read_csv(args.common_predictions)
    y = df[TARGETS].to_numpy(dtype=np.float64)
    base = df[[f"expansion500k_full_hybrid_{t}" for t in TARGETS]].to_numpy(dtype=np.float64)
    dual = df[[f"expansion500k_dualgps_hybrid_{t}" for t in TARGETS]].to_numpy(dtype=np.float64)
    route = base[:, 2] < args.threshold
    routed = base.copy()
    routed[route] = dual[route]
    blocks = {}
    for name in ["all", "ood1000", "p8_targeted_hard"]:
        mask = np.ones(len(df), dtype=bool) if name == "all" else df["eval_set"].eq(name).to_numpy()
        blocks[name] = metric_block(y[mask], base[mask], routed[mask])
        blocks[name]["route_n"] = int(route[mask].sum())
    return blocks


def external_pcqm(args: argparse.Namespace) -> dict:
    df = pd.read_csv(args.pcqm_predictions)
    y = df["gap_true"].to_numpy(dtype=np.float64)
    base = df["v3_gap_pred"].to_numpy(dtype=np.float64)
    dual = df["dualgps_gap_pred"].to_numpy(dtype=np.float64)
    route = base < args.threshold
    routed = base.copy()
    routed[route] = dual[route]
    base_err = np.abs(base - y)
    routed_err = np.abs(routed - y)
    return {
        "n": int(len(df)),
        "route_n": int(route.sum()),
        "base_gap_mae": float(base_err.mean()),
        "routed_gap_mae": float(routed_err.mean()),
        **bootstrap(routed_err - base_err),
    }


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = {
        "threshold_eV": args.threshold,
        "rule": "use dual-GPS hybrid when base v3 predicted Gap is below threshold",
        "internal_test": internal_test(args, device),
        "common_eval": external_common(args),
        "pcqm_proxy": external_pcqm(args),
    }
    internal_gap = result["internal_test"]["metrics"]["gap"]
    common_gap = result["common_eval"]["all"]["gap"]
    ood_gap = result["common_eval"]["ood1000"]["gap"]
    pcqm_gap = result["pcqm_proxy"]
    promoted = bool(
        internal_gap["ci95"][1] < 0
        and common_gap["ci95"][1] < 0
        and ood_gap["ci95"][1] < 0
        and pcqm_gap["ci95"][0] <= 0
    )
    result["decision"] = {
        "promote_as_v4_accuracy_predictor": promoted,
        "keep_v3_component_loader": True,
        "reason": (
            "Independent internal/common/OOD gains are significant and PCQM does not "
            "significantly regress."
            if promoted else
            "At least one promotion gate failed."
        ),
    }
    args.metrics_out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    internal = result["internal_test"]["metrics"]
    common = result["common_eval"]
    pcqm = result["pcqm_proxy"]
    lines = [
        "# Phase 8 Routed Dual-GPS Architecture Decision",
        "",
        f"Rule: use the dual-GPS hybrid only when the base v3 predicted Gap is `< {args.threshold:g} eV`.",
        "Training data and SchNet are unchanged.",
        "",
        "| evaluation | routed n / n | avg MAE delta | Gap MAE delta | Gap 95% CI |",
        "|---|---:|---:|---:|---:|",
        f"| internal held-out test | {result['internal_test']['route_n']} / {internal['n']} | "
        f"{internal['average']['delta']:+.6f} | {internal['gap']['delta']:+.6f} | "
        f"[{internal['gap']['ci95'][0]:+.6f}, {internal['gap']['ci95'][1]:+.6f}] |",
    ]
    for name in ["all", "ood1000", "p8_targeted_hard"]:
        block = common[name]
        lines.append(
            f"| common {name} | {block['route_n']} / {block['n']} | "
            f"{block['average']['delta']:+.6f} | {block['gap']['delta']:+.6f} | "
            f"[{block['gap']['ci95'][0]:+.6f}, {block['gap']['ci95'][1]:+.6f}] |"
        )
    lines.extend([
        f"| PCQM proxy | {pcqm['route_n']} / {pcqm['n']} | n/a | {pcqm['delta']:+.6f} | "
        f"[{pcqm['ci95'][0]:+.6f}, {pcqm['ci95'][1]:+.6f}] |",
        "",
        "",
        (
            "**Decision: positive. Promote as the v4 B3LYP accuracy predictor; keep the "
            "v3 single hybrid as the component/compatibility loader.**"
            if promoted else
            "**Decision: negative. Keep v3 as the selected B3LYP predictor.**"
        ),
        "The next gate is re-running Phase 9/10 Delta and UQ against routed v4 outputs.",
    ])
    args.decision_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    print(f"Decision -> {args.decision_out}", flush=True)


if __name__ == "__main__":
    main()
