"""Evaluate the archive-r01 learned-router Oracle ceiling before training a router."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from molgap.constants import MODELS_DIR, RAW_DIR, RESULTS_DIR, SEED
from molgap.fusion import FusionHead
from molgap.router import DEFAULT_TARGET_WEIGHTS, oracle_router_analysis, router_descriptor_row
from molgap.utils import ensure_dirs, load_aligned_encoder_embeddings


PHASE8 = RESULTS_DIR / "phase8"
OUT_DIR = PHASE8 / "archive" / "archive-r01-learned-router"
TARGETS = ["homo", "lumo", "gap"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=float, default=4.0)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max-internal", type=int, default=None)
    parser.add_argument("--skip-descriptors", action="store_true")
    parser.add_argument(
        "--raw-csv", type=Path, default=RAW_DIR / "phase8_expansion_500k.csv"
    )
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
        "--fusion-expert", type=Path,
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
    parser.add_argument("--metrics-out", type=Path, default=OUT_DIR / "oracle_metrics.json")
    parser.add_argument("--regions-out", type=Path, default=OUT_DIR / "oracle_regions.csv")
    parser.add_argument(
        "--predictions-out", type=Path, default=OUT_DIR / "oracle_predictions.parquet"
    )
    parser.add_argument("--decision-out", type=Path, default=OUT_DIR / "oracle_decision.md")
    return parser.parse_args()


@torch.no_grad()
def load_internal_predictions(args: argparse.Namespace, device: torch.device):
    h2, h3, y, source_idx = load_aligned_encoder_embeddings(
        [args.emb_base, args.emb_extra], args.emb_3d, args.graphs_3d
    )
    permutation = np.random.RandomState(SEED).permutation(len(y))
    test_idx = permutation[int(0.9 * len(permutation)):]
    if args.max_internal is not None:
        test_idx = test_idx[:args.max_internal]
    h2, h3, y, source_idx = h2[test_idx], h3[test_idx], y[test_idx], source_idx[test_idx]

    base = FusionHead("gate", 192, 0.0, dim_2d=192, dim_3d=192).to(device)
    base.load_state_dict(torch.load(args.fusion_base, weights_only=True, map_location=device))
    base.eval()
    expert = FusionHead("gate", 192, 0.0, dim_2d=384, dim_3d=192).to(device)
    expert.load_state_dict(
        torch.load(args.fusion_expert, weights_only=True, map_location=device)
    )
    expert.eval()

    base_pred, expert_pred = [], []
    for batch_2d, batch_3d in DataLoader(
        TensorDataset(h2, h3), batch_size=2048, shuffle=False
    ):
        batch_2d, batch_3d = batch_2d.to(device), batch_3d.to(device)
        base_pred.append(base(batch_2d[:, :192], batch_3d).float().cpu())
        expert_pred.append(expert(batch_2d, batch_3d).float().cpu())

    raw = pd.read_csv(
        args.raw_csv,
        usecols=lambda name: name in {"cid", "smiles", "canonical_smiles", "mw"},
    )
    rows = raw.iloc[source_idx.numpy()].copy().reset_index(drop=True)
    rows.insert(0, "source_idx", source_idx.numpy())
    return rows, y.numpy(), torch.cat(base_pred).numpy(), torch.cat(expert_pred).numpy()


def load_common_predictions(args: argparse.Namespace):
    frame = pd.read_csv(args.common_predictions)
    y = frame[TARGETS].to_numpy(dtype=np.float64)
    base = frame[[f"expansion500k_full_hybrid_{t}" for t in TARGETS]].to_numpy(
        dtype=np.float64
    )
    expert = frame[[f"expansion500k_dualgps_hybrid_{t}" for t in TARGETS]].to_numpy(
        dtype=np.float64
    )
    keep = [name for name in ("eval_set", "bucket", "cid", "smiles", "canonical_smiles", "mw")
            if name in frame]
    return frame[keep].copy(), y, base, expert


def load_pcqm_predictions(args: argparse.Namespace):
    frame = pd.read_csv(args.pcqm_predictions)
    y = frame["gap_true"].to_numpy(dtype=np.float64)[:, None]
    base = frame["v3_gap_pred"].to_numpy(dtype=np.float64)[:, None]
    expert = frame["dualgps_gap_pred"].to_numpy(dtype=np.float64)[:, None]
    keep = [name for name in ("pcqm_idx", "smiles", "canonical_smiles") if name in frame]
    return frame[keep].copy(), y, base, expert


def evaluate_frame(
    frame: pd.DataFrame,
    y: np.ndarray,
    base: np.ndarray,
    expert: np.ndarray,
    *,
    dataset: str,
    args: argparse.Namespace,
):
    target_names = TARGETS if y.shape[1] == 3 else ["gap"]
    weights = DEFAULT_TARGET_WEIGHTS if y.shape[1] == 3 else [1.0]
    fixed_route = base[:, target_names.index("gap")] < args.threshold
    metrics, arrays = oracle_router_analysis(
        y,
        base,
        expert,
        fixed_route,
        target_names=target_names,
        weights=weights,
        n_bootstrap=args.bootstrap,
        seed=args.seed,
    )

    out = frame.reset_index(drop=True).copy()
    out.insert(0, "dataset", dataset)
    for i, target in enumerate(target_names):
        out[f"y_{target}"] = y[:, i]
        out[f"base_{target}"] = base[:, i]
        out[f"expert_{target}"] = expert[:, i]
    for name, values in arrays.items():
        out[name] = values
    out["expert_wins"] = out["gain"] > 0
    return metrics, out


def add_descriptors(frame: pd.DataFrame) -> pd.DataFrame:
    smiles = frame["canonical_smiles"].fillna(frame["smiles"]).tolist()
    rows = []
    for i, value in enumerate(smiles, start=1):
        rows.append(router_descriptor_row(value))
        if i % 10_000 == 0:
            print(f"Descriptors: {i}/{len(smiles)}", flush=True)
    descriptors = pd.DataFrame(rows)
    for column in descriptors:
        frame[column] = descriptors[column].to_numpy()
    return frame


def region_rows(frame: pd.DataFrame) -> list[dict[str, object]]:
    specs = {
        "base_gap": ([-np.inf, 2, 3, 4, 5, np.inf], ["<2", "2-3", "3-4", "4-5", ">=5"]),
        "mw": ([-np.inf, 200, 300, 400, 500, np.inf], ["<200", "200-300", "300-400", "400-500", ">=500"]),
        "aromatic_rings": ([-np.inf, 0.5, 1.5, 2.5, 4.5, np.inf], ["0", "1", "2", "3-4", ">=5"]),
        "rotatable_bonds": ([-np.inf, 0.5, 3.5, 6.5, 10.5, np.inf], ["0", "1-3", "4-6", "7-10", ">=11"]),
        "conjugated_bonds": ([-np.inf, 2.5, 5.5, 9.5, 14.5, np.inf], ["0-2", "3-5", "6-9", "10-14", ">=15"]),
    }
    rows: list[dict[str, object]] = []
    for dataset, dataset_frame in frame.groupby("dataset", sort=False):
        group_specs = dict(specs)
        if "eval_set" in dataset_frame and dataset_frame["eval_set"].notna().any():
            for value, group in dataset_frame.groupby("eval_set", dropna=True):
                rows.append(_summarize_region(dataset, "eval_set", str(value), group))
        for feature, (bins, labels) in group_specs.items():
            if feature not in dataset_frame:
                continue
            categories = pd.cut(dataset_frame[feature], bins=bins, labels=labels, right=False)
            for label in labels:
                group = dataset_frame[categories == label]
                if len(group):
                    rows.append(_summarize_region(dataset, feature, label, group))
    return rows


def _summarize_region(
    dataset: str, feature: str, label: str, frame: pd.DataFrame
) -> dict[str, object]:
    fixed = frame["fixed_route"].astype(bool).to_numpy()
    wins = frame["expert_wins"].astype(bool).to_numpy()
    routed_wins = int(np.count_nonzero(fixed & wins))
    return {
        "dataset": dataset,
        "feature": feature,
        "bin": label,
        "n": int(len(frame)),
        "base_loss": float(frame["base_loss"].mean()),
        "expert_loss": float(frame["expert_loss"].mean()),
        "fixed_loss": float(frame["fixed_loss"].mean()),
        "mean_gain": float(frame["gain"].mean()),
        "expert_win_rate": float(wins.mean()),
        "fixed_route_rate": float(fixed.mean()),
        "fixed_precision": float(routed_wins / fixed.sum()) if fixed.sum() else None,
        "fixed_recall": float(routed_wins / wins.sum()) if wins.sum() else None,
    }


def build_decision(metrics: dict[str, object], args: argparse.Namespace) -> tuple[dict, str]:
    internal = metrics["datasets"]["internal"]
    common = metrics["datasets"]["common_all"]
    internal_gap = -internal["bootstrap"]["oracle_minus_fixed_gap"]["delta"]
    common_gap = -common["bootstrap"]["oracle_minus_fixed_gap"]["delta"]
    budget_internal_gap = -internal["bootstrap"]["budget_oracle_minus_fixed_gap"]["delta"]
    if internal_gap > 0.0015:
        verdict = "go"
        reason = "Oracle leaves more than 0.0015 eV additional internal Gap improvement."
    elif internal_gap >= 0.0005:
        verdict = "research_only"
        reason = "Oracle headroom is measurable but below the formal-router priority threshold."
    else:
        verdict = "stop"
        reason = "Oracle leaves less than 0.0005 eV additional internal Gap improvement."
    decision = {
        "verdict": verdict,
        "reason": reason,
        "internal_oracle_extra_gap_improvement": float(internal_gap),
        "internal_budget_oracle_extra_gap_improvement": float(budget_internal_gap),
        "common_oracle_extra_gap_improvement": float(common_gap),
        "next_step": (
            "Build the leakage-controlled router development table and train feature ablations."
            if verdict == "go" else
            "Keep the fixed Gap<4 eV v4 router."
        ),
    }

    lines = [
        "# archive-r01 Learned Router Oracle Decision",
        "",
        "Frozen models: v3 base and dual-GPS expert. Objective weights: HOMO/LUMO/Gap = 0.25/0.25/0.50.",
        f"Fixed control: route when base Gap is `< {args.threshold:g} eV`.",
        "",
        "| evaluation | n | fixed route | expert wins | fixed precision | fixed recall | fixed Gap MAE | budget Oracle Gap MAE | Oracle Gap MAE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("internal", "common_all", "common_ood1000", "common_p8_targeted_hard", "pcqm_proxy"):
        block = metrics["datasets"][name]
        fixed_diag = block["routes"]["fixed"]["delta_0"]
        methods = block["methods"]
        lines.append(
            f"| {name} | {block['n']} | {fixed_diag['route_fraction']:.1%} | "
            f"{block['expert_win_rates']['gain_gt_0']:.1%} | "
            f"{fixed_diag['precision']:.1%} | {fixed_diag['recall']:.1%} | "
            f"{methods['fixed']['gap']['mae']:.6f} | "
            f"{methods['budget_oracle']['gap']['mae']:.6f} | "
            f"{methods['oracle']['gap']['mae']:.6f} |"
        )
    lines.extend([
        "",
        f"**Decision: {verdict.upper()}.** {reason}",
        f"Internal unrestricted Oracle adds `{internal_gap:.6f} eV` Gap improvement over v4; "
        f"the same-budget Oracle adds `{budget_internal_gap:.6f} eV`.",
        "",
        decision["next_step"],
    ])
    return decision, "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    ensure_dirs(args.metrics_out.parent, args.regions_out.parent, args.predictions_out.parent)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    print("Loading internal held-out predictions", flush=True)
    internal_frame, y, base, expert = load_internal_predictions(args, device)
    internal_metrics, internal_rows = evaluate_frame(
        internal_frame, y, base, expert, dataset="internal", args=args
    )

    print("Loading common predictions", flush=True)
    common_frame, y, base, expert = load_common_predictions(args)
    common_metrics, common_rows = evaluate_frame(
        common_frame, y, base, expert, dataset="common_all", args=args
    )
    common_blocks = {"common_all": common_metrics}
    for eval_set in ("ood1000", "p8_targeted_hard"):
        mask = common_frame["eval_set"].eq(eval_set).to_numpy()
        block_metrics, _ = evaluate_frame(
            common_frame.loc[mask].reset_index(drop=True),
            y[mask], base[mask], expert[mask], dataset=f"common_{eval_set}", args=args,
        )
        common_blocks[f"common_{eval_set}"] = block_metrics

    print("Loading PCQM proxy predictions", flush=True)
    pcqm_frame, y, base, expert = load_pcqm_predictions(args)
    pcqm_metrics, pcqm_rows = evaluate_frame(
        pcqm_frame, y, base, expert, dataset="pcqm_proxy", args=args
    )

    predictions = pd.concat([internal_rows, common_rows, pcqm_rows], ignore_index=True)
    if not args.skip_descriptors:
        predictions = add_descriptors(predictions)
        pd.DataFrame(region_rows(predictions)).to_csv(args.regions_out, index=False)
    predictions.to_parquet(args.predictions_out, index=False)

    metrics = {
        "experiment": "archive-r01 learned-router Oracle ceiling",
        "frozen_models": {"base": str(args.fusion_base), "expert": str(args.fusion_expert)},
        "fixed_threshold_eV": float(args.threshold),
        "objective_weights": {"homo": 0.25, "lumo": 0.25, "gap": 0.50},
        "bootstrap_draws": int(args.bootstrap),
        "datasets": {
            "internal": internal_metrics,
            **common_blocks,
            "pcqm_proxy": pcqm_metrics,
        },
        "artifacts": {
            "predictions": str(args.predictions_out),
            "regions": None if args.skip_descriptors else str(args.regions_out),
        },
    }
    decision, markdown = build_decision(metrics, args)
    metrics["decision"] = decision
    args.metrics_out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    args.decision_out.write_text(markdown, encoding="utf-8")
    print(json.dumps(decision, indent=2), flush=True)
    print(f"Metrics -> {args.metrics_out}", flush=True)
    print(f"Decision -> {args.decision_out}", flush=True)


if __name__ == "__main__":
    main()
