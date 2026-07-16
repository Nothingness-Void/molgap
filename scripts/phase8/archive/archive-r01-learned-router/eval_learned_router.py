"""Evaluate the locked archive-r01 learned Router on external B3LYP benchmarks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from molgap.constants import RESULTS_DIR, SEED
from molgap.inference import (
    load_routed_dual_gps_hybrid,
    predict_smiles_batch_dual_gps_candidates,
)
from molgap.router import (
    DEFAULT_TARGET_WEIGHTS,
    apply_utility_policy,
    oracle_router_analysis,
    route_policy_metrics,
    router_descriptor_row,
    select_top_budget,
)
from molgap.utils import ensure_dirs


PHASE8 = RESULTS_DIR / "phase8"
OUT_DIR = PHASE8 / "archive" / "archive-r01-learned-router"
TARGETS = ["homo", "lumo", "gap"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--common-csv", type=Path,
        default=PHASE8 / "gps_arch_dualgps_common_eval_predictions.csv",
    )
    parser.add_argument(
        "--pcqm-csv", type=Path,
        default=PHASE8 / "gps_arch_dualgps_pcqm_proxy_predictions.csv",
    )
    parser.add_argument("--schema", type=Path, default=OUT_DIR / "feature_schema.json")
    parser.add_argument("--thresholds", type=Path, default=OUT_DIR / "validation_thresholds.json")
    parser.add_argument("--gain-model", type=Path, default=OUT_DIR / "router_gain_model.txt")
    parser.add_argument("--win-model", type=Path, default=OUT_DIR / "router_win_model.txt")
    parser.add_argument("--downside-model", type=Path, default=OUT_DIR / "router_downside_model.txt")
    parser.add_argument("--calibration", type=Path, default=OUT_DIR / "calibration.pkl")
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--metrics-out", type=Path, default=OUT_DIR / "external_metrics.json")
    parser.add_argument("--predictions-out", type=Path, default=OUT_DIR / "external_predictions.parquet")
    parser.add_argument("--decision-out", type=Path, default=OUT_DIR / "decision.md")
    return parser.parse_args()


def load_booster(path: Path) -> lgb.Booster:
    return lgb.Booster(model_str=path.read_text(encoding="utf-8"))


def build_router_features(
    metadata: pd.DataFrame,
    base: np.ndarray,
    gps: np.ndarray,
    schnet: np.ndarray,
) -> pd.DataFrame:
    smiles = metadata["canonical_smiles"].fillna(metadata["smiles"]).tolist()
    descriptors = pd.DataFrame([router_descriptor_row(value) for value in smiles])
    features = descriptors.copy()
    for i, target in enumerate(TARGETS):
        features[f"base_{target}"] = base[:, i]
        features[f"gps_{target}"] = gps[:, i]
        features[f"schnet_{target}"] = schnet[:, i]
        features[f"abs_gps_schnet_{target}"] = np.abs(gps[:, i] - schnet[:, i])
    features["gap_consistency_signed"] = base[:, 2] - (base[:, 1] - base[:, 0])
    features["gap_consistency_abs"] = features["gap_consistency_signed"].abs()
    features["fixed_route_flag"] = base[:, 2] < 4.0
    features["fixed_route_margin"] = 4.0 - base[:, 2]
    return features


def predict_dataset(
    source: pd.DataFrame,
    models: dict,
    *,
    seed: int,
    dataset: str,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    output = predict_smiles_batch_dual_gps_candidates(
        source["smiles"].tolist(), models=models, random_seed=seed
    )
    valid_idx, base, expert, _, _, gps, schnet = output
    metadata = source.iloc[valid_idx].copy().reset_index(drop=True)
    features = build_router_features(metadata, base, gps, schnet)
    metadata.insert(0, "dataset", dataset)
    return metadata, base, expert, valid_idx, features


def learned_scores(features, feature_names, gain_model, downside_model, win_model, calibrator):
    x = features[feature_names].to_numpy(dtype=np.float32)
    gain = gain_model.predict(x)
    downside = np.maximum(downside_model.predict(x), 0.0)
    probability = calibrator.predict(win_model.predict(x))
    return gain, downside, probability


def evaluate_block(y, base, expert, fixed, learned, matched, target_names, weights, args):
    fixed_metrics = route_policy_metrics(
        y, base, expert, fixed, target_names=target_names, weights=weights
    )
    learned_metrics = route_policy_metrics(
        y, base, expert, learned,
        target_names=target_names, weights=weights, reference_route=fixed,
        n_bootstrap=args.bootstrap, seed=args.seed,
    )
    matched_metrics = route_policy_metrics(
        y, base, expert, matched,
        target_names=target_names, weights=weights, reference_route=fixed,
        n_bootstrap=args.bootstrap, seed=args.seed + 10,
    )
    oracle_metrics, _ = oracle_router_analysis(
        y, base, expert, fixed,
        target_names=target_names, weights=weights,
        n_bootstrap=args.bootstrap, seed=args.seed + 20,
    )
    return {
        "fixed": fixed_metrics,
        "learned_capped": learned_metrics,
        "learned_budget_matched": matched_metrics,
        "oracle": oracle_metrics,
    }


def append_prediction_columns(
    metadata, y, base, expert, fixed, learned, matched, gain, downside, probability, utility,
    target_names,
):
    out = metadata.copy()
    for i, target in enumerate(target_names):
        out[f"y_{target}"] = y[:, i]
        out[f"base_{target}"] = base[:, i]
        out[f"expert_{target}"] = expert[:, i]
    out["fixed_route"] = fixed
    out["learned_capped_route"] = learned
    out["learned_budget_matched_route"] = matched
    out["predicted_gain"] = gain
    out["predicted_downside"] = downside
    out["p_expert_wins"] = probability
    out["utility"] = utility
    return out


def main() -> None:
    args = parse_args()
    ensure_dirs(args.metrics_out.parent)
    schema = json.loads(args.schema.read_text(encoding="utf-8"))
    feature_names = schema["features"]
    policy = json.loads(args.thresholds.read_text(encoding="utf-8"))
    gain_model = load_booster(args.gain_model)
    win_model = load_booster(args.win_model)
    downside_model = load_booster(args.downside_model)
    calibrator = joblib.load(args.calibration)
    models = load_routed_dual_gps_hybrid()

    print("Recomputing common base/expert candidates with seeded ETKDG", flush=True)
    common_source = pd.read_csv(args.common_csv)
    common_meta, common_base, common_expert, common_valid, common_features = predict_dataset(
        common_source, models, seed=args.seed, dataset="common_all"
    )
    common_y = common_source.iloc[common_valid][TARGETS].to_numpy(dtype=np.float64)
    common_gain, common_downside, common_probability = learned_scores(
        common_features, feature_names, gain_model, downside_model, win_model, calibrator
    )
    common_learned, common_utility = apply_utility_policy(
        common_gain, common_downside, common_probability, policy
    )
    common_fixed = common_base[:, 2] < 4.0
    common_matched = select_top_budget(common_utility, int(common_fixed.sum()))
    datasets = {
        "common_all": evaluate_block(
            common_y, common_base, common_expert, common_fixed, common_learned,
            common_matched, TARGETS, DEFAULT_TARGET_WEIGHTS, args,
        )
    }
    for eval_set in ("ood1000", "p8_targeted_hard"):
        mask = common_meta["eval_set"].eq(eval_set).to_numpy()
        datasets[f"common_{eval_set}"] = evaluate_block(
            common_y[mask], common_base[mask], common_expert[mask], common_fixed[mask],
            common_learned[mask], common_matched[mask], TARGETS,
            DEFAULT_TARGET_WEIGHTS, args,
        )

    common_rows = append_prediction_columns(
        common_meta, common_y, common_base, common_expert, common_fixed,
        common_learned, common_matched, common_gain, common_downside,
        common_probability, common_utility, TARGETS,
    )

    print("Recomputing PCQM proxy base/expert candidates with seeded ETKDG", flush=True)
    pcqm_source = pd.read_csv(args.pcqm_csv)
    pcqm_meta, pcqm_base_all, pcqm_expert_all, pcqm_valid, pcqm_features = predict_dataset(
        pcqm_source, models, seed=args.seed + 100_000, dataset="pcqm_proxy"
    )
    pcqm_y = pcqm_source.iloc[pcqm_valid]["gap_true"].to_numpy(dtype=np.float64)[:, None]
    pcqm_gain, pcqm_downside, pcqm_probability = learned_scores(
        pcqm_features, feature_names, gain_model, downside_model, win_model, calibrator
    )
    pcqm_learned, pcqm_utility = apply_utility_policy(
        pcqm_gain, pcqm_downside, pcqm_probability, policy
    )
    pcqm_fixed = pcqm_base_all[:, 2] < 4.0
    pcqm_matched = select_top_budget(pcqm_utility, int(pcqm_fixed.sum()))
    pcqm_base = pcqm_base_all[:, 2:3]
    pcqm_expert = pcqm_expert_all[:, 2:3]
    datasets["pcqm_proxy"] = evaluate_block(
        pcqm_y, pcqm_base, pcqm_expert, pcqm_fixed, pcqm_learned,
        pcqm_matched, ["gap"], [1.0], args,
    )
    pcqm_rows = append_prediction_columns(
        pcqm_meta, pcqm_y, pcqm_base, pcqm_expert, pcqm_fixed,
        pcqm_learned, pcqm_matched, pcqm_gain, pcqm_downside,
        pcqm_probability, pcqm_utility, ["gap"],
    )

    candidate = "learned_budget_matched"
    common_ci = datasets["common_all"][candidate]["bootstrap_gap_vs_reference"]["ci95"]
    ood_ci = datasets["common_ood1000"][candidate]["bootstrap_gap_vs_reference"]["ci95"]
    hard_ci = datasets["common_p8_targeted_hard"][candidate]["bootstrap_gap_vs_reference"]["ci95"]
    pcqm_ci = datasets["pcqm_proxy"][candidate]["bootstrap_gap_vs_reference"]["ci95"]
    internal = json.loads((OUT_DIR / "ablation_metrics.json").read_text(encoding="utf-8"))[
        "selected"
    ]["test_budget_policy"]
    promoted = bool(
        internal["bootstrap_gap_vs_reference"]["ci95"][1] < 0
        and common_ci[1] < 0
        and hard_ci[1] < 0
        and ood_ci[0] <= 0
        and pcqm_ci[0] <= 0
        and -internal["bootstrap_gap_vs_reference"]["delta"] >= 0.001
    )
    result = {
        "feature_set": schema["selected_feature_set"],
        "features": feature_names,
        "policy": policy,
        "evaluation_note": "Base, expert, and branch features recomputed together with seeded ETKDG.",
        "datasets": datasets,
        "decision": {
            "promote": promoted,
            "candidate": candidate,
            "reason": (
                "Internal/common/hard improve and OOD/PCQM do not significantly regress."
                if promoted else
                "At least one external promotion gate failed; keep fixed v4."
            ),
        },
    }
    args.metrics_out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    pd.concat([common_rows, pcqm_rows], ignore_index=True).to_parquet(
        args.predictions_out, index=False
    )

    lines = [
        "# archive-r01 Learned Router External Decision",
        "",
        f"Selected features: `{schema['selected_feature_set']}` ({len(feature_names)} features).",
        "Candidate below is budget-matched to the fixed Gap<4 route on each full evaluation set.",
        "",
        "| evaluation | fixed route | learned route | fixed precision | learned precision | weighted delta | Gap delta | Gap 95% CI |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("common_all", "common_ood1000", "common_p8_targeted_hard", "pcqm_proxy"):
        block = datasets[name]
        fixed = block["fixed"]
        learned = block[candidate]
        gap = learned["bootstrap_gap_vs_reference"]
        lines.append(
            f"| {name} | {fixed['route']['route_fraction']:.1%} | "
            f"{learned['route']['route_fraction']:.1%} | "
            f"{fixed['route']['precision']:.1%} | {learned['route']['precision']:.1%} | "
            f"{learned['weighted_delta_vs_reference']:+.6f} | {gap['delta']:+.6f} | "
            f"[{gap['ci95'][0]:+.6f}, {gap['ci95'][1]:+.6f}] |"
        )
    lines.extend([
        "",
        (
            "**Decision: promote to the speed/reproducibility gate.**"
            if promoted else
            "**Decision: do not promote. Keep fixed routed-v4 as the default.**"
        ),
        result["decision"]["reason"],
    ])
    args.decision_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result["decision"], indent=2), flush=True)
    print(f"Decision -> {args.decision_out}", flush=True)


if __name__ == "__main__":
    main()
