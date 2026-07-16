"""Train archive-r02 feature ablations and lock conservative policies on validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from molgap.constants import RESULTS_DIR
from molgap.router import paired_bootstrap_mean
from molgap.archive.phase8_r01_router.router_training import (
    apply_conservative_policy, fit_router_bundle, optimize_gap_threshold,
    predict_router_bundle, router_feature_sets, select_conservative_policies,
)
from molgap.utils import ensure_dirs


OUT_DIR = RESULTS_DIR / "phase8" / "archive" / "archive-r02-pubchemqc-router"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=OUT_DIR / "router_development_dataset.parquet")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR / "router_seed42")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--win-delta", type=float, default=0.002)
    parser.add_argument("--max-route-fraction", type=float, default=0.25)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    return parser.parse_args()


def evaluate(frame: pd.DataFrame, route: np.ndarray, fixed: np.ndarray, bootstrap: int, seed: int):
    y = frame.gap.to_numpy(dtype=np.float64)
    base = frame.base_gap.to_numpy(dtype=np.float64)
    expert = frame.expert_gap.to_numpy(dtype=np.float64)
    errors = np.where(route, np.abs(expert - y), np.abs(base - y))
    fixed_errors = np.where(fixed, np.abs(expert - y), np.abs(base - y))
    gain = np.abs(base - y) - np.abs(expert - y)
    routed_gain = gain[route]
    return {
        "n": len(frame),
        "gap_mae": float(errors.mean()),
        "fixed_gap_mae": float(fixed_errors.mean()),
        "gap_delta_vs_fixed": float((errors - fixed_errors).mean()),
        "bootstrap_gap_vs_fixed": paired_bootstrap_mean(
            errors - fixed_errors, n_bootstrap=bootstrap, seed=seed
        ),
        "route_n": int(route.sum()),
        "route_fraction": float(route.mean()),
        "expert_win_precision": float(np.mean(routed_gain > 0)) if len(routed_gain) else None,
        "expert_meaningful_win_precision": float(np.mean(routed_gain > 0.002)) if len(routed_gain) else None,
        "downside_p95": float(np.quantile(np.maximum(-routed_gain, 0), 0.95)) if len(routed_gain) else None,
        "downside_p99": float(np.quantile(np.maximum(-routed_gain, 0), 0.99)) if len(routed_gain) else None,
    }


def main() -> None:
    args = parse_args()
    ensure_dirs(args.out_dir)
    table = pd.read_parquet(args.dataset)
    train = table[table.split == "train"].reset_index(drop=True)
    validation = table[table.split == "validation"].reset_index(drop=True)
    test = table[table.split == "dev_test"].reset_index(drop=True)
    feature_results, fitted = {}, {}
    for name, features in router_feature_sets(table.columns).items():
        print(f"fit {name} ({len(features)} features)", flush=True)
        bundle = fit_router_bundle(
            train, validation, features, seed=args.seed, win_delta=args.win_delta
        )
        val_predictions = predict_router_bundle(bundle, validation)
        policies = select_conservative_policies(
            validation, val_predictions, max_route_fraction=args.max_route_fraction
        )
        feature_results[name] = {
            "n_features": len(features),
            "features": features,
            "ood_threshold": bundle.ood_threshold,
            "validation_policies": policies,
        }
        fitted[name] = bundle

    best_loss = min(
        block["validation_policies"]["bidirectional"]["weighted_gap_mae"]
        for block in feature_results.values()
    )
    tied = [
        name for name, block in feature_results.items()
        if block["validation_policies"]["bidirectional"]["weighted_gap_mae"] <= best_loss + 0.0001
    ]
    selected_name = min(tied, key=lambda name: feature_results[name]["n_features"])
    bundle = fitted[selected_name]
    test_predictions = predict_router_bundle(bundle, test)
    fixed_test = test.fixed_route_flag.to_numpy(dtype=bool)
    test_results, prediction_columns = {}, {}
    for offset, (strategy, policy) in enumerate(
        feature_results[selected_name]["validation_policies"].items()
    ):
        route = apply_conservative_policy(test_predictions, fixed_test, policy)
        test_results[strategy] = evaluate(
            test, route, fixed_test, args.bootstrap, args.seed + offset
        )
        prediction_columns[f"route_{strategy}"] = route

    gap_policy = optimize_gap_threshold(validation)
    gap_route = test.base_gap.to_numpy() < gap_policy["threshold_eV"]
    test_results["optimized_gap_threshold"] = {
        **gap_policy,
        **evaluate(test, gap_route, fixed_test, args.bootstrap, args.seed + 10),
    }
    test_results["fixed_v4"] = evaluate(
        test, fixed_test, fixed_test, args.bootstrap, args.seed + 11
    )
    predictions = test[["probe_idx", "cid", "sampling_source", "gap", "base_gap", "expert_gap"]].copy()
    for name, values in test_predictions.items():
        predictions[name] = values
    for name, values in prediction_columns.items():
        predictions[name] = values
    predictions["route_optimized_gap"] = gap_route
    predictions["route_fixed"] = fixed_test
    predictions.to_parquet(args.out_dir / "dev_test_predictions.parquet", index=False)
    joblib.dump(bundle, args.out_dir / "router_bundle.pkl")
    result = {
        "seed": args.seed,
        "win_delta_eV": args.win_delta,
        "selection": "R0-R3 and policy thresholds selected on weighted scaffold validation only",
        "feature_ablation": feature_results,
        "selected_feature_set": selected_name,
        "selected_features": bundle.features,
        "validation_gap_threshold_baseline": gap_policy,
        "dev_test": test_results,
        "sealed_metrics_opened": False,
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"selected": selected_name, "dev_test": test_results}, indent=2), flush=True)


if __name__ == "__main__":
    main()
