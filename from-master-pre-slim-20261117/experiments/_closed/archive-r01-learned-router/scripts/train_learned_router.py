"""Train and select the archive-r01 learned Router with feature ablations."""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from molgap.constants import RESULTS_DIR, SEED
from molgap.router import (
    DEFAULT_TARGET_WEIGHTS,
    apply_utility_policy,
    per_molecule_loss,
    route_policy_metrics,
    select_top_budget,
)
from molgap.utils import ensure_dirs


OUT_DIR = RESULTS_DIR / "phase8" / "archive" / "archive-r01-learned-router"
TARGETS = ["homo", "lumo", "gap"]
BASE_FEATURES = ["base_homo", "base_lumo", "base_gap"]
CONSISTENCY_FEATURES = [
    "gap_consistency_signed", "gap_consistency_abs", "fixed_route_flag",
    "fixed_route_margin",
]
DESCRIPTOR_FEATURES = [
    "mw", "heavy_atoms", "ring_count", "aromatic_rings", "rotatable_bonds",
    "tpsa", "logp", "fraction_csp3", "hbd", "hba", "formal_charge",
    "conjugated_bonds", "aromatic_atom_fraction", "n_N", "n_O", "n_S",
    "n_F", "n_Cl", "n_Br", "n_B", "n_P", "n_Si",
]
BRANCH_FEATURES = [
    f"{prefix}_{target}"
    for target in TARGETS
    for prefix in ("gps", "schnet", "abs_gps_schnet")
]
PCA_FEATURES = [
    *[f"gps_pca_{i:02d}" for i in range(1, 17)],
    *[f"schnet_pca_{i:02d}" for i in range(1, 17)],
    "gps_embedding_norm", "schnet_embedding_norm",
]
PROTOTYPE_FEATURES = [
    "gps_prototype_min_distance", "gps_prototype_5mean_distance",
    "gps_prototype_distance_ratio", "gps_prototype_over_p95",
    "schnet_prototype_min_distance", "schnet_prototype_5mean_distance",
    "schnet_prototype_distance_ratio", "schnet_prototype_over_p95",
]


def feature_sets() -> dict[str, list[str]]:
    r0 = ["base_gap"]
    r1 = BASE_FEATURES + CONSISTENCY_FEATURES
    r2 = r1 + DESCRIPTOR_FEATURES
    r3 = r2 + BRANCH_FEATURES
    r4 = r3 + PCA_FEATURES
    r5 = r4 + PROTOTYPE_FEATURES
    return {"R0": r0, "R1": r1, "R2": r2, "R3": r3, "R4": r4, "R5": r5}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=OUT_DIR / "router_dataset.parquet")
    parser.add_argument("--win-delta", type=float, default=0.001)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--metrics-out", type=Path, default=OUT_DIR / "ablation_metrics.json")
    parser.add_argument("--predictions-out", type=Path, default=OUT_DIR / "router_predictions.parquet")
    parser.add_argument("--schema-out", type=Path, default=OUT_DIR / "feature_schema.json")
    parser.add_argument("--thresholds-out", type=Path, default=OUT_DIR / "validation_thresholds.json")
    parser.add_argument("--gain-model-out", type=Path, default=OUT_DIR / "router_gain_model.txt")
    parser.add_argument("--win-model-out", type=Path, default=OUT_DIR / "router_win_model.txt")
    parser.add_argument("--downside-model-out", type=Path, default=OUT_DIR / "router_downside_model.txt")
    parser.add_argument("--calibration-out", type=Path, default=OUT_DIR / "calibration.pkl")
    return parser.parse_args()


def lgb_regressor(seed: int, objective: str = "huber") -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective=objective,
        n_estimators=1500,
        learning_rate=0.025,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=40,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )


def lgb_classifier(seed: int) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=1500,
        learning_rate=0.025,
        num_leaves=31,
        min_child_samples=40,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )


def fit_models(x_train, x_val, gain_train, gain_val, win_train, win_val, seed):
    callbacks = [lgb.early_stopping(75, verbose=False)]
    gain_model = lgb_regressor(seed)
    gain_model.fit(
        x_train, gain_train, eval_set=[(x_val, gain_val)], eval_metric="l1",
        callbacks=callbacks,
    )
    downside_model = lgb_regressor(seed + 1)
    downside_model.fit(
        x_train, np.maximum(-gain_train, 0.0),
        eval_set=[(x_val, np.maximum(-gain_val, 0.0))], eval_metric="l1",
        callbacks=callbacks,
    )
    win_model = lgb_classifier(seed + 2)
    win_model.fit(
        x_train, win_train, eval_set=[(x_val, win_val)], eval_metric="binary_logloss",
        callbacks=callbacks,
    )
    return gain_model, downside_model, win_model


def classification_metrics(y_true: np.ndarray, probability: np.ndarray) -> dict[str, object]:
    frac_pos, mean_pred = calibration_curve(y_true, probability, n_bins=10, strategy="quantile")
    return {
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "average_precision": float(average_precision_score(y_true, probability)),
        "brier": float(brier_score_loss(y_true, probability)),
        "calibration": {
            "mean_predicted": mean_pred.tolist(),
            "fraction_positive": frac_pos.tolist(),
        },
    }


def regression_metrics(y_true: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, prediction)),
        "r2": float(r2_score(y_true, prediction)),
        "spearman": float(spearmanr(y_true, prediction).statistic),
    }


def arrays(frame: pd.DataFrame):
    y = frame[[f"y_{target}" for target in TARGETS]].to_numpy(dtype=np.float64)
    base = frame[[f"base_{target}" for target in TARGETS]].to_numpy(dtype=np.float64)
    expert = frame[[f"expert_{target}" for target in TARGETS]].to_numpy(dtype=np.float64)
    fixed = frame["fixed_route_flag"].to_numpy(dtype=bool)
    return y, base, expert, fixed


def route_loss(frame: pd.DataFrame, route: np.ndarray) -> float:
    y, base, expert, _ = arrays(frame)
    pred = base.copy()
    pred[route] = expert[route]
    return float(per_molecule_loss(y, pred, DEFAULT_TARGET_WEIGHTS).mean())


def choose_validation_policy(
    frame: pd.DataFrame,
    pred_gain: np.ndarray,
    pred_downside: np.ndarray,
    probability: np.ndarray,
) -> dict[str, float]:
    max_routes = int(frame["fixed_route_flag"].sum())
    candidates = []
    for alpha in (0.0, 0.25, 0.5, 1.0):
        utility = pred_gain - alpha * pred_downside
        thresholds = {0.0}
        for fraction in (0.10, 0.20, max_routes / len(frame), 0.30):
            n = min(len(frame) - 1, max(1, int(round(fraction * len(frame)))))
            thresholds.add(float(np.partition(utility, len(utility) - n)[len(utility) - n]))
        for p_min in (0.0, 0.50, 0.60, 0.70, 0.80):
            for threshold in thresholds:
                route = (utility >= threshold) & (probability >= p_min)
                if route.sum() > max_routes:
                    eligible = np.flatnonzero(route)
                    keep = eligible[np.argsort(-utility[eligible], kind="stable")[:max_routes]]
                    route[:] = False
                    route[keep] = True
                candidates.append({
                    "alpha": alpha,
                    "p_min": p_min,
                    "utility_threshold": threshold,
                    "max_route_fraction": float(max_routes / len(frame)),
                    "route_n": int(route.sum()),
                    "route_fraction": float(route.mean()),
                    "weighted_loss": route_loss(frame, route),
                })
    return min(candidates, key=lambda row: (row["weighted_loss"], row["route_n"]))


def main() -> None:
    args = parse_args()
    warnings.filterwarnings(
        "ignore",
        message="X does not have valid feature names, but LGBM.* was fitted with feature names",
    )
    ensure_dirs(args.metrics_out.parent)
    frame = pd.read_parquet(args.dataset)
    split_frames = {
        name: frame.loc[frame["split"].eq(name)].copy().reset_index(drop=True)
        for name in ("train", "validation", "test")
    }
    train, val, test = (split_frames[name] for name in ("train", "validation", "test"))
    gain_train, gain_val, gain_test = (part["gain"].to_numpy() for part in (train, val, test))
    win_train, win_val, win_test = (
        gain > args.win_delta for gain in (gain_train, gain_val, gain_test)
    )

    result: dict[str, object] = {
        "win_delta_eV": float(args.win_delta),
        "selection_rule": "feature set and utility policy selected on scaffold validation only",
        "feature_sets": {},
    }
    fitted = {}
    prediction_rows = []
    for name, features in feature_sets().items():
        print(f"Training {name} ({len(features)} features)", flush=True)
        x_train = train[features].to_numpy(dtype=np.float32)
        x_val = val[features].to_numpy(dtype=np.float32)
        x_test = test[features].to_numpy(dtype=np.float32)
        gain_model, downside_model, win_model = fit_models(
            x_train, x_val, gain_train, gain_val, win_train, win_val, args.seed
        )
        raw_val_probability = win_model.predict_proba(x_val)[:, 1]
        calibrator = IsotonicRegression(out_of_bounds="clip").fit(
            raw_val_probability, win_val.astype(float)
        )
        val_probability = calibrator.predict(raw_val_probability)
        test_probability = calibrator.predict(win_model.predict_proba(x_test)[:, 1])
        val_gain = gain_model.predict(x_val)
        test_gain = gain_model.predict(x_test)
        val_downside = np.maximum(downside_model.predict(x_val), 0.0)
        test_downside = np.maximum(downside_model.predict(x_test), 0.0)

        policy = choose_validation_policy(val, val_gain, val_downside, val_probability)
        val_route, val_utility = apply_utility_policy(
            val_gain, val_downside, val_probability, policy
        )
        test_route, test_utility = apply_utility_policy(
            test_gain, test_downside, test_probability, policy
        )
        budget_test_route = select_top_budget(
            test_utility, int(test["fixed_route_flag"].sum())
        )

        logistic = make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, random_state=args.seed)
        )
        logistic.fit(x_train, win_train)
        logistic_test_probability = logistic.predict_proba(x_test)[:, 1]
        logistic_route = select_top_budget(
            logistic_test_probability, int(test["fixed_route_flag"].sum())
        )

        y_val, base_val, expert_val, fixed_val = arrays(val)
        y_test, base_test, expert_test, fixed_test = arrays(test)
        block = {
            "n_features": len(features),
            "features": features,
            "best_iterations": {
                "gain": int(gain_model.best_iteration_),
                "downside": int(downside_model.best_iteration_),
                "win": int(win_model.best_iteration_),
            },
            "validation": {
                "gain": regression_metrics(gain_val, val_gain),
                "win": classification_metrics(win_val, val_probability),
                "policy": policy,
                "route": route_policy_metrics(
                    y_val, base_val, expert_val, val_route,
                    target_names=TARGETS, weights=DEFAULT_TARGET_WEIGHTS,
                    reference_route=fixed_val,
                ),
            },
            "test": {
                "gain": regression_metrics(gain_test, test_gain),
                "win": classification_metrics(win_test, test_probability),
                "threshold_policy": route_policy_metrics(
                    y_test, base_test, expert_test, test_route,
                    target_names=TARGETS, weights=DEFAULT_TARGET_WEIGHTS,
                    reference_route=fixed_test,
                ),
                "budget_policy": route_policy_metrics(
                    y_test, base_test, expert_test, budget_test_route,
                    target_names=TARGETS, weights=DEFAULT_TARGET_WEIGHTS,
                    reference_route=fixed_test,
                ),
                "logistic_budget_policy": route_policy_metrics(
                    y_test, base_test, expert_test, logistic_route,
                    target_names=TARGETS, weights=DEFAULT_TARGET_WEIGHTS,
                    reference_route=fixed_test,
                ),
                "logistic_win": classification_metrics(win_test, logistic_test_probability),
            },
        }
        result["feature_sets"][name] = block
        fitted[name] = (gain_model, downside_model, win_model, calibrator, policy, features)
        prediction_rows.append(pd.DataFrame({
            "source_idx": test["source_idx"],
            "feature_set": name,
            "gain": gain_test,
            "predicted_gain": test_gain,
            "predicted_downside": test_downside,
            "p_expert_wins": test_probability,
            "utility": test_utility,
            "threshold_route": test_route,
            "budget_route": budget_test_route,
            "fixed_route": fixed_test,
        }))

    validation_losses = {
        name: block["validation"]["policy"]["weighted_loss"]
        for name, block in result["feature_sets"].items()
    }
    best_loss = min(validation_losses.values())
    practically_tied = [
        name for name, loss in validation_losses.items()
        if loss <= best_loss + 0.0001
    ]
    best_name = min(
        practically_tied,
        key=lambda name: result["feature_sets"][name]["n_features"],
    )
    gain_model, downside_model, win_model, calibrator, policy, features = fitted[best_name]
    best_test = result["feature_sets"][best_name]["test"]
    # Run the expensive paired bootstrap only once, after validation locks the winner.
    test_frame = split_frames["test"]
    x_test = test_frame[features].to_numpy(dtype=np.float32)
    probability = calibrator.predict(win_model.predict_proba(x_test)[:, 1])
    route, utility = apply_utility_policy(
        gain_model.predict(x_test),
        np.maximum(downside_model.predict(x_test), 0.0),
        probability,
        policy,
    )
    y_test, base_test, expert_test, fixed_test = arrays(test_frame)
    best_test["threshold_policy"] = route_policy_metrics(
        y_test, base_test, expert_test, route,
        target_names=TARGETS, weights=DEFAULT_TARGET_WEIGHTS,
        reference_route=fixed_test, n_bootstrap=args.bootstrap, seed=args.seed,
    )
    budget_route = select_top_budget(utility, int(fixed_test.sum()))
    best_test["budget_policy"] = route_policy_metrics(
        y_test, base_test, expert_test, budget_route,
        target_names=TARGETS, weights=DEFAULT_TARGET_WEIGHTS,
        reference_route=fixed_test, n_bootstrap=args.bootstrap, seed=args.seed + 10,
    )

    result["selected"] = {
        "feature_set": best_name,
        "selection_tolerance_eV": 0.0001,
        "numerical_best_feature_set": min(validation_losses, key=validation_losses.get),
        "features": features,
        "validation_policy": policy,
        "test_threshold_policy": best_test["threshold_policy"],
        "test_budget_policy": best_test["budget_policy"],
    }
    # LightGBM's native Windows writer cannot open non-ASCII paths reliably.
    args.gain_model_out.write_text(gain_model.booster_.model_to_string(), encoding="utf-8")
    args.win_model_out.write_text(win_model.booster_.model_to_string(), encoding="utf-8")
    args.downside_model_out.write_text(
        downside_model.booster_.model_to_string(), encoding="utf-8"
    )
    joblib.dump(calibrator, args.calibration_out)
    args.schema_out.write_text(json.dumps({
        "selected_feature_set": best_name,
        "features": features,
        "all_feature_sets": feature_sets(),
        "win_delta_eV": args.win_delta,
    }, indent=2), encoding="utf-8")
    args.thresholds_out.write_text(json.dumps(policy, indent=2), encoding="utf-8")
    pd.concat(prediction_rows, ignore_index=True).to_parquet(args.predictions_out, index=False)
    args.metrics_out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["selected"], indent=2), flush=True)
    print(f"Metrics -> {args.metrics_out}", flush=True)


if __name__ == "__main__":
    main()
