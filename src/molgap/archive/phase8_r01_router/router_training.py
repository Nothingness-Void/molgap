"""Training and conservative policy selection for learned Expert routing."""

from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler


TARGETS = ("homo", "lumo", "gap")
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
    *[f"gps_pca_{index:02d}" for index in range(1, 17)],
    *[f"schnet_pca_{index:02d}" for index in range(1, 17)],
    "gps_embedding_norm", "schnet_embedding_norm",
]
PROTOTYPE_FEATURES = [
    "gps_prototype_min_distance", "gps_prototype_5mean_distance",
    "gps_prototype_distance_ratio", "gps_prototype_over_p95",
    "schnet_prototype_min_distance", "schnet_prototype_5mean_distance",
    "schnet_prototype_distance_ratio", "schnet_prototype_over_p95",
]


def router_feature_sets(available_columns=None) -> dict[str, list[str]]:
    r0 = ["base_gap"]
    r1 = BASE_FEATURES + CONSISTENCY_FEATURES
    r2 = r1 + DESCRIPTOR_FEATURES
    r3 = r2 + BRANCH_FEATURES
    r4 = r3 + PCA_FEATURES
    r5 = r4 + PROTOTYPE_FEATURES
    result = {"R0": r0, "R1": r1, "R2": r2, "R3": r3}
    available = set(() if available_columns is None else available_columns)
    if all(feature in available for feature in PCA_FEATURES):
        result["R4"] = r4
    if all(feature in available for feature in PROTOTYPE_FEATURES):
        result["R5"] = r5
    return result


def _regressor(seed: int) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective="huber", n_estimators=1500, learning_rate=0.025,
        num_leaves=31, min_child_samples=40, subsample=0.85,
        colsample_bytree=0.85, reg_alpha=0.1, reg_lambda=1.0,
        random_state=seed, n_jobs=-1, verbosity=-1,
    )


def _classifier(seed: int) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary", n_estimators=1500, learning_rate=0.025,
        num_leaves=31, min_child_samples=40, subsample=0.85,
        colsample_bytree=0.85, reg_alpha=0.1, reg_lambda=1.0,
        random_state=seed, n_jobs=-1, verbosity=-1,
    )


@dataclass
class RouterModelBundle:
    features: list[str]
    gain_model: object
    downside_model: object
    win_model: object
    calibrator: object
    ood_scaler: object
    ood_covariance: object
    ood_threshold: float
    win_delta: float
    seed: int


def fit_router_bundle(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    features: list[str],
    *,
    seed: int,
    win_delta: float = 0.002,
) -> RouterModelBundle:
    x_train = train[features].to_numpy(dtype=np.float32)
    x_val = validation[features].to_numpy(dtype=np.float32)
    gain_train = train.gain.to_numpy(dtype=np.float64)
    gain_val = validation.gain.to_numpy(dtype=np.float64)
    weight_train = train.training_weight.to_numpy(dtype=np.float64)
    weight_val = validation.training_weight.to_numpy(dtype=np.float64)
    callbacks = [lgb.early_stopping(75, verbose=False)]

    gain_model = _regressor(seed)
    gain_model.fit(
        x_train, gain_train, sample_weight=weight_train,
        eval_set=[(x_val, gain_val)], eval_sample_weight=[weight_val],
        eval_metric="l1", callbacks=callbacks,
    )
    downside_model = _regressor(seed + 1)
    downside_train = np.maximum(-gain_train, 0.0)
    downside_val = np.maximum(-gain_val, 0.0)
    downside_model.fit(
        x_train, downside_train, sample_weight=weight_train,
        eval_set=[(x_val, downside_val)], eval_sample_weight=[weight_val],
        eval_metric="l1", callbacks=callbacks,
    )
    win_model = _classifier(seed + 2)
    win_train = gain_train > win_delta
    win_val = gain_val > win_delta
    win_model.fit(
        x_train, win_train, sample_weight=weight_train,
        eval_set=[(x_val, win_val)], eval_sample_weight=[weight_val],
        eval_metric="binary_logloss", callbacks=callbacks,
    )
    raw_val = win_model.predict_proba(x_val)[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip").fit(
        raw_val, win_val.astype(float), sample_weight=weight_val
    )

    scaler = StandardScaler().fit(x_train)
    scaled_train = scaler.transform(x_train)
    covariance = LedoitWolf().fit(scaled_train)
    train_distance = np.sqrt(np.maximum(covariance.mahalanobis(scaled_train), 0.0))
    ood_threshold = float(np.quantile(train_distance, 0.95))
    return RouterModelBundle(
        features=features, gain_model=gain_model, downside_model=downside_model,
        win_model=win_model, calibrator=calibrator, ood_scaler=scaler,
        ood_covariance=covariance, ood_threshold=ood_threshold,
        win_delta=win_delta, seed=seed,
    )


def predict_router_bundle(
    bundle: RouterModelBundle, frame: pd.DataFrame
) -> dict[str, np.ndarray]:
    x = frame[bundle.features].to_numpy(dtype=np.float32)
    raw_probability = bundle.win_model.predict_proba(x)[:, 1]
    scaled = bundle.ood_scaler.transform(x)
    distance = np.sqrt(np.maximum(bundle.ood_covariance.mahalanobis(scaled), 0.0))
    return {
        "predicted_gain": bundle.gain_model.predict(x),
        "predicted_downside": np.maximum(bundle.downside_model.predict(x), 0.0),
        "p_expert_win": bundle.calibrator.predict(raw_probability),
        "ood_distance": distance,
        "ood_fallback": distance > bundle.ood_threshold,
    }


def _cap_route(
    route: np.ndarray, utility: np.ndarray, fallback: np.ndarray,
    fixed: np.ndarray, max_fraction: float,
) -> np.ndarray:
    max_routes = int(round(max_fraction * len(route)))
    if route.sum() <= max_routes:
        return route
    locked = fallback & fixed
    available = np.flatnonzero(route & ~locked)
    keep_n = max(0, max_routes - int(locked.sum()))
    result = locked.copy()
    if keep_n:
        chosen = available[np.argsort(-utility[available], kind="stable")[:keep_n]]
        result[chosen] = True
    return result


def apply_conservative_policy(
    predictions: dict[str, np.ndarray], fixed: np.ndarray, policy: dict
) -> np.ndarray:
    gain = predictions["predicted_gain"]
    downside = predictions["predicted_downside"]
    probability = predictions["p_expert_win"]
    fallback = predictions["ood_fallback"]
    utility = gain - policy["alpha"] * downside
    add = (utility >= policy["add_utility"]) & (probability >= policy["add_probability"])
    keep = (utility >= policy["keep_utility"]) & (probability >= policy["keep_probability"])
    strategy = policy["strategy"]
    if strategy == "full_replacement":
        route = add
    elif strategy == "suppress_only":
        route = fixed & keep
    elif strategy == "add_only":
        route = fixed | (~fixed & add)
    elif strategy == "bidirectional":
        route = (fixed & keep) | (~fixed & add)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    route = np.where(fallback, fixed, route).astype(bool)
    return _cap_route(route, utility, fallback, fixed, policy["max_route_fraction"])


def _weighted_gap_loss(frame: pd.DataFrame, route: np.ndarray) -> float:
    y = frame.gap.to_numpy(dtype=np.float64)
    base = frame.base_gap.to_numpy(dtype=np.float64)
    expert = frame.expert_gap.to_numpy(dtype=np.float64)
    errors = np.where(route, np.abs(expert - y), np.abs(base - y))
    return float(np.average(errors, weights=frame.training_weight))


def select_conservative_policies(
    validation: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    *,
    max_route_fraction: float = 0.25,
) -> dict[str, dict]:
    fixed = validation.fixed_route_flag.to_numpy(dtype=bool)
    gain = predictions["predicted_gain"]
    downside = predictions["predicted_downside"]
    probability = predictions["p_expert_win"]
    candidates: dict[str, list[dict]] = {
        name: [] for name in ("full_replacement", "suppress_only", "add_only", "bidirectional")
    }
    for alpha in (0.0, 0.5, 1.0, 2.0):
        utility = gain - alpha * downside
        thresholds = np.unique(np.quantile(utility, [0.25, 0.40, 0.55, 0.70, 0.80, 0.90]))
        for threshold in thresholds:
            for p_min in (0.45, 0.50, 0.55, 0.60, 0.65, 0.70):
                for strategy in candidates:
                    policy = {
                        "strategy": strategy,
                        "alpha": alpha,
                        "keep_utility": float(threshold),
                        "keep_probability": p_min,
                        "add_utility": float(threshold if strategy != "bidirectional" else np.quantile(utility, 0.70)),
                        "add_probability": min(0.90, p_min + (0.10 if strategy == "bidirectional" else 0.0)),
                        "max_route_fraction": max_route_fraction,
                    }
                    route = apply_conservative_policy(predictions, fixed, policy)
                    candidates[strategy].append({
                        **policy,
                        "route_n": int(route.sum()),
                        "route_fraction": float(route.mean()),
                        "weighted_gap_mae": _weighted_gap_loss(validation, route),
                    })
    selected = {}
    fixed_loss = _weighted_gap_loss(validation, fixed)
    for strategy, rows in candidates.items():
        best = min(rows, key=lambda row: (row["weighted_gap_mae"], row["route_n"]))
        best["weighted_gap_delta_vs_fixed"] = best["weighted_gap_mae"] - fixed_loss
        selected[strategy] = best
    return selected


def optimize_gap_threshold(validation: pd.DataFrame) -> dict[str, float]:
    best = None
    for threshold in np.arange(1.5, 6.51, 0.05):
        route = validation.base_gap.to_numpy() < threshold
        row = {
            "threshold_eV": float(threshold),
            "route_n": int(route.sum()),
            "route_fraction": float(route.mean()),
            "weighted_gap_mae": _weighted_gap_loss(validation, route),
        }
        if best is None or (row["weighted_gap_mae"], row["route_n"]) < (best["weighted_gap_mae"], best["route_n"]):
            best = row
    return best
