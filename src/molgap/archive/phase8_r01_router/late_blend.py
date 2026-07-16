"""Continuous Base/Expert blending helpers for post-Expert inference."""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd


TARGETS = ("homo", "lumo", "gap")
GAP_BINS = np.asarray([-np.inf, 2.0, 3.0, 4.0, 5.0, np.inf])


def alpha_supervision(
    y_true: np.ndarray, base: np.ndarray, expert: np.ndarray,
    sampling_weight: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    correction = expert - base
    stable = np.abs(correction) > 1e-6
    alpha = np.full(len(base), 0.5, dtype=np.float64)
    alpha[stable] = np.clip((y_true[stable] - base[stable]) / correction[stable], 0.0, 1.0)
    weight = np.asarray(sampling_weight, dtype=np.float64) * np.maximum(np.abs(correction), 1e-4)
    return alpha, weight / max(weight.mean(), 1e-12)


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values, kind="stable")
    values, weights = values[order], weights[order]
    cutoff = 0.5 * weights.sum()
    return float(values[np.searchsorted(np.cumsum(weights), cutoff, side="left")])


def fit_fixed_alpha(
    y_true: np.ndarray, base: np.ndarray, expert: np.ndarray,
    sampling_weight: np.ndarray,
) -> float:
    alpha, weight = alpha_supervision(y_true, base, expert, sampling_weight)
    return float(np.clip(weighted_median(alpha, weight), 0.0, 1.0))


def fit_gap_binned_alpha(
    y_true: np.ndarray, base: np.ndarray, expert: np.ndarray,
    base_gap: np.ndarray, sampling_weight: np.ndarray,
) -> dict:
    global_alpha = fit_fixed_alpha(y_true, base, expert, sampling_weight)
    bin_ids = np.digitize(base_gap, GAP_BINS[1:-1], right=False)
    values = []
    for bin_id in range(len(GAP_BINS) - 1):
        mask = bin_ids == bin_id
        values.append(
            fit_fixed_alpha(y_true[mask], base[mask], expert[mask], sampling_weight[mask])
            if mask.sum() >= 50 else global_alpha
        )
    return {"global": global_alpha, "bins": values}


def predict_gap_binned_alpha(model: dict, base_gap: np.ndarray) -> np.ndarray:
    bin_ids = np.digitize(base_gap, GAP_BINS[1:-1], right=False)
    return np.asarray(model["bins"], dtype=np.float64)[bin_ids]


def fit_alpha_regressor(
    frame: pd.DataFrame,
    features: list[str],
    target: str,
    *,
    seed: int,
):
    y = frame[target].to_numpy(dtype=np.float64)
    base = frame[f"base_{target}"].to_numpy(dtype=np.float64)
    expert = frame[f"expert_{target}"].to_numpy(dtype=np.float64)
    alpha, weight = alpha_supervision(
        y, base, expert, frame.training_weight.to_numpy(dtype=np.float64)
    )
    model = lgb.LGBMRegressor(
        objective="huber", n_estimators=500, learning_rate=0.03,
        num_leaves=31, min_child_samples=50, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0, random_state=seed,
        n_jobs=-1, verbosity=-1,
    )
    model.fit(frame[features], alpha, sample_weight=weight)
    return model


def predict_alpha_regressor(model, frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    return np.clip(model.predict(frame[features]), 0.0, 1.0)


def blend(base: np.ndarray, expert: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    return base + np.asarray(alpha) * (expert - base)


def physics_project(predictions: np.ndarray, strength: float) -> np.ndarray:
    """Partially project [H, L, G] onto H - L + G = 0."""
    predictions = np.asarray(predictions, dtype=np.float64)
    normal = np.asarray([1.0, -1.0, 1.0])
    residual = predictions @ normal
    return predictions - float(strength) * residual[:, None] * normal[None, :] / 3.0


def target_metrics(y_true: np.ndarray, predictions: np.ndarray, weights: np.ndarray) -> dict:
    errors = np.abs(np.asarray(predictions) - np.asarray(y_true))
    return {
        target: {
            "mae": float(errors[:, index].mean()),
            "weighted_mae": float(np.average(errors[:, index], weights=weights)),
        }
        for index, target in enumerate(TARGETS)
    }
