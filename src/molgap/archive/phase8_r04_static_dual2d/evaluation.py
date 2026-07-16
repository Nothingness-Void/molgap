"""Expert complementarity and static-ensemble evaluation."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


TARGETS = ("homo", "lumo", "gap")


def metrics(y, prediction):
    error = np.abs(np.asarray(prediction) - np.asarray(y))
    result = {
        target: {"mae": float(error[:, index].mean())}
        for index, target in enumerate(TARGETS)
    }
    result["weighted_mae"] = float((error @ np.asarray([0.25, 0.25, 0.50])).mean())
    return result


def fit_static_weights(y, expert_predictions):
    """Fit one non-negative simplex weight vector per target on validation."""
    y = np.asarray(y)
    predictions = np.asarray(expert_predictions)
    weights = np.zeros((len(TARGETS), predictions.shape[-1]), dtype=np.float64)
    for target_index in range(len(TARGETS)):
        values = predictions[:, target_index, :]

        def objective(candidate):
            return np.mean(np.abs(values @ candidate - y[:, target_index]))

        result = minimize(
            objective,
            np.full(values.shape[1], 1.0 / values.shape[1]),
            method="SLSQP",
            bounds=[(0.0, 1.0)] * values.shape[1],
            constraints={"type": "eq", "fun": lambda value: value.sum() - 1.0},
            options={"maxiter": 500, "ftol": 1e-12},
        )
        if not result.success:
            raise RuntimeError(f"Static weight optimization failed: {result.message}")
        weights[target_index] = result.x
    return weights


def apply_static_weights(expert_predictions, weights):
    return np.sum(np.asarray(expert_predictions) * np.asarray(weights)[None, :, :], axis=-1)


def expert_complementarity(y, expert_predictions, sources):
    y = np.asarray(y)
    predictions = np.asarray(expert_predictions)
    absolute = np.abs(predictions - y[:, :, None])
    winners = absolute.argmin(axis=-1)
    result = {"targets": {}, "slices": {}}
    for target_index, target in enumerate(TARGETS):
        correlations = np.corrcoef(absolute[:, target_index, :].T)
        result["targets"][target] = {
            "win_fraction": [
                float(np.mean(winners[:, target_index] == expert))
                for expert in range(predictions.shape[-1])
            ],
            "absolute_error_correlation": correlations.tolist(),
        }
    sources = np.asarray(sources)
    for source in sorted(set(sources)):
        mask = sources == source
        result["slices"][source] = {
            target: {
                "n": int(mask.sum()),
                "expert_gap_or_orbital_mae": [
                    float(absolute[mask, target_index, expert].mean())
                    for expert in range(predictions.shape[-1])
                ],
                "win_fraction": [
                    float(np.mean(winners[mask, target_index] == expert))
                    for expert in range(predictions.shape[-1])
                ],
            }
            for target_index, target in enumerate(TARGETS)
        }
    return result
