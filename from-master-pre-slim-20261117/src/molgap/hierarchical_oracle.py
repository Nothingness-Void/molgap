"""Oracle ceilings for a general model plus an expensive hard-region teacher."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _arrays(
    y_true: np.ndarray, base: np.ndarray, expert: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = tuple(np.asarray(value, dtype=np.float64) for value in (y_true, base, expert))
    if not (values[0].shape == values[1].shape == values[2].shape):
        raise ValueError("Truth, base, and expert arrays must have identical shapes")
    if values[0].ndim != 2 or not values[0].shape[0] or not values[0].shape[1]:
        raise ValueError("Expected non-empty [molecule, target] arrays")
    if not all(np.isfinite(value).all() for value in values):
        raise ValueError("Oracle inputs must be finite")
    return values


def _metric_block(
    y_true: np.ndarray, prediction: np.ndarray, target_names: Sequence[str]
) -> dict[str, object]:
    errors = np.abs(prediction - y_true)
    per_target = {
        name: {"mae_eV": float(errors[:, index].mean())}
        for index, name in enumerate(target_names)
    }
    return {
        "targets": per_target,
        "average_mae_eV": float(errors.mean()),
    }


def _optimal_alpha(
    y_true: np.ndarray, base: np.ndarray, expert: np.ndarray
) -> np.ndarray:
    correction = expert - base
    alpha = np.zeros_like(correction)
    np.divide(y_true - base, correction, out=alpha, where=np.abs(correction) > 1e-12)
    return np.clip(alpha, 0.0, 1.0)


def hierarchical_oracle_analysis(
    y_true: np.ndarray,
    base: np.ndarray,
    expert: np.ndarray,
    *,
    target_names: Sequence[str],
    budgets: Sequence[float] = (0.05, 0.10, 0.20),
    base_encoder_passes: float = 1.0,
    expert_encoder_passes: float = 4.0,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """Compute targetwise switch and line-segment residual Oracle ceilings.

    Budgets constrain the fraction of molecules sent to the expert. Once the
    expert has run, each target may independently retain the base prediction.
    """
    y_true, base, expert = _arrays(y_true, base, expert)
    if len(target_names) != y_true.shape[1]:
        raise ValueError("target_names does not match the target dimension")
    if any(not 0.0 <= float(value) <= 1.0 for value in budgets):
        raise ValueError("Budgets must be fractions in [0, 1]")

    base_error = np.abs(base - y_true)
    expert_error = np.abs(expert - y_true)
    switch_gain = base_error - expert_error
    switch_wins = switch_gain > 0.0
    switch_prediction = np.where(switch_wins, expert, base)

    alpha = _optimal_alpha(y_true, base, expert)
    residual_prediction = base + alpha * (expert - base)
    residual_gain = base_error - np.abs(residual_prediction - y_true)

    methods: dict[str, object] = {
        "base": _metric_block(y_true, base, target_names),
        "expert": _metric_block(y_true, expert, target_names),
        "unconstrained_switch": _metric_block(
            y_true, switch_prediction, target_names
        ),
        "unconstrained_residual": _metric_block(
            y_true, residual_prediction, target_names
        ),
    }
    budget_masks: dict[str, np.ndarray] = {}
    molecule_gain = np.maximum(switch_gain, 0.0).sum(axis=1)
    residual_molecule_gain = np.maximum(residual_gain, 0.0).sum(axis=1)
    order = np.argsort(-molecule_gain, kind="stable")
    residual_order = np.argsort(-residual_molecule_gain, kind="stable")

    budget_details: dict[str, object] = {}
    for budget in budgets:
        key = f"{int(round(100 * float(budget)))}pct"
        count = min(len(y_true), int(np.ceil(len(y_true) * float(budget))))
        mask = np.zeros(len(y_true), dtype=bool)
        residual_mask = np.zeros(len(y_true), dtype=bool)
        if count:
            mask[order[:count]] = True
            residual_mask[residual_order[:count]] = True
        budget_masks[f"switch_{key}"] = mask
        budget_masks[f"residual_{key}"] = residual_mask

        switch_budget_prediction = base.copy()
        residual_budget_prediction = base.copy()
        switch_budget_prediction[mask] = switch_prediction[mask]
        residual_budget_prediction[residual_mask] = residual_prediction[residual_mask]
        methods[f"switch_{key}"] = _metric_block(
            y_true, switch_budget_prediction, target_names
        )
        methods[f"residual_{key}"] = _metric_block(
            y_true, residual_budget_prediction, target_names
        )
        actual_fraction = float(count / len(y_true))
        budget_details[key] = {
            "requested_fraction": float(budget),
            "called_molecules": int(count),
            "actual_fraction": actual_fraction,
            "expected_encoder_passes_per_molecule": float(
                base_encoder_passes + actual_fraction * expert_encoder_passes
            ),
        }

    result = {
        "n": int(len(y_true)),
        "target_names": list(target_names),
        "cost_contract": {
            "base_encoder_passes": float(base_encoder_passes),
            "hard_teacher_encoder_passes_when_called": float(expert_encoder_passes),
        },
        "methods": methods,
        "budgets": budget_details,
        "expert_win_rate_by_target": {
            name: float(switch_wins[:, index].mean())
            for index, name in enumerate(target_names)
        },
        "unconstrained_any_target_call_fraction": float(switch_wins.any(axis=1).mean()),
        "mean_optimal_alpha_by_target": {
            name: float(alpha[:, index].mean())
            for index, name in enumerate(target_names)
        },
    }
    arrays = {
        "switch_gain": switch_gain,
        "switch_wins": switch_wins,
        "optimal_alpha": alpha,
        "residual_gain": residual_gain,
        **budget_masks,
    }
    return result, arrays
