"""Saved-array diagnostics; never load models, graphs, or target labels.

Effective rank describes representation geometry, not information or quality.
Directional derivatives are local sensitivities, not removal counterfactuals.
"""

from __future__ import annotations

import numpy as np


def _matrix(value, name):
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or min(array.shape) == 0 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a nonempty finite matrix")
    return array


def spectrum_summary(states):
    """One molecule's centered state geometry using squared singular values."""
    states = _matrix(states, "states")
    centered = states - states.mean(axis=0, keepdims=True)
    energy = np.linalg.svd(centered, compute_uv=False) ** 2
    total = float(energy.sum())
    ceiling = min(states.shape[0] - 1, states.shape[1])
    if total == 0:
        effective_rank = stable_rank = leading_fraction = 0.0
    else:
        probabilities = energy[energy > 0] / total
        effective_rank = float(np.exp(-np.sum(probabilities * np.log(probabilities))))
        stable_rank = total / float(energy[0])
        leading_fraction = float(energy[0]) / total
    return {
        "rows": int(states.shape[0]), "channels": int(states.shape[1]),
        "centered_rank_ceiling": ceiling,
        "dispersion": total / states.shape[0],
        "energy_effective_rank": effective_rank,
        "normalized_effective_rank": effective_rank / ceiling if ceiling else 0.0,
        "stable_rank": stable_rank,
        "leading_energy_fraction": leading_fraction,
    }


def exchange_summary(hidden, update, prediction_gradient=None):
    """Summarize H -> H+U; gradient, if supplied, is d(prediction_eV)/d(H+U).

    Undefined ratios stay null instead of inventing epsilon-dependent evidence.
    Remote hooks must detach copies; this pure-array function cannot mutate a
    model or certify that the caller supplied the correct gradient location.
    """
    hidden = _matrix(hidden, "hidden")
    update = _matrix(update, "update")
    if hidden.shape != update.shape:
        raise ValueError("hidden/update shape mismatch")
    before, after = spectrum_summary(hidden), spectrum_summary(hidden + update)
    singular = np.linalg.svd(update, compute_uv=False)
    update_energy = float(np.sum(singular ** 2))
    hidden_norm, update_norm = float(np.linalg.norm(hidden)), float(np.linalg.norm(update))
    result = {
        "before": before, "after": after,
        "dispersion_ratio": after["dispersion"] / before["dispersion"] if before["dispersion"] else None,
        "update_to_hidden_norm": update_norm / hidden_norm if hidden_norm else None,
        "update_rank1_energy_fraction": float(singular[0] ** 2) / update_energy if update_energy else None,
        "prediction_sensitivity": None,
    }
    if prediction_gradient is not None:
        gradient = _matrix(prediction_gradient, "prediction_gradient")
        if gradient.shape != hidden.shape:
            raise ValueError("gradient/state shape mismatch")
        gradient_norm = float(np.linalg.norm(gradient))
        directional = float(np.sum(gradient * update))
        result["prediction_sensitivity"] = {
            "directional_derivative_eV": directional,
            "gradient_norm": gradient_norm,
            "gradient_update_cosine": directional / (gradient_norm * update_norm)
            if gradient_norm and update_norm else None,
        }
    return result
