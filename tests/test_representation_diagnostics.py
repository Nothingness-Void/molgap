"""Synthetic arrays only: no molecular model or inference."""
import json

import numpy as np
import pytest

from molgap.representation_diagnostics import exchange_summary, spectrum_summary


def test_rank_one_update_does_not_replace_hidden():
    hidden = np.eye(4)
    update = np.outer(np.arange(4), np.ones(4))
    result = exchange_summary(hidden, update)
    assert result["update_rank1_energy_fraction"] == pytest.approx(1)
    assert result["after"]["energy_effective_rank"] > 1
    assert result["before"]["energy_effective_rank"] == pytest.approx(3)


def test_translation_and_scaling():
    states = np.eye(4)
    a, b = spectrum_summary(states), spectrum_summary(2 * states + 7)
    assert a["energy_effective_rank"] == pytest.approx(b["energy_effective_rank"])
    assert b["dispersion"] == pytest.approx(4 * a["dispersion"])


def test_collapse_and_singleton_are_explicit():
    assert spectrum_summary(np.ones((1, 4)))["centered_rank_ceiling"] == 0
    result = exchange_summary(np.zeros((3, 4)), np.zeros((3, 4)), np.zeros((3, 4)))
    assert result["dispersion_ratio"] is None
    assert result["prediction_sensitivity"]["gradient_update_cosine"] is None
    json.dumps(result, allow_nan=False)


def test_local_prediction_derivative_and_no_mutation():
    hidden = np.eye(3)
    original = hidden.copy()
    update = np.ones((3, 3))
    result = exchange_summary(hidden, update, 2 * update)
    assert result["prediction_sensitivity"]["directional_derivative_eV"] == 18
    assert result["prediction_sensitivity"]["gradient_update_cosine"] == pytest.approx(1)
    np.testing.assert_array_equal(hidden, original)


@pytest.mark.parametrize("bad", [[], [[float('nan')]], [[float('inf')]], [1, 2]])
def test_invalid_arrays_rejected(bad):
    with pytest.raises(ValueError):
        spectrum_summary(bad)


def test_shape_mismatch_rejected():
    with pytest.raises(ValueError):
        exchange_summary(np.eye(3), np.eye(2))
    with pytest.raises(ValueError):
        exchange_summary(np.eye(3), np.eye(3), np.eye(2))
