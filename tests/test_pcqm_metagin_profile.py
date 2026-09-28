"""CPU-only checks for warm-cost and actual 40-epoch budget projection."""
from __future__ import annotations

import math

import pytest

from molgap.pcqm_metagin_profile import projected_40_epoch_seconds
from molgap.pcqm_metagin_screen import _warmed_step_seconds


def test_cold_cuda_step_is_observed_but_not_extrapolated():
    result = _warmed_step_seconds([
        (0.1, "cold", 10.0),
        (0.1, "repeat", 0.4),
        (0.1, "repeat", 0.5),
    ])
    assert result == 0.5
    with pytest.raises(RuntimeError, match="not deterministic"):
        _warmed_step_seconds([(0.1, "cold", 10.0), (0.1, "a", 0.4), (0.1, "b", 0.5)])


def test_projection_includes_training_and_development_forward_work():
    assert projected_40_epoch_seconds(0.4, 0.1) == pytest.approx(
        40 * (781 * 0.4 + 391 * 0.1)
    )
    for value in (0, -1, math.nan, math.inf):
        with pytest.raises(ValueError):
            projected_40_epoch_seconds(value, 0.1)
