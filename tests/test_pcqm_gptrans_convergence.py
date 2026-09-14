from __future__ import annotations

import pytest

from molgap.pcqm_gptrans_convergence import (
    BASE_VALID_MAE_EV,
    CONTRACT,
    MAX_ADDITIONAL_STEPS,
    MIN_LR,
    START_LR,
    continuation_learning_rate,
)


def test_continuation_contract_is_bounded_and_consumes_validation():
    assert CONTRACT["source_official_valid_mae_eV"] == BASE_VALID_MAE_EV
    assert CONTRACT["max_additional_passes"] == 6
    assert CONTRACT["patience_evaluations"] == 3
    assert CONTRACT["selection_role"] == "official-valid-consumed-for-convergence-selection"
    assert CONTRACT["test_dev_role"] == "sealed"


def test_continuation_learning_rate_is_monotonic_and_bounded():
    values = [continuation_learning_rate(step) for step in (1, MAX_ADDITIONAL_STEPS // 2, MAX_ADDITIONAL_STEPS)]
    assert START_LR > values[0] > values[1] > values[2]
    assert values[-1] == pytest.approx(MIN_LR)
    with pytest.raises(ValueError):
        continuation_learning_rate(0)
