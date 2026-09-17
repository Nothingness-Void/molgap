"""Conservative backtesting gate for future low-cost screening ladders."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, Sequence


@dataclass(frozen=True)
class ScaleObservation:
    """One low/high budget pair considered for prospective calibration."""

    family_id: str
    low_budget: str
    high_budget: str
    low_gain_eV: Optional[float]
    high_gain_eV: Optional[float]
    same_scientific_contract: bool
    early_late_trace_available: bool
    notes: str = ""

    def validate(self) -> None:
        for name in ("family_id", "low_budget", "high_budget"):
            if not getattr(self, name):
                raise ValueError(f"Scale observation field is empty: {name}")
        for name in ("low_gain_eV", "high_gain_eV"):
            value = getattr(self, name)
            if value is not None and not math.isfinite(float(value)):
                raise ValueError(f"Scale observation is non-finite: {name}")


def backtest_low_cost_screening(
    observations: Sequence[ScaleObservation],
) -> dict[str, Any]:
    """Return a gate result; never release a ladder from endpoint anecdotes.

    A prospective ladder needs at least three same-contract pairs with complete
    early/late traces.  Fewer pairs, mixed contracts, or missing traces remain
    ``PENDING`` and cannot authorize a schedule change.
    """

    for observation in observations:
        observation.validate()
    eligible = [
        observation
        for observation in observations
        if observation.same_scientific_contract
        and observation.early_late_trace_available
        and observation.low_gain_eV is not None
        and observation.high_gain_eV is not None
    ]
    if len(eligible) < 3:
        return {
            "calibration_status": "PENDING",
            "early_stop_rule_released": False,
            "eligible_pairs": [item.family_id for item in eligible],
            "required_pairs": 3,
            "reason": "insufficient_same_contract_trace_pairs",
        }
    direction_preserved = all(
        (item.low_gain_eV >= 0) == (item.high_gain_eV >= 0) for item in eligible
    )
    return {
        "calibration_status": "CALIBRATED" if direction_preserved else "PENDING",
        "early_stop_rule_released": direction_preserved,
        "eligible_pairs": [item.family_id for item in eligible],
        "required_pairs": 3,
        "direction_preserved": direction_preserved,
        "reason": "three_same_contract_trace_pairs" if direction_preserved else "direction_not_preserved",
    }
