"""Pre-registered matched-prefix futility checks for costly model screens."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MatchedPrefixGate:
    """Reject a candidate only when it is materially behind a frozen prefix."""

    completed_epochs: int
    reference_best_mae_eV: float
    maximum_deficit_eV: float

    def __post_init__(self) -> None:
        if self.completed_epochs < 1:
            raise ValueError("completed_epochs must be positive")
        if self.reference_best_mae_eV <= 0:
            raise ValueError("reference_best_mae_eV must be positive")
        if self.maximum_deficit_eV <= 0:
            raise ValueError("maximum_deficit_eV must be positive")


def evaluate_matched_prefix_futility(
    *,
    completed_epochs: int,
    candidate_best_mae_eV: float,
    gates: tuple[MatchedPrefixGate, ...],
) -> dict | None:
    """Return a decision at an exact gate boundary, otherwise ``None``.

    A gate is deliberately one-sided. It can terminate a clearly inferior run,
    but it never promotes a candidate or changes the final selection rule.
    """

    gate = next(
        (item for item in gates if item.completed_epochs == completed_epochs),
        None,
    )
    if gate is None:
        return None
    deficit = float(candidate_best_mae_eV - gate.reference_best_mae_eV)
    stopped = deficit >= gate.maximum_deficit_eV
    return {
        "completed_epochs": completed_epochs,
        "candidate_best_mae_eV": float(candidate_best_mae_eV),
        "reference_best_mae_eV": gate.reference_best_mae_eV,
        "candidate_minus_reference_eV": deficit,
        "maximum_deficit_eV": gate.maximum_deficit_eV,
        "futility_stopped": stopped,
    }
