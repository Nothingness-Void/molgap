"""Hypothesis cards and trajectory outcomes for the V5 screening funnel."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


VALID_TRAJECTORY_OUTCOMES = frozenset(
    {
        "NO_TRAIN",
        "NEGATIVE_UNDER_CONTRACT",
        "INCONCLUSIVE",
        "INFRASTRUCTURE",
        "STOP_FOR_COST",
        "DUPLICATE_EVIDENCE",
        "QUALIFIED",
    }
)


@dataclass(frozen=True)
class HypothesisCard:
    hypothesis_id: str
    family_id: str
    current_baseline_deficiency: str
    supporting_evidence: Sequence[str]
    alternative_explanation: str
    changed_mechanism: str
    earliest_cheap_falsifier: str
    closed_related_routes: Sequence[str]
    expected_native_cost: Mapping[str, Any]
    decision_changed: str
    literature_refs: Sequence[str] = ()

    def validate(self) -> dict[str, Any]:
        values = {
            "hypothesis_id": self.hypothesis_id,
            "family_id": self.family_id,
            "current_baseline_deficiency": self.current_baseline_deficiency,
            "alternative_explanation": self.alternative_explanation,
            "changed_mechanism": self.changed_mechanism,
            "earliest_cheap_falsifier": self.earliest_cheap_falsifier,
            "decision_changed": self.decision_changed,
        }
        for key, value in values.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Hypothesis card field is empty: {key}")
        if not self.supporting_evidence:
            raise ValueError("Hypothesis card needs direct supporting evidence")
        if not self.closed_related_routes:
            raise ValueError("Hypothesis card must list related closed routes")
        if not self.expected_native_cost:
            raise ValueError("Hypothesis card must estimate native cost")
        return {
            "hypothesis_id": self.hypothesis_id,
            "family_id": self.family_id,
            "current_baseline_deficiency": self.current_baseline_deficiency,
            "supporting_evidence": list(self.supporting_evidence),
            "alternative_explanation": self.alternative_explanation,
            "changed_mechanism": self.changed_mechanism,
            "earliest_cheap_falsifier": self.earliest_cheap_falsifier,
            "closed_related_routes": list(self.closed_related_routes),
            "expected_native_cost": dict(self.expected_native_cost),
            "decision_changed": self.decision_changed,
            "literature_refs": list(self.literature_refs),
        }


def validate_trajectory(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the minimal ``state -> evidence -> action -> result -> decision`` record."""

    required = ("state", "evidence", "action", "result", "decision")
    missing = [key for key in required if key not in record]
    if missing:
        raise ValueError(f"Trajectory is missing: {missing}")
    outcome = record["result"].get("outcome") if isinstance(record["result"], Mapping) else None
    if outcome not in VALID_TRAJECTORY_OUTCOMES:
        raise ValueError(f"Unknown trajectory outcome: {outcome}")
    return {key: record[key] for key in required}
