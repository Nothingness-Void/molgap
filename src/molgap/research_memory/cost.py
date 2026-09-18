"""Native-unit cost aggregation with measured/estimated/unknown separation."""

from __future__ import annotations

from collections import defaultdict
from typing import Any


UNITS = ("device_hours", "cpu_hours", "wall_hours", "queue_hours")


def _new_unit_bucket() -> dict[str, Any]:
    return {
        "measured": {"known_total": None, "known_records": 0},
        "estimated": {"known_total": None, "known_records": 0},
        "unknown_records": 0,
        "not_applicable_records": 0,
    }


def _add(target: dict[str, Any], unit: str, measurement: dict[str, Any]) -> None:
    bucket = target.setdefault(unit, _new_unit_bucket())
    status = measurement["status"]
    if status in {"measured", "estimated"}:
        value = float(measurement["value"])
        prior = bucket[status]["known_total"]
        bucket[status]["known_total"] = value if prior is None else prior + value
        bucket[status]["known_records"] += 1
    elif status == "measurement_missing":
        bucket["unknown_records"] += 1
    elif status == "not_applicable":
        bucket["not_applicable_records"] += 1


def _sorted_nested(value: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        outer: {unit: units[unit] for unit in sorted(units)}
        for outer, units in sorted(value.items())
    }


def build_cost_ledger(costs: list[dict[str, Any]]) -> dict[str, Any]:
    by_hardware: dict[str, dict[str, Any]] = defaultdict(dict)
    by_category: dict[str, dict[str, Any]] = defaultdict(dict)
    by_trajectory: dict[str, dict[str, Any]] = defaultdict(dict)
    unknown_events = set()
    not_applicable_events = set()
    estimated_events = set()
    infrastructure = []
    retry = []
    for event in sorted(costs, key=lambda item: item["cost_event_id"]):
        for unit in UNITS:
            measurement = event["measurement"][unit]
            _add(by_hardware[event["hardware"]], unit, measurement)
            _add(by_category[event["category"]], f"{event['hardware']}:{unit}", measurement)
            _add(by_trajectory[event["trajectory_id"]], f"{event['hardware']}:{unit}", measurement)
            if measurement["status"] == "measurement_missing":
                unknown_events.add(event["cost_event_id"])
            elif measurement["status"] == "not_applicable":
                not_applicable_events.add(event["cost_event_id"])
            elif measurement["status"] == "estimated":
                estimated_events.add(event["cost_event_id"])
        if event["category"] == "infrastructure_failure":
            infrastructure.append(event["cost_event_id"])
        if event["category"] == "retry":
            retry.append(event["cost_event_id"])
    return {
        "format": "molgap-rml-cost-ledger-v1",
        "event_count": len(costs),
        "totals_by_hardware": _sorted_nested(by_hardware),
        "totals_by_category": _sorted_nested(by_category),
        "totals_by_trajectory": _sorted_nested(by_trajectory),
        "infrastructure_event_ids": sorted(infrastructure),
        "retry_event_ids": sorted(retry),
        "estimated_event_count": len(estimated_events),
        "measurement_missing_event_count": len(unknown_events),
        "not_applicable_event_count": len(not_applicable_events),
        "cross_hardware_total": None,
    }
