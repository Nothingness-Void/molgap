"""Deterministic JSON-to-Markdown RML summary rendering."""

from __future__ import annotations

from typing import Any


def _basis_text(basis: str, value: dict[str, Any]) -> str:
    records = value["known_records"]
    if records == 0:
        return f"{basis}=unknown (0 {basis} records)"
    suffix = "record" if records == 1 else "records"
    return f"{basis}={value['known_total']:.6f} ({records} {basis} {suffix})"


def _cost_bucket_text(bucket: dict[str, Any]) -> str:
    if (
        bucket["measured"]["known_records"] == 0
        and bucket["estimated"]["known_records"] == 0
        and bucket["unknown_records"] == 0
        and bucket["not_applicable_records"] > 0
    ):
        return (
            f"not applicable ({bucket['not_applicable_records']} records); "
            "measurement_missing=0"
        )
    return (
        f"{_basis_text('measured', bucket['measured'])}, "
        f"{_basis_text('estimated', bucket['estimated'])}, "
        f"measurement_missing={bucket['unknown_records']}, "
        f"not_applicable={bucket['not_applicable_records']}"
    )


def render_summary_markdown(summary: dict[str, Any]) -> str:
    outcomes = summary["trajectories"]["outcomes"]
    lines = [
        "# MolGap Research Memory Summary",
        "",
        "## Evidence",
        f"- Validated V5 evidence: {summary['evidence']['validated']}",
        f"- Evidence without trajectory: {summary['evidence']['without_trajectory']}",
        "",
        "## Trajectories",
    ]
    for key in sorted(outcomes):
        lines.append(f"- {key}: {outcomes[key]}")
    lines.extend(
        [
            "",
            "## Transfer",
            f"- READY_FOR_DESKTOP valid: {summary['transfer']['ready_valid']}",
            f"- READY blocked/invalid: {summary['transfer']['ready_blocked']}",
            f"- Missing references: {summary['transfer']['missing_references']}",
            "",
            "## Roles",
            f"- Explicit role events: {summary['roles']['explicit_events']}",
            f"- Coarse historical role records: {summary['roles']['coarse_historical']}",
            f"- Repeatedly selected role identities: {summary['roles']['repeatedly_selected']}",
            "",
            "## Cost",
        ]
    )
    totals = summary["cost"]["totals_by_hardware"]
    if totals:
        for hardware, units in sorted(totals.items()):
            for unit, bucket in sorted(units.items()):
                lines.append(
                    f"- {hardware} {unit}: {_cost_bucket_text(bucket)}"
                )
    else:
        lines.append("- No native-cost events recorded")
    lines.extend(
        [
            f"- Measurement-missing cost events: {summary['cost']['unknown_events']}",
            (
                "- Cost-event coverage: "
                f"{summary['cost']['completeness']['with_cost_event']}/"
                f"{summary['cost']['completeness']['trajectories_total']} trajectories; "
                f"incomplete native measurement="
                f"{summary['cost']['completeness']['with_incomplete_native_measurement']}"
            ),
            "",
            "## Screening Backtest",
            f"- Status: {summary['backtest']['status']}",
            f"- Eligible traces: {summary['backtest']['eligible']}",
            f"- Excluded traces: {summary['backtest']['excluded']}",
            "",
            "## Memory Gaps",
        ]
    )
    for issue in summary["memory_gaps"]:
        lines.append(f"- [{issue['severity']}] {issue['code']}: {issue['count']}")
    return "\n".join(lines) + "\n"
