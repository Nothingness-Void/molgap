"""Deterministic compiler from canonical evidence to disposable RML indexes."""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from .backtest import build_screening_backtest
from .backtest import build_policy_backtests
from .policy import load_policy_registry
from .replay import build_replay_pool
from .cost import build_cost_ledger
from .derived import validate_derived_outputs
from .references import build_reference_reuse_index
from .roles import build_role_reuse_index
from .summary import render_summary_markdown
from .validate import validate_repository_records
from .paths import repo_local_path


DERIVED_FILENAMES = (
    "trajectory_index.json",
    "trajectory_graph.json",
    "cost_ledger.json",
    "role_reuse_index.json",
    "reference_reuse_index.json",
    "ready_for_desktop_index.json",
    "screening_backtest.json",
    "replay_pool.json",
    "policy_backtest.json",
    "completeness_report.json",
    "research_summary.json",
    "research_summary.md",
)


def stable_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode("utf-8")


def _relative(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def _source_digest(records: dict[str, list[tuple[Path, dict[str, Any]]]], root: Path) -> str:
    digest = hashlib.sha256()
    for kind in sorted(records):
        for path, _ in sorted(records[kind], key=lambda item: str(item[0])):
            digest.update(kind.encode("utf-8"))
            digest.update(b"\0")
            digest.update(_relative(root, path).encode("utf-8"))
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
    return digest.hexdigest()


def compile_research_memory(repo_root: str | Path) -> dict[str, bytes]:
    validated = validate_repository_records(repo_root)
    root: Path = validated["root"]
    records = validated["records"]

    evidence = []
    for path, record in records["evidence"]:
        evidence.append({**record, "_path": _relative(root, path)})
    trajectories = []
    for path, record in records["trajectories"]:
        trajectories.append({**record, "_path": _relative(root, path)})
    costs = [record for _, record in records["costs"]]
    roles = [record for _, record in records["roles"]]
    traces = [record for _, record in records["traces"]]
    ready = []
    for path, record in records["ready"]:
        ready.append({**record, "_path": _relative(root, path)})

    evidence_ids = {item["evidence_id"] for item in evidence}
    trajectories_by_id = {item["trajectory_id"]: item for item in trajectories}
    ready_by_trajectory: dict[str, list[str]] = {}
    for package in ready:
        ready_by_trajectory.setdefault(package["trajectory_id"], []).append(package["package_id"])

    trajectory_index = {
        "format": "molgap-rml-trajectory-index-v1",
        "source_digest": _source_digest(records, root),
        "trajectories": [
            {
                "trajectory_id": item["trajectory_id"],
                "track": item["track"],
                "owner": item["owner"],
                "family_id": item["family_id"],
                "record_mode": item["record_mode"],
                "outcome": item["decision"]["outcome"],
                "parent_trajectory_ids": item["state_at_start"]["parent_trajectory_ids"],
                "result_evidence_ids": item["result"]["evidence_ids"],
                "decision_ref": item["decision"]["decision_ref"],
                "ready_package_ids": sorted(ready_by_trajectory.get(item["trajectory_id"], [])),
                "completeness_status": (
                    "partial" if item["record_mode"] == "retrospective_partial" else "complete"
                ),
                "record_path": item["_path"],
            }
            for item in sorted(trajectories, key=lambda row: row["trajectory_id"])
        ],
    }

    edges = []
    for item in sorted(trajectories, key=lambda row: row["trajectory_id"]):
        target = item["trajectory_id"]
        for parent in item["state_at_start"]["parent_trajectory_ids"]:
            edges.append({"type": "parent_trajectory", "source": parent, "target": target})
        for prior in item["state_at_start"]["prior_evidence_ids"]:
            edges.append({"type": "prior_evidence", "source": prior, "target": target})
        for reference in item["state_at_start"]["reference_ids"]:
            edges.append({"type": "reference_used", "source": reference, "target": target})
        for action in item["actions"]:
            for evidence_ref in action["evidence_refs"]:
                edges.append(
                    {
                        "type": "action_produced_evidence",
                        "source": f"{target}:{action['action_id']}",
                        "target": evidence_ref,
                    }
                )
    for package in ready:
        edges.append(
            {
                "type": "trajectory_produced_READY",
                "source": package["trajectory_id"],
                "target": package["package_id"],
            }
        )
    trajectory_graph = {
        "format": "molgap-rml-trajectory-graph-v1",
        "nodes": sorted(
            [
                {"id": item["trajectory_id"], "type": "trajectory"}
                for item in trajectories
            ]
            + [{"id": item["evidence_id"], "type": "evidence"} for item in evidence]
            + [{"id": item["package_id"], "type": "ready_package"} for item in ready],
            key=lambda node: (node["type"], node["id"]),
        ),
        "edges": sorted(edges, key=lambda edge: (edge["type"], edge["source"], edge["target"])),
    }

    cost_ledger = build_cost_ledger(costs)
    role_index = build_role_reuse_index(roles, evidence)
    reference_index = build_reference_reuse_index(evidence, trajectories)
    ready_index = {
        "format": "molgap-rml-ready-for-desktop-index-v1",
        "valid": [
            {
                "package_id": item["package_id"],
                "trajectory_id": item["trajectory_id"],
                "path": item["_path"],
                "status": "valid_unconsumed",
            }
            for item in sorted(ready, key=lambda row: row["package_id"])
        ],
        "invalid": [],
        "superseded": [],
        "consumed": [],
        "blocked": [],
    }
    backtest = build_screening_backtest(traces)
    replay_pool = build_replay_pool(root, records)
    policy_backtest = build_policy_backtests(load_policy_registry(root), replay_pool)

    trajectory_evidence_ids = {
        evidence_id
        for item in trajectories
        for evidence_id in item["result"]["evidence_ids"]
    }
    issues = []

    def issue(severity: str, code: str, items: list[str]) -> None:
        if items:
            issues.append({"severity": severity, "code": code, "count": len(items), "items": sorted(items)})

    issue("WARN", "v5_evidence_without_trajectory", sorted(evidence_ids - trajectory_evidence_ids))
    issue(
        "WARN",
        "trajectory_without_v5_evidence",
        sorted(
            item["trajectory_id"]
            for item in trajectories
            if not set(item["result"]["evidence_ids"]) & evidence_ids
        ),
    )
    issue(
        "WARN",
        "historical_partial_record",
        sorted(item["trajectory_id"] for item in trajectories if item["record_mode"] == "retrospective_partial"),
    )
    cost_trajectory_ids = {event["trajectory_id"] for event in costs}
    complete_native_cost_trajectory_ids = {
        event["trajectory_id"]
        for event in costs
        if (
            any(
                event["measurement"][unit]["status"] == "measured"
                for unit in ("device_hours", "cpu_hours")
            )
            and all(
                event["measurement"][unit]["status"] in {"measured", "not_applicable"}
                for unit in ("device_hours", "cpu_hours")
            )
        )
    }
    incomplete_native_cost_trajectory_ids = (
        cost_trajectory_ids - complete_native_cost_trajectory_ids
    )
    issue(
        "WARN",
        "missing_cost_event",
        sorted(item["trajectory_id"] for item in trajectories if item["trajectory_id"] not in cost_trajectory_ids),
    )
    issue(
        "WARN",
        "incomplete_native_cost_measurement",
        sorted(incomplete_native_cost_trajectory_ids),
    )
    explicit_role_trajectory_ids = {event["trajectory_id"] for event in roles}
    issue(
        "WARN",
        "missing_explicit_role_identity",
        sorted(item["trajectory_id"] for item in trajectories if item["trajectory_id"] not in explicit_role_trajectory_ids),
    )
    trace_trajectory_ids = {trace["trajectory_id"] for trace in traces}
    issue(
        "INFO",
        "trace_unavailable_for_backtest",
        sorted(item["trajectory_id"] for item in trajectories if item["trajectory_id"] not in trace_trajectory_ids),
    )
    issue("WARN", "missing_strict_reference", reference_index["missing_reference_ids"])
    cost_completeness = {
        "trajectories_total": len(trajectories),
        "with_cost_event": len(cost_trajectory_ids),
        "without_cost_event": len(trajectories) - len(cost_trajectory_ids),
        "with_complete_native_measurement": len(complete_native_cost_trajectory_ids),
        "with_incomplete_native_measurement": len(incomplete_native_cost_trajectory_ids),
    }
    completeness = {
        "format": "molgap-rml-completeness-report-v1",
        "issues": sorted(issues, key=lambda row: (row["severity"], row["code"])),
        "counts_by_severity": dict(sorted(Counter(row["severity"] for row in issues).items())),
        "broken_required_pointers": [],
        "ready_prerequisites_missing": [],
        "cost_completeness": cost_completeness,
    }

    outcomes = Counter(item["decision"]["outcome"] for item in trajectories)
    summary = {
        "format": "molgap-rml-research-summary-v1",
        "source_digest": trajectory_index["source_digest"],
        "evidence": {
            "validated": len(evidence),
            "without_trajectory": len(evidence_ids - trajectory_evidence_ids),
        },
        "trajectories": {
            "count": len(trajectories),
            "outcomes": dict(sorted(outcomes.items())),
        },
        "transfer": {
            "ready_valid": len(ready),
            "ready_blocked": 0,
            "missing_references": len(reference_index["missing_reference_ids"]),
        },
        "roles": {
            "explicit_events": len(roles),
            "coarse_historical": role_index["ambiguous_historical_role_record_count"],
            "repeatedly_selected": len(role_index["repeatedly_selected_role_identities"]),
        },
        "cost": {
            "totals_by_hardware": cost_ledger["totals_by_hardware"],
            "unknown_events": cost_ledger["measurement_missing_event_count"],
            "completeness": cost_completeness,
        },
        "backtest": {
            "status": backtest["status"],
            "eligible": len(backtest["included_trajectory_ids"]),
            "excluded": len(backtest["excluded_traces"]),
        },
        "memory_gaps": [
            {"severity": row["severity"], "code": row["code"], "count": row["count"]}
            for row in completeness["issues"]
        ],
    }

    derived_objects = {
        "trajectory_index.json": trajectory_index,
        "trajectory_graph.json": trajectory_graph,
        "cost_ledger.json": cost_ledger,
        "role_reuse_index.json": role_index,
        "reference_reuse_index.json": reference_index,
        "ready_for_desktop_index.json": ready_index,
        "screening_backtest.json": backtest,
        "replay_pool.json": replay_pool,
        "policy_backtest.json": policy_backtest,
        "completeness_report.json": completeness,
        "research_summary.json": summary,
    }
    validate_derived_outputs(derived_objects)
    payloads = {
        name: stable_json(value) for name, value in derived_objects.items()
    }
    payloads.update({
        "research_summary.md": render_summary_markdown(summary).encode("utf-8"),
    })
    return payloads


def rebuild_research_memory(repo_root: str | Path) -> dict[str, bytes]:
    root = Path(repo_root).resolve()
    payloads = compile_research_memory(root)
    output_root = repo_local_path(root, "research_memory/derived")
    output_root.mkdir(parents=True, exist_ok=True)
    for name, payload in sorted(payloads.items()):
        target = repo_local_path(root, output_root / name)
        with tempfile.NamedTemporaryFile(dir=output_root, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
        temporary.replace(target)
    return payloads


def frozen_differences(repo_root: str | Path) -> list[str]:
    root = Path(repo_root).resolve()
    payloads = compile_research_memory(root)
    output_root = root / "research_memory" / "derived"
    differences = []
    for name in DERIVED_FILENAMES:
        path = output_root / name
        if not path.is_file() or path.read_bytes() != payloads[name]:
            differences.append(name)
    return differences
