"""Derived V5 reference reuse index."""

from __future__ import annotations

from collections import defaultdict
from typing import Any


def build_reference_reuse_index(
    evidence: list[dict[str, Any]], trajectories: list[dict[str, Any]]
) -> dict[str, Any]:
    uses: dict[str, list[str]] = defaultdict(list)
    for trajectory in trajectories:
        for reference_id in trajectory["state_at_start"]["reference_ids"]:
            uses[reference_id].append(trajectory["trajectory_id"])

    references = []
    known_ids = set()
    for envelope in sorted(evidence, key=lambda item: item["evidence_id"]):
        reference_id = envelope["evidence_id"]
        known_ids.add(reference_id)
        references.append(
            {
                "reference_id": reference_id,
                "track": envelope["track"],
                "contract": envelope["contract"],
                "scientific_scope": envelope["scope"],
                "comparison_status": envelope["outcome"]["comparison_status"],
                "transfer_status": envelope["outcome"]["transfer_status"],
                "artifact_availability": sorted(
                    {artifact["availability"] for artifact in envelope["artifacts"]}
                ),
                "role_use": envelope["role_use"],
                "source_envelope": envelope["_path"],
                "used_by_trajectory_ids": sorted(set(uses.get(reference_id, []))),
                "native_cost_available": False,
            }
        )
    missing = sorted(set(uses) - known_ids)
    return {
        "format": "molgap-rml-reference-reuse-index-v1",
        "references": references,
        "missing_reference_ids": missing,
        "missing_reference_policy": "comparison_pending_no_automatic_rerun",
        "human_reference_index": "models/REFERENCE_INDEX.md",
    }
