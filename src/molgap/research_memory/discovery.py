"""Deterministic discovery of explicit RML and V5 records."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from .paths import repo_local_path


@dataclass(frozen=True)
class DiscoveredRecords:
    evidence: tuple[Path, ...]
    trajectories: tuple[Path, ...]
    costs: tuple[Path, ...]
    roles: tuple[Path, ...]
    traces: tuple[Path, ...]
    ready: tuple[Path, ...]
    comparison_prelaunch: tuple[Path, ...] = ()
    comparison_readiness: tuple[Path, ...] = ()
    reference_bundles: tuple[Path, ...] = ()
    target_transform_assets: tuple[Path, ...] = ()


def _find(root: Path, pattern: str) -> tuple[Path, ...]:
    paths = (repo_local_path(root, path) for path in root.glob(pattern))
    return tuple(sorted(path for path in paths if path.is_file()))


def _role_events(root: Path) -> tuple[Path, ...]:
    """An input-role manifest is not an observed RML access event.

    Frozen packages may retain accepted data metadata in a roles directory.
    Keep its explicit format with its data owner; unknown or ambiguous role
    records still reach the strict RML validator and fail closed.
    """
    from molgap.evidence_pointers import load_json_object
    result = []
    for path in _find(root, "experiments/**/roles/*.json"):
        record = load_json_object(path)
        if record.get("format") == "molgap-pcqm4mv2-unified-roles-v2":
            if ("schema" in record or "role_event_id" in record or
                    record.get("role") not in {"train_100k", "train_500k", "common_dev"} or
                    record.get("status") != "complete" or record.get("protected") is not False or
                    record.get("ordering") != "ascending-source-index" or
                    type(record.get("rows")) is not int or record["rows"] < 1 or
                    type(record.get("graph_chunks")) is not list or
                    not record.get("source_index_sha256")):
                raise ValueError("Ambiguous or malformed retained input-role manifest")
            continue
        result.append(path)
    return tuple(result)


def discover_records(repo_root: str | Path) -> DiscoveredRecords:
    root = Path(repo_root).resolve()
    discovered = DiscoveredRecords(
        evidence=_find(root, "experiments/**/v5_evidence.json"),
        trajectories=_find(root, "experiments/**/trajectory.json"),
        costs=_find(root, "experiments/**/costs/*.json"),
        roles=_role_events(root),
        traces=_find(root, "experiments/**/trace_manifest.json"),
        ready=_find(root, "experiments/**/handoff/READY_FOR_DESKTOP.json"),
        comparison_prelaunch=_find(root, "experiments/**/comparison_readiness_prelaunch.json"),
        comparison_readiness=_find(root, "experiments/**/comparison_readiness.json"),
        reference_bundles=_find(root, "experiments/**/reference_bundle.json"),
        target_transform_assets=_find(root, "experiments/**/target_transform.json"),
    )
    from .finalize import verified_receipt

    paths = {field.name: set(getattr(discovered, field.name)) for field in fields(discovered)}
    for directory in sorted((root / "experiments").glob("**/rml_finalized")):
        directory = repo_local_path(root, directory)
        receipt = verified_receipt(directory)
        for published, original in receipt["replacements"].items():
            old = repo_local_path(root, original)
            old.relative_to(directory.parent.resolve())
            for candidates in paths.values():
                if directory / published in candidates:
                    candidates.discard(old)
    return DiscoveredRecords(**{name: tuple(sorted(values)) for name, values in paths.items()})
