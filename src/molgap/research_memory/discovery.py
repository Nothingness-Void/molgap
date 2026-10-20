"""Deterministic discovery of explicit RML and V5 records."""

from __future__ import annotations

import os
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


def _role_events(candidates) -> tuple[Path, ...]:
    """An input-role manifest is not an observed RML access event.

    Frozen packages may retain accepted data metadata in a roles directory.
    Keep its explicit format with its data owner; unknown or ambiguous role
    records still reach the strict RML validator and fail closed.
    """
    from molgap.evidence_pointers import load_json_object
    result = []
    for path in candidates:
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
    paths = {field.name: set() for field in fields(DiscoveredRecords)}
    names = {
        "v5_evidence.json": "evidence", "trajectory.json": "trajectories",
        "trace_manifest.json": "traces",
        "comparison_readiness_prelaunch.json": "comparison_prelaunch",
        "comparison_readiness.json": "comparison_readiness",
        "reference_bundle.json": "reference_bundles",
        "target_transform.json": "target_transform_assets",
    }
    finalized = set()

    def collect(path: Path, *, suffix_only: bool = False) -> None:
        name, parent = os.path.normcase(path.name), os.path.normcase(path.parent.name)
        kinds = set()
        if not suffix_only and name in names and path.exists():
            kinds.add(names[name])
        if parent in {"costs", "roles"} and name.endswith(".json"):
            kinds.add(parent)
        elif parent == "handoff" and name == os.path.normcase("READY_FOR_DESKTOP.json"):
            if path.exists():
                kinds.add("ready")
        if kinds:
            local = repo_local_path(root, path)
            if local.is_file():
                for kind in kinds:
                    paths[kind].add(local)
        if not suffix_only and name == "rml_finalized" and path.exists():
            finalized.add(repo_local_path(root, path))

    # One recursive inventory replaces one complete walk per record kind. Match
    # lexical names before resolving links, just as the original glob selectors do.
    for path in root.glob("experiments/**/*"):
        collect(path)
        # Recursive glob does not descend symbolic links, but the explicit
        # costs/roles/handoff suffixes historically inspect their direct children.
        name = os.path.normcase(path.name)
        if name in {"costs", "roles", "handoff"} and path.is_symlink() and path.is_dir():
            if name in {"costs", "roles"}:
                for child in path.glob("*.json"):
                    collect(child, suffix_only=True)
            elif name == "handoff":
                collect(path / "READY_FOR_DESKTOP.json", suffix_only=True)
    paths["roles"] = set(_role_events(sorted(paths["roles"])))
    from .finalize import verified_receipt

    for directory in sorted(finalized):
        receipt = verified_receipt(directory)
        for published, original in receipt["replacements"].items():
            old = repo_local_path(root, original)
            old.relative_to(directory.parent.resolve())
            for candidates in paths.values():
                if directory / published in candidates:
                    candidates.discard(old)
    return DiscoveredRecords(**{name: tuple(sorted(values)) for name, values in paths.items()})
