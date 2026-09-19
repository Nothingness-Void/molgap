"""Deterministic discovery of explicit RML and V5 records."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


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
    return tuple(sorted(path for path in root.glob(pattern) if path.is_file()))


def discover_records(repo_root: str | Path) -> DiscoveredRecords:
    root = Path(repo_root).resolve()
    return DiscoveredRecords(
        evidence=_find(root, "experiments/**/v5_evidence.json"),
        trajectories=_find(root, "experiments/**/trajectory.json"),
        costs=_find(root, "experiments/**/costs/*.json"),
        roles=_find(root, "experiments/**/roles/*.json"),
        traces=_find(root, "experiments/**/trace_manifest.json"),
        ready=_find(root, "experiments/**/handoff/READY_FOR_DESKTOP.json"),
        comparison_prelaunch=_find(
            root, "experiments/**/comparison_readiness_prelaunch.json"
        ),
        comparison_readiness=_find(root, "experiments/**/comparison_readiness.json"),
        reference_bundles=_find(root, "experiments/**/reference_bundle.json"),
        target_transform_assets=_find(root, "experiments/**/target_transform.json"),
    )
