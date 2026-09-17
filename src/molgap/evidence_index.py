"""Compact, discoverable index for positive, negative, and inconclusive evidence."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional


INDEX_FORMAT = "molgap-evidence-index-v1"
REQUIRED_FIELDS = (
    "evidence_id",
    "family_id",
    "status",
    "source",
    "decision_ref",
    "contract_fingerprint",
)
DISCOVERABLE_STATUSES = frozenset(
    {"POSITIVE", "NEGATIVE", "INCONCLUSIVE", "INFRASTRUCTURE", "STOP_FOR_COST", "DUPLICATE"}
)


def normalize_entries(entries: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    normalized: dict[str, dict[str, Any]] = {}
    for entry in entries:
        missing = [field for field in REQUIRED_FIELDS if field not in entry]
        if missing:
            raise ValueError(f"Evidence entry is missing: {missing}")
        status = str(entry["status"]).upper()
        if status not in DISCOVERABLE_STATUSES:
            raise ValueError(f"Evidence status is not discoverable: {status}")
        evidence_id = str(entry["evidence_id"])
        if not evidence_id:
            raise ValueError("Evidence id must be non-empty")
        if evidence_id in normalized and normalized[evidence_id] != dict(entry):
            raise ValueError(f"Evidence id has conflicting records: {evidence_id}")
        item = dict(entry)
        item["status"] = status
        normalized[evidence_id] = item
    return [normalized[key] for key in sorted(normalized)]


def write_evidence_index(path: Path, entries: Iterable[Mapping[str, Any]]) -> None:
    """Atomically write a deterministic index; no evidence is silently dropped."""

    payload = {
        "format": INDEX_FORMAT,
        "entries": normalize_entries(entries),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Optional[Path] = None
    try:
        fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
        temporary = Path(name)
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def load_evidence_index(path: Path) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("format") != INDEX_FORMAT:
        raise ValueError("Evidence index format is not supported")
    return normalize_entries(data.get("entries", []))
