"""Explicit field-mapped CSV/JSON recovery; no model-specific parsers or inference."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from .trace import FIELDS, NUMBERS, atomic_write, canonicalize_trace, file_digest, json_bytes


def recover_trace(sources: list[str | Path], spec: dict[str, Any], output: str | Path) -> dict[str, Any]:
    mapping = spec["field_mapping"]
    if not isinstance(mapping, dict) or set(mapping) - set(FIELDS):
        raise ValueError("recovery mapping uses unknown canonical fields")
    if any(not isinstance(value, str) or not value for value in mapping.values()):
        raise ValueError("source columns must be explicit names")
    target = Path(output)
    if target.exists():
        raise ValueError("recovery never overwrites a retained artifact")
    result = {"trajectory_id": spec["trajectory_id"], "run_id": spec["run_id"],
              "metric_semantics": spec["metric_semantics"],
              "device_time_semantics": spec.get("device_time_semantics"),
              "observations": [], "provenance": []}
    for source in sources:
        path = Path(source)
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if path.suffix.lower() == ".csv":
            rows = list(csv.DictReader(raw.decode("utf-8-sig").splitlines()))
        elif path.suffix.lower() == ".jsonl":
            rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
        elif path.suffix.lower() == ".json":
            rows = json.loads(raw)
            if isinstance(rows, dict):
                rows = rows[spec["rows_key"]]
        else:
            raise ValueError("recovery supports explicit CSV, JSON and JSONL records only")
        if not isinstance(rows, list) or any(not isinstance(r, dict) for r in rows):
            raise ValueError("source must contain an ordered record array")
        offset = len(result["observations"])
        recovered = set()
        for row in rows:
            observation = {}
            for field, column in mapping.items():
                value = row.get(column)
                if value is None or value == "":
                    observation[field] = None
                    continue
                if field in NUMBERS and isinstance(value, str):
                    value = int(value) if field in {"optimizer_step", "sample_presentations"} else float(value)
                observation[field] = value
                recovered.add(field)
            result["observations"].append(observation)
        if file_digest(path) != digest:
            raise ValueError("recovery source changed while reading")
        result["provenance"].append({"source": str(path.resolve()), "sha256": digest,
                                     "field_mapping": mapping, "first_sequence": offset,
                                     "row_count": len(rows), "recovered_fields": sorted(recovered),
                                     "missing_fields": sorted(set(FIELDS) - recovered)})
    result = canonicalize_trace(result)
    atomic_write(target, json_bytes(result))
    return result
