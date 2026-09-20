"""Observed-only canonical traces and a single-writer, atomic recorder."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

from .schemas import validate_id

TRACE_FORMAT = "molgap-trace-v1"
METRICS = ("live_train_metric", "live_dev_metric", "ema_dev_metric")
NUMBERS = ("optimizer_step", "sample_presentations", "epoch_or_pass", "learning_rate",
           *METRICS, "wall_time_seconds", "cumulative_wall_time_seconds",
           "device_time_seconds", "cumulative_device_time_seconds")
FIELDS = (*NUMBERS, "checkpoint_identity")


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def file_digest(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: str | Path, payload: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        sync_directory(path.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def sync_directory(path: Path) -> None:
    # Windows does not expose directory fsync through the standard library.
    if os.name != "nt":
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def validate_canonical_trace(record: dict[str, Any]) -> dict[str, Any]:
    if record.get("schema") != TRACE_FORMAT:
        raise ValueError("unsupported canonical trace schema")
    for field in ("trajectory_id", "run_id"):
        if not isinstance(record.get(field), str) or not record[field].strip():
            raise ValueError(f"missing trace {field}")
    validate_id(record["trajectory_id"], "trace.trajectory_id")
    semantics = record.get("metric_semantics")
    if not isinstance(semantics, dict) or set(semantics) != set(METRICS):
        raise ValueError("every metric needs explicit semantics or null")
    for name, definition in semantics.items():
        if definition is None:
            continue
        if not isinstance(definition, dict) or any(
            not isinstance(definition.get(field), str) or not definition[field].strip()
            for field in ("metric", "unit", "target", "role_identity", "weights", "direction")
        ):
            raise ValueError(f"incomplete metric semantics: {name}")
        if definition["weights"] != ("ema" if name.startswith("ema_") else "live"):
            raise ValueError("live/EMA weight semantics mismatch")
        if definition["direction"] not in {"minimize", "maximize"}:
            raise ValueError("invalid metric direction")
    rows = record.get("observations")
    if not isinstance(rows, list):
        raise ValueError("observations must be ordered array")
    last: dict[str, float] = {}
    terminal = False
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or isinstance(row.get("sequence"), bool) or not isinstance(row.get("sequence"), int) or row.get("sequence") != index:
            raise ValueError("observation sequence must be contiguous and ordered")
        if terminal or row.get("event") not in {"observation", "checkpoint", "resume", "terminal"}:
            raise ValueError("invalid event or observation after terminal")
        terminal = row["event"] == "terminal"
        if not set(FIELDS) <= set(row):
            raise ValueError("missing fields must be explicitly null")
        for field in NUMBERS:
            value = row[field]
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"non-finite/non-numeric {field}")
            if field not in METRICS and value < 0:
                raise ValueError(f"negative {field}")
            if field in {"optimizer_step", "sample_presentations"} and not isinstance(value, int):
                raise ValueError(f"{field} must be integer")
            if field in {"optimizer_step", "sample_presentations", "cumulative_wall_time_seconds",
                         "cumulative_device_time_seconds"}:
                if value < last.get(field, 0):
                    raise ValueError(f"non-monotonic {field}")
                last[field] = value
            if field in METRICS and semantics[field] is None:
                raise ValueError(f"observed metric without semantics: {field}")
            if "device_time_seconds" in field and record.get("device_time_semantics") != "sum_over_devices":
                raise ValueError("observed device time requires sum_over_devices semantics")
        checkpoint = row["checkpoint_identity"]
        if checkpoint is not None and (not isinstance(checkpoint, str) or not checkpoint.strip()):
            raise ValueError("invalid checkpoint identity")
        if row["event"] in {"checkpoint", "resume"} and checkpoint is None:
            raise ValueError("checkpoint/resume requires observed checkpoint identity")
    return record


def canonicalize_trace(record: dict[str, Any]) -> dict[str, Any]:
    """Complete absent observations with null; never sort or infer measurements."""
    result = copy.deepcopy(record)
    result.setdefault("schema", TRACE_FORMAT)
    result.setdefault("metric_semantics", {field: None for field in METRICS})
    result.setdefault("observations", [])
    for index, row in enumerate(result["observations"]):
        row.setdefault("sequence", index)
        row.setdefault("event", "observation")
        for field in FIELDS:
            row.setdefault(field, None)
    return validate_canonical_trace(result)


def load_canonical_trace(path: str | Path) -> dict[str, Any]:
    return validate_canonical_trace(json.loads(Path(path).read_text(encoding="utf-8")))


def trace_digest(trace: dict[str, Any] | str | Path) -> str:
    """Paths hash exact artifact bytes; objects hash canonical serialization."""
    if isinstance(trace, (str, Path)):
        load_canonical_trace(trace)
        return file_digest(trace)
    return hashlib.sha256(json_bytes(canonicalize_trace(trace))).hexdigest()


def validate_manifest_trace(manifest: dict[str, Any], trace: dict[str, Any]) -> None:
    validate_canonical_trace(trace)
    if any(manifest[k] != trace[k] for k in ("trajectory_id", "run_id")):
        raise ValueError("manifest/canonical trace identity mismatch")
    for axis, exposure in (("optimizer_step", "optimizer_steps"), ("sample_presentations", "sample_presentations")):
        observed = [r[axis] for r in trace["observations"] if r[axis] is not None]
        terminal = manifest["exposure"].get(exposure)
        if observed and terminal is not None and max(observed) > terminal:
            raise ValueError("observations exceed manifest exposure")
    for field, declared in manifest.get("trace_fields", {}).items():
        if not declared and any(r[field] is not None for r in trace["observations"]):
            raise ValueError(f"manifest declares observed field unavailable: {field}")


class RMLTraceRecorder:
    """One writer per path; each successful append is a durable full snapshot.

    On write failure the in-memory state remains unchanged. Reopen after a
    process failure; resume events never reset cumulative counters.
    """

    def __init__(self, path: str | Path, *, trajectory_id: str, run_id: str,
                 metric_semantics: dict[str, Any], device_time_semantics: str | None = None):
        self.path = Path(path)
        self.record = canonicalize_trace({"trajectory_id": trajectory_id, "run_id": run_id,
                                         "metric_semantics": metric_semantics,
                                         "device_time_semantics": device_time_semantics})
        if self.path.exists():
            existing = load_canonical_trace(self.path)
            for key in ("trajectory_id", "run_id", "metric_semantics", "device_time_semantics"):
                if existing.get(key) != self.record.get(key):
                    raise ValueError(f"resume identity mismatch: {key}")
            self.record = existing
        else:
            self.flush()

    def flush(self) -> None:
        atomic_write(self.path, json_bytes(validate_canonical_trace(self.record)))

    def append_observation(self, *, event: str = "observation", **fields: Any) -> None:
        if set(fields) - set(FIELDS):
            raise ValueError("unknown observation fields")
        candidate = copy.deepcopy(self.record)
        candidate["observations"].append({"sequence": len(candidate["observations"]),
                                           "event": event, **fields})
        candidate = canonicalize_trace(candidate)
        atomic_write(self.path, json_bytes(candidate))
        self.record = candidate

    def checkpoint_event(self, checkpoint_identity: str, **fields: Any) -> None:
        self.append_observation(event="checkpoint", checkpoint_identity=checkpoint_identity, **fields)

    def resume_event(self, checkpoint_identity: str, **fields: Any) -> None:
        self.append_observation(event="resume", checkpoint_identity=checkpoint_identity, **fields)

    def terminal_event(self, **fields: Any) -> None:
        self.append_observation(event="terminal", **fields)
