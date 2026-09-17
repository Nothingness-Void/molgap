"""Hardware-native cost ledger for V5 server experiments."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


LEDGER_FORMAT = "molgap-native-cost-ledger-v1"
NATIVE_UNITS = frozenset(
    {
        "T4_device_hours",
        "A100_hours",
        "DCU_hours",
        "CPU_hours",
        "wall_hours",
        "queue_hours",
    }
)
COST_KINDS = frozenset(
    {"preflight", "infrastructure_failure", "retry", "audit", "training", "acceptance"}
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


@dataclass(frozen=True)
class CostEntry:
    run_id: str
    kind: str
    unit: str
    amount: float
    platform: str
    recorded_at: Optional[str] = None
    note: str = ""
    entry_id: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        if not self.run_id or self.kind not in COST_KINDS:
            raise ValueError("run_id and a recognized native cost kind are required")
        if self.unit not in NATIVE_UNITS:
            raise ValueError(f"Unsupported native unit: {self.unit}")
        if not math.isfinite(float(self.amount)) or self.amount < 0:
            raise ValueError("Cost amount must be finite and non-negative")
        payload = {
            "run_id": self.run_id,
            "kind": self.kind,
            "unit": self.unit,
            "amount": float(self.amount),
            "platform": self.platform,
            "recorded_at": self.recorded_at or _now(),
            "note": self.note,
        }
        payload["entry_id"] = self.entry_id or hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:24]
        return payload


class NativeCostLedger:
    """Append-only, idempotent JSON ledger with no cross-unit conversion."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"format": LEDGER_FORMAT, "entries": {}}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid cost ledger: {self.path}") from exc
        if data.get("format") != LEDGER_FORMAT or not isinstance(data.get("entries"), dict):
            raise ValueError("Cost ledger format is not supported")
        return data

    def _write(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary: Optional[Path] = None
        try:
            fd, name = tempfile.mkstemp(prefix=f".{self.path.name}.", suffix=".tmp", dir=str(self.path.parent))
            temporary = Path(name)
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(data, handle, indent=2, sort_keys=True, ensure_ascii=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
            temporary = None
        finally:
            if temporary is not None:
                try:
                    temporary.unlink()
                except FileNotFoundError:
                    pass

    def record(self, entry: CostEntry) -> dict[str, Any]:
        data = self._load()
        normalized = entry.to_dict()
        existing = data["entries"].get(normalized["entry_id"])
        if existing is not None:
            return existing
        data["entries"][normalized["entry_id"]] = normalized
        self._write(data)
        return normalized

    def entries(self) -> list[dict[str, Any]]:
        return sorted(self._load()["entries"].values(), key=lambda item: item["entry_id"])

    def totals_by_native_unit(self) -> dict[str, float]:
        totals = {unit: 0.0 for unit in sorted(NATIVE_UNITS)}
        for entry in self.entries():
            totals[entry["unit"]] += float(entry["amount"])
        return totals
