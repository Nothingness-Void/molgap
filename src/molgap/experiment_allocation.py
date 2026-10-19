"""Physical allocation accounting, independent of families and training metrics."""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
from pathlib import Path
import time

from .training_reproducibility import atomic_json


def _prior_segments(records, spec_identity):
    from .screen_policy import canonical_fingerprint
    pending, flattened, seen = list(records), [], set()
    while pending:
        segment = pending.pop(0)
        if (type(segment) is not dict or segment.get("spec_identity") != spec_identity
                or segment.get("format") != "molgap-allocation-ledger-v1"):
            raise ValueError("Prior allocation segment identity differs")
        # Each record measures one invocation. Flatten ancestry so repeated
        # recovery does not duplicate costs or grow a recursive manifest.
        children = segment.get("prior_segments", [])
        if type(children) is not list:
            raise ValueError("Prior allocation ancestry must be a list")
        canonical_fingerprint(segment)  # Reject cyclic/nonfinite input before traversal.
        retained = {**segment, **({"prior_segments": []} if "prior_segments" in segment else {})}
        identity = canonical_fingerprint(retained)
        if identity not in seen:
            seen.add(identity)
            flattened.append(retained)
            pending.extend(children)
    return flattened


class AllocationLedger:
    """Account for every allocated device, including unassigned and idle time."""

    def __init__(self, *, spec_identity, hardware, assignments, started=None,
                 prior_segments=(), clock=time.perf_counter):
        if not hardware or any(type(name) is not str or not name for name in hardware):
            raise ValueError("Allocation requires observed hardware")
        if (type(assignments) is not dict or len(set(assignments.values())) != len(assignments)
                or any(type(i) is not int or not 0 <= i < len(hardware) for i in assignments.values())):
            raise ValueError("Invalid physical allocation assignments")
        self.clock = clock
        self.started = clock() if started is None else started
        if self.started > clock():
            raise ValueError("Allocation start is in the future")
        self.started_at = (datetime.now(timezone.utc) - timedelta(seconds=clock() - self.started)).isoformat()
        self.spec_identity = spec_identity
        self.hardware = tuple(hardware)
        self.assignments = dict(assignments)
        self.prior_segments = _prior_segments(prior_segments, spec_identity)

    def snapshot(self, status):
        if status not in {"running", "complete", "failed"}:
            raise ValueError("Invalid allocation terminal status")
        elapsed = max(0.0, self.clock() - self.started)
        assigned = {device: arm for arm, device in self.assignments.items()}
        return {"format": "molgap-allocation-ledger-v1", "spec_identity": self.spec_identity,
                "started_at": self.started_at, "observed_at": datetime.now(timezone.utc).isoformat(),
                "status": status, "allocation_device_count": len(self.hardware),
                "wall_seconds": elapsed, "allocated_device_seconds": len(self.hardware) * elapsed,
                "unassigned_device_seconds": (len(self.hardware) - len(assigned)) * elapsed,
                "devices": [{"device": i, "hardware": name, "arm_id": assigned.get(i),
                             "allocated_seconds": elapsed} for i, name in enumerate(self.hardware)],
                "scope": "Python bootstrap/runtime observation window; includes idle allocated devices",
                "provisioning_before_python_seconds": {"value": None, "status": "measurement_missing"},
                "queue_seconds": {"value": None, "status": "measurement_missing"},
                "prior_segments": self.prior_segments,
                "prior_unobserved_intervals": any(
                    s.get("status") == "running" or s.get("prior_unobserved_intervals") is True
                    or set(s) == {"format", "spec_identity", "status"}
                    for s in self.prior_segments)}

    def write(self, root: Path, status, *, arm_roots=()):
        record = self.snapshot(status)
        atomic_json(Path(root) / "allocation_ledger.json", record)
        for arm_root in arm_roots:
            atomic_json(Path(arm_root) / "allocation_ledger.json", record)
        return record
