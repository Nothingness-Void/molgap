"""Physical allocation accounting, independent of families and training metrics."""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
import hashlib
import json
import math
from pathlib import Path
import time

from .training_reproducibility import atomic_json


def _bound_json(reference):
    """Read exactly the caller's retained bytes, independent of a mutable path."""
    from .experiment_retention import _digest
    if type(reference) is not dict or set(reference) != {"path", "sha256"}:
        raise ValueError("Physical allocation evidence requires path and sha256")
    path = Path(reference["path"])
    if not path.is_absolute():
        raise ValueError("Physical allocation evidence paths must be absolute")
    expected = _digest(reference["sha256"], "physical allocation evidence SHA256")
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError(f"Physical allocation evidence SHA mismatch: {path}")
    value = json.loads(payload)
    if type(value) is not dict:
        raise ValueError("Physical allocation evidence must be an object")
    return value


def current_physical_allocation_records(resolved, retention, observation_path):
    """Bind one explicitly observed physical entry to the resolved run.

    The caller identifies the observation; its origin is not inferred here.
    Multi-job or historical allocation accounting still uses explicit records.
    """
    if retention is None:
        raise ValueError("Physical observation requires verified execution retention")
    receipt_path = Path(resolved["options"]["receipt_path"]).absolute()
    receipt_bytes = receipt_path.read_bytes()
    receipt = json.loads(receipt_bytes)
    contexts = list(resolved["contexts"].values())
    runs = receipt.get("physical_runs", {}).get("value") or []
    matches = [run for run in runs if contexts and all(
        context.platform == receipt.get("binding", {}).get("requested_logical_platform")
        and context.run_reference == run.get("canonical_reference")
        and context.platform_version == run.get("platform_version", {}).get("value")
        and context.arm_id in run.get("arm_ids", []) for context in contexts)]
    if len(matches) != 1:
        raise ValueError("Physical observation requires one matching job/version for every arm")
    observation = Path(observation_path).absolute()
    report = retention["manifest"]["execution_report"]
    report_path = Path(retention["manifest_path"]).parent / report["path"]
    return [{"platform": contexts[0].platform, "physical_run": matches[0],
        "receipt": {"path": str(receipt_path), "sha256": hashlib.sha256(receipt_bytes).hexdigest()},
        "observation": {"path": str(observation), "sha256": hashlib.sha256(observation.read_bytes()).hexdigest()},
        "execution_report": {"path": str(report_path), "sha256": report["sha256"]}}]


def aggregate_physical_allocation_costs(records):
    """Count explicit physical job/version entry windows once, including failures.

    Each record supplies platform, physical_run (the existing launch-receipt
    item), and SHA-bound receipt, observation and execution_report files.
    Hardware comes from the runtime report; elapsed time comes only from the
    measured entry observation. Inherited Python-ledger segments are never
    added to that wider entry window. This is read-only cost accounting, not
    authentication of caller-supplied bindings or scientific acceptance.
    """
    from .experiment_retention import _finite, _timestamp
    from .research_memory.cost import build_cost_ledger
    from .screen_policy import canonical_fingerprint

    if type(records) is not list or not records:
        raise ValueError("Physical allocations require a nonempty explicit record list")
    windows, events, seen, duplicate_count = [], [], {}, 0
    for record in records:
        if type(record) is not dict or set(record) != {
                "platform", "physical_run", "receipt", "observation", "execution_report"}:
            raise ValueError("Physical allocation record fields are incomplete or unknown")
        platform, physical = record["platform"], record["physical_run"]
        if type(platform) is not str or not platform or type(physical) is not dict:
            raise ValueError("Physical allocation requires explicit platform and physical run")
        version = physical.get("platform_version", {}).get("value")
        identity, canonical = physical.get("run_identity"), physical.get("canonical_reference")
        if any(type(value) is not str or not value for value in (version, identity, canonical)):
            raise ValueError("Physical allocation job/version identity is missing")
        receipt = _bound_json(record["receipt"])
        if (receipt.get("format") != "molgap-experiment-launch-receipt"
                or physical not in (receipt.get("physical_runs", {}).get("value") or [])
                or receipt.get("binding", {}).get("requested_logical_platform") != platform):
            raise ValueError("Physical allocation differs from its retained launch receipt")
        observation = _bound_json(record["observation"])
        execution = _bound_json(record["execution_report"])
        spec_identity = receipt["binding"].get("spec_identity")
        if (type(spec_identity) is not str or not spec_identity
                or execution.get("format") != "molgap-kaggle-two-phase-pair-v2"
                or execution.get("spec_identity") != spec_identity):
            raise ValueError("Physical allocation runtime report differs from receipt Spec")
        if (observation.get("format") != "molgap-kaggle-allocation-entry-v1"
                or observation.get("status") not in {"running", "complete", "failed"}):
            raise ValueError("Physical allocation requires a retained entry observation")
        start = _timestamp(observation.get("wall_start_utc"), "entry wall_start_utc")
        end = _timestamp(observation.get("wall_end_utc"), "entry wall_end_utc")
        if end < start:
            raise ValueError("Physical allocation entry ends before it starts")
        elapsed = _finite(observation.get("monotonic_seconds"), "entry monotonic_seconds")
        count, seconds = observation.get("observed_gpu_count"), observation.get("allocated_device_seconds")
        hardware = execution.get("hardware")
        if (type(hardware) is not list or not hardware
                or any(type(name) is not str or not name or name.lower() in {
                    "unknown", "unspecified", "unavailable"} for name in hardware)):
            raise ValueError("Physical allocation runtime hardware is missing")
        if count is None:
            if seconds is not None:
                raise ValueError("Unobserved physical allocation count cannot have device seconds")
        else:
            if (type(count) is not int or count <= 0 or count != len(hardware)
                    or observation.get("gpu_count_source") != "nvidia-smi"):
                raise ValueError("Physical allocation observed device count differs from runtime")
            seconds = _finite(seconds, "entry allocated_device_seconds")
            if not math.isclose(seconds, count * elapsed, rel_tol=1e-9, abs_tol=1e-6):
                raise ValueError("Physical allocation device seconds disagree with observed count")
        # Equal copied arm/root evidence must not charge the same physical
        # window twice. A changed observation for that identity needs review.
        key = (platform, identity, version)
        fingerprint = canonical_fingerprint({"observation": observation, "hardware": hardware})
        if key in seen:
            if seen[key] != fingerprint:
                raise ValueError("Conflicting observations for one physical allocation job/version")
            duplicate_count += 1
            continue
        seen[key] = fingerprint
        window = {"platform": platform, "physical_run": physical,
            "status": observation["status"], "entry_wall_seconds": elapsed,
            "device_hours": None if seconds is None else seconds / 3600,
            "measurement_status": "measurement_missing" if seconds is None else "measured",
            "scope": observation.get("scope"),
            "platform_release_scope": observation.get("platform_release_scope"),
            "evidence": {name: record[name] for name in ("receipt", "observation", "execution_report")}}
        windows.append(window)
        for name in sorted(set(hardware)):
            measurement = {unit: {"value": None, "status": "measurement_missing"}
                for unit in ("device_hours", "cpu_hours", "wall_hours", "queue_hours")}
            if seconds is not None:
                measurement["device_hours"] = {"value": hardware.count(name) * elapsed / 3600,
                                               "status": "measured"}
            events.append({"cost_event_id": canonical_fingerprint({"physical_key": key, "hardware": name}),
                "trajectory_id": canonical_fingerprint({"physical_key": key}), "hardware": name,
                "category": "infrastructure_failure" if observation["status"] == "failed" else "other",
                "measurement": measurement})
    return {"windows": windows, "duplicates_ignored": duplicate_count,
        "cost_ledger": build_cost_ledger(events),
        "accounting": "Physical entry windows counted once per job/version; inherited ledger segments excluded. Queue, pre-entry allocation and post-entry release remain unmeasured."}


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
