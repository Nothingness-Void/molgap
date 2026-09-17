"""Durable, server-local control state for the MolGap V5 A/B loop.

This module contains operational state only.  It does not choose a candidate,
interpret a metric, submit a job, or inspect desktop-owned work.  A JSON file is
used deliberately: V5 permits atomic local files, and keeping the state
inspectable makes recovery and synthetic testing straightforward.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional


CONTROL_FORMAT = "molgap-server-control-v5"
OWNER_SCOPE = "server"
HEALTHY_STATUSES = frozenset({"QUEUED", "RUNNING"})
UNKNOWN_STATUSES = frozenset({"UNKNOWN", "API_UNKNOWN", "STATUS_UNKNOWN"})
EVENT_STATUSES = frozenset({"COMPLETE", "FAILED", "ERROR", "CANCELLED"})
FINAL_EVENT_STATES = frozenset(
    {"NEXT_RUN_BOUND", "PAUSED", "CLOSED", "READY_FOR_DESKTOP"}
)
PENDING_EVENT_STATES = frozenset(
    {"EVENT_DURABLE", "DELIVERED_TO_A", "A_CLAIMED", "DECISION_COMMITTED"}
)


class ControlStateError(RuntimeError):
    """Base error for invalid or unsafe local control-state transitions."""


class StaleGenerationError(ControlStateError):
    """A monitor tried to mutate a binding that has already been replaced."""


class EventClaimedError(ControlStateError):
    """A different A consumer already owns an event claim."""


class StateConflictError(ControlStateError):
    """A new binding would discard unresolved durable state."""


def utc_timestamp() -> str:
    """Return a compact UTC timestamp suitable for durable JSON evidence."""

    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _copy_mapping(value: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    return dict(value or {})


@dataclass(frozen=True)
class BoundRun:
    """The compact identity B is allowed to observe and mutate."""

    campaign_id: str
    chain_id: str
    run_id: str
    attempt_id: str
    a_thread_id: str
    b_thread_id: str
    monitor_generation: int
    remote_platform: str
    remote_job_identity: Mapping[str, Any]
    release_identity: str
    reference_identity: Optional[str] = None
    budget_reserved_native: Mapping[str, float] = field(default_factory=dict)
    decision_ref: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        if self.monitor_generation < 1:
            raise ValueError("monitor_generation must be positive")
        required = {
            "campaign_id": self.campaign_id,
            "chain_id": self.chain_id,
            "run_id": self.run_id,
            "attempt_id": self.attempt_id,
            "A_thread_id": self.a_thread_id,
            "B_thread_id": self.b_thread_id,
            "monitor_generation": self.monitor_generation,
            "remote_platform": self.remote_platform,
            "remote_job_identity": _copy_mapping(self.remote_job_identity),
            "release_identity": self.release_identity,
            "reference_identity": self.reference_identity,
            "budget_reserved_native": _copy_mapping(self.budget_reserved_native),
            "decision_ref": self.decision_ref,
            "owner_scope": OWNER_SCOPE,
        }
        for key in (
            "campaign_id",
            "chain_id",
            "run_id",
            "attempt_id",
            "A_thread_id",
            "B_thread_id",
            "remote_platform",
            "release_identity",
        ):
            if not isinstance(required[key], str) or not required[key]:
                raise ValueError(f"{key} must be a non-empty string")
        return required


@dataclass(frozen=True)
class ObservationResult:
    """Mechanical result returned to B; no scientific action is implied."""

    action: str
    status: str
    event_id: Optional[str] = None


def _empty_state() -> dict[str, Any]:
    return {
        "format": CONTROL_FORMAT,
        "owner_scope": OWNER_SCOPE,
        "version": 1,
        "binding": None,
        "last_observation": None,
        "events": {},
        "last_decision_ref": None,
    }


class LocalServerControlStore:
    """Atomic local state store shared by the server's A and B conversations."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return _empty_state()
        try:
            state = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ControlStateError(f"Cannot read control state: {self.path}") from exc
        if not isinstance(state, dict):
            raise ControlStateError("Control state must be a JSON object")
        if state.get("format") != CONTROL_FORMAT or state.get("owner_scope") != OWNER_SCOPE:
            raise ControlStateError("Control state is not a V5 server-local record")
        if not isinstance(state.get("events"), dict):
            raise ControlStateError("Control state events must be an object")
        return state

    def _write(self, state: Mapping[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary: Optional[Path] = None
        try:
            fd, name = tempfile.mkstemp(
                prefix=f".{self.path.name}.", suffix=".tmp", dir=str(self.path.parent)
            )
            temporary = Path(name)
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(state, handle, indent=2, sort_keys=True, ensure_ascii=True)
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

    def snapshot(self) -> dict[str, Any]:
        """Return a validated copy of the current state."""

        return json.loads(json.dumps(self.load()))

    def bind_run(self, run: BoundRun) -> dict[str, Any]:
        """Bind B to one server-owned run and advance its fencing generation."""

        binding = run.to_dict()
        state = self.load()
        old_binding = state.get("binding")
        if old_binding is not None and old_binding != binding:
            unresolved = [
                event
                for event in state["events"].values()
                if event.get("event_status") in PENDING_EVENT_STATES
            ]
            if unresolved:
                raise StateConflictError(
                    "Cannot replace a binding while a durable A/B event is unresolved"
                )
        state["binding"] = binding
        state["last_observation"] = None
        state["last_decision_ref"] = binding.get("decision_ref")
        self._write(state)
        return binding

    def _require_generation(
        self, state: Mapping[str, Any], *, run_id: str, attempt_id: str, monitor_generation: int
    ) -> Mapping[str, Any]:
        binding = state.get("binding")
        if not isinstance(binding, Mapping):
            raise ControlStateError("No server-owned run is bound")
        expected = (binding.get("run_id"), binding.get("attempt_id"), binding.get("monitor_generation"))
        actual = (run_id, attempt_id, monitor_generation)
        if expected != actual:
            raise StaleGenerationError(
                f"Stale monitor generation: expected {expected}, received {actual}"
            )
        return binding

    @staticmethod
    def _event_id(
        binding: Mapping[str, Any], event_type: str, remote_state_version: Optional[str]
    ) -> str:
        payload = {
            "campaign_id": binding["campaign_id"],
            "chain_id": binding["chain_id"],
            "run_id": binding["run_id"],
            "attempt_id": binding["attempt_id"],
            "monitor_generation": binding["monitor_generation"],
            "event_type": event_type,
            # A terminal state is one event per bound attempt.  Scheduler/API
            # version strings may change between duplicate observations and
            # must not manufacture a second successor path.
            "remote_state_version": (
                remote_state_version if event_type not in EVENT_STATUSES else "terminal"
            )
            or "terminal",
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return f"evt-{digest[:24]}"

    def _persist_event(
        self,
        state: dict[str, Any],
        *,
        event_type: str,
        event_id: str,
        payload: Optional[Mapping[str, Any]],
        observed_at: str,
    ) -> dict[str, Any]:
        events = state["events"]
        existing = events.get(event_id)
        if existing is not None:
            return existing
        event = {
            "event_id": event_id,
            "event_type": event_type,
            "event_status": "EVENT_DURABLE",
            "created_at": observed_at,
            "monitor_generation": state["binding"]["monitor_generation"],
            "run_id": state["binding"]["run_id"],
            "attempt_id": state["binding"]["attempt_id"],
            "delivery_attempts": 0,
            "claimed_by": None,
            "claimed_at": None,
            "decision_ref": None,
            "outcome": None,
            "payload": _copy_mapping(payload),
        }
        events[event_id] = event
        return event

    def observe(
        self,
        *,
        run_id: str,
        attempt_id: str,
        monitor_generation: int,
        remote_status: str,
        owner_scope: str = OWNER_SCOPE,
        remote_state_version: Optional[str] = None,
        detail: Optional[Mapping[str, Any]] = None,
        observed_at: Optional[str] = None,
    ) -> ObservationResult:
        """Record one authoritative status observation using B-only semantics."""

        status = str(remote_status).upper()
        if owner_scope != OWNER_SCOPE:
            return ObservationResult(action="IGNORED", status=status)
        state = self.load()
        binding = self._require_generation(
            state,
            run_id=run_id,
            attempt_id=attempt_id,
            monitor_generation=monitor_generation,
        )
        timestamp = observed_at or utc_timestamp()
        state["last_observation"] = {
            "status": status,
            "observed_at": timestamp,
            "remote_state_version": remote_state_version,
            "detail": _copy_mapping(detail),
        }
        if status in HEALTHY_STATUSES:
            self._write(state)
            return ObservationResult(action="SILENT", status=status)
        if status in UNKNOWN_STATUSES:
            self._write(state)
            return ObservationResult(action="UNKNOWN", status=status)
        if status not in EVENT_STATUSES:
            raise ValueError(f"Unsupported authoritative remote status: {remote_status}")
        event_id = self._event_id(binding, status, remote_state_version)
        event = self._persist_event(
            state,
            event_type=status,
            event_id=event_id,
            payload=detail,
            observed_at=timestamp,
        )
        self._write(state)
        return ObservationResult(
            action="EVENT_DURABLE", status=status, event_id=event["event_id"]
        )

    def record_submit_unknown(
        self,
        *,
        run_id: str,
        attempt_id: str,
        monitor_generation: int,
        detail: Optional[Mapping[str, Any]] = None,
        observed_at: Optional[str] = None,
    ) -> ObservationResult:
        """Persist an ambiguous submission without authorizing a blind retry."""

        state = self.load()
        binding = self._require_generation(
            state,
            run_id=run_id,
            attempt_id=attempt_id,
            monitor_generation=monitor_generation,
        )
        timestamp = observed_at or utc_timestamp()
        payload = _copy_mapping(detail)
        payload["retry_allowed"] = False
        event_id = self._event_id(binding, "SUBMIT_UNKNOWN", None)
        event = self._persist_event(
            state,
            event_type="SUBMIT_UNKNOWN",
            event_id=event_id,
            payload=payload,
            observed_at=timestamp,
        )
        self._write(state)
        return ObservationResult(
            action="EVENT_DURABLE", status="SUBMIT_UNKNOWN", event_id=event["event_id"]
        )

    def deliver_to_a(self, event_id: str) -> dict[str, Any]:
        """Record delivery acknowledgement; it is not A's processing acknowledgement."""

        state = self.load()
        event = state["events"].get(event_id)
        if event is None:
            raise ControlStateError(f"Unknown event: {event_id}")
        if event["event_status"] == "EVENT_DURABLE":
            event["event_status"] = "DELIVERED_TO_A"
            event["delivery_attempts"] = int(event.get("delivery_attempts", 0)) + 1
            event["delivered_at"] = utc_timestamp()
            self._write(state)
        return event

    def claim_event(self, event_id: str, a_thread_id: str) -> dict[str, Any]:
        """Claim one event idempotently for the existing server A."""

        state = self.load()
        event = state["events"].get(event_id)
        if event is None:
            raise ControlStateError(f"Unknown event: {event_id}")
        binding = state.get("binding") or {}
        if event["event_status"] == "A_CLAIMED" and event.get("claimed_by") != a_thread_id:
            raise EventClaimedError(f"Event is already claimed: {event_id}")
        if a_thread_id != binding.get("A_thread_id"):
            raise ControlStateError("Event can only be claimed by the bound server A")
        if event["event_status"] in FINAL_EVENT_STATES:
            return event
        if event["event_status"] == "A_CLAIMED":
            return event
        if event["event_status"] not in {"EVENT_DURABLE", "DELIVERED_TO_A"}:
            raise ControlStateError(
                f"Event cannot be claimed from {event['event_status']}: {event_id}"
            )
        event["event_status"] = "A_CLAIMED"
        event["claimed_by"] = a_thread_id
        event["claimed_at"] = utc_timestamp()
        self._write(state)
        return event

    def recover_claim(self, event_id: str, a_thread_id: str) -> dict[str, Any]:
        """Return a claimed event to the durable queue after an A crash."""

        state = self.load()
        event = state["events"].get(event_id)
        if event is None:
            raise ControlStateError(f"Unknown event: {event_id}")
        if event.get("event_status") != "A_CLAIMED" or event.get("claimed_by") != a_thread_id:
            raise ControlStateError("Only the claimed A may recover an unfinished claim")
        event["event_status"] = "EVENT_DURABLE"
        event["claimed_by"] = None
        event["claimed_at"] = None
        self._write(state)
        return event

    def commit_decision(
        self,
        event_id: str,
        *,
        a_thread_id: str,
        decision_ref: str,
        outcome: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Commit A's interpretation exactly once; B has no path to this method."""

        if not decision_ref:
            raise ValueError("decision_ref must be non-empty")
        state = self.load()
        event = state["events"].get(event_id)
        if event is None:
            raise ControlStateError(f"Unknown event: {event_id}")
        if event["event_status"] in {"DECISION_COMMITTED", *FINAL_EVENT_STATES}:
            return event
        if event["event_status"] != "A_CLAIMED" or event.get("claimed_by") != a_thread_id:
            raise ControlStateError("A must claim an event before committing a decision")
        event["event_status"] = "DECISION_COMMITTED"
        event["decision_ref"] = decision_ref
        event["outcome"] = dict(outcome)
        state["last_decision_ref"] = decision_ref
        self._write(state)
        return event

    def finalize_event(self, event_id: str, final_state: str) -> dict[str, Any]:
        """Move a decided event to one durable terminal handoff state."""

        if final_state not in FINAL_EVENT_STATES:
            raise ValueError(f"Invalid final event state: {final_state}")
        state = self.load()
        event = state["events"].get(event_id)
        if event is None:
            raise ControlStateError(f"Unknown event: {event_id}")
        if event["event_status"] in FINAL_EVENT_STATES:
            if event["event_status"] != final_state:
                raise StateConflictError("A finalized event cannot change terminal state")
            return event
        if event["event_status"] != "DECISION_COMMITTED":
            raise ControlStateError("Finalize requires a committed A decision")
        event["event_status"] = final_state
        event["finalized_at"] = utc_timestamp()
        self._write(state)
        return event

    def pending_events(self) -> list[dict[str, Any]]:
        """Return unresolved events in deterministic creation order."""

        state = self.load()
        return sorted(
            [
                event
                for event in state["events"].values()
                if event.get("event_status") in PENDING_EVENT_STATES
            ],
            key=lambda event: (event.get("created_at", ""), event.get("event_id", "")),
        )
