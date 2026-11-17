"""Host-neutral declaration mechanisms; experiment profiles supply all facts."""
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
import math
import re

from .wire import canonical_fingerprint


def safe_comparison_base(value, *, base, require_under_base=True):
    if isinstance(value, Path):
        value = value.as_posix()
    if type(value) is not str or not value or "\\" in value or ":" in value:
        raise ValueError("Comparison base must be a repository-relative POSIX path")
    if value.startswith("/") or PureWindowsPath(value).drive or PureWindowsPath(value).root:
        raise ValueError("Comparison base must be relative")
    if any(not p or p in {".", ".."} or p.endswith((" ", "."))
           or PureWindowsPath(p).is_reserved() for p in value.split("/")):
        raise ValueError("Comparison base contains an unsafe path component")
    if require_under_base and value != base and not value.startswith(base + "/"):
        raise ValueError("Comparison base must be under the frozen comparison root")
    return value


def run_version(value):
    if type(value) is not str or re.fullmatch(r"v[1-9][0-9]*", value) is None:
        raise ValueError("Comparison run_version must be a version such as v1 or v2")
    return value


def comparison_base(spec, *, experiment_id, base, arm_ids):
    declaration = spec.to_dict()
    if declaration.get("experiment_id") != experiment_id:
        raise ValueError("Not the frozen execution comparison")
    arms, prospective = declaration.get("arms"), declaration.get("prospective")
    ids = list(arm_ids)
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Comparison arm IDs must be explicit and unique")
    if (type(arms) is not list or [a.get("arm_id") for a in arms] != ids
            or type(prospective) is not dict or type(prospective.get("arms")) is not list
            or [a.get("arm_id") for a in prospective["arms"]] != ids):
        raise ValueError("Comparison Spec must declare the profile prospective arms")
    bases = []
    for item, arm_id in zip(prospective["arms"], ids):
        path = item.get("plan_spec_ref")
        safe_comparison_base(path, base=base, require_under_base=False)
        suffix = "/" + arm_id + "/plan_input.json"
        if not path.endswith(suffix):
            raise ValueError("Comparison plan reference has an unexpected arm suffix")
        bases.append(safe_comparison_base(path[:-len(suffix)], base=base))
    if len(set(bases)) != 1:
        raise ValueError("Comparison prospective arms do not share a storage root")
    return bases[0]


@dataclass(frozen=True)
class ExecutionQuestion:
    policy_id: str
    action: str
    question: str
    deficiency: str
    changed_mechanism: str
    supporting_evidence: tuple[str, ...]
    alternatives: tuple[str, ...]
    closed_families: tuple[str, ...]
    positive_decision: str
    negative_decision: str
    reference_ids: tuple[str, ...] = ()
    prior_trajectories: tuple[str, ...] = ()


@dataclass(frozen=True)
class ExecutionProfile:
    """Explicit question facts, not execution permission or platform defaults."""

    owner: str
    track: str
    platform: str
    hardware: str
    cheapest_falsifier: str
    estimated_device_hours: float | None
    estimated_wall_hours: float | None

    def __post_init__(self):
        if self.owner not in {"desktop", "server"} or self.track not in {"A", "B", "C"}:
            raise ValueError("Comparison ownership and track must be explicit")
        for name in ("platform", "hardware", "cheapest_falsifier"):
            if type(getattr(self, name)) is not str or not getattr(self, name).strip():
                raise ValueError(name + " must be explicit")
        for value in (self.estimated_device_hours, self.estimated_wall_hours):
            if value is not None and (type(value) not in {int, float}
                                      or not math.isfinite(value) or value < 0):
                raise ValueError("Estimated costs must be finite nonnegative numbers or None")


def _estimated(value):
    return {"status": "measurement_missing" if value is None else "estimated", "value": value}


def build_execution_proposal(*, question, arm, trajectory_id, source_commit, run, version,
                       contract_ref, contract_refs, authority_ref, role_ref, decision_ref,
                       timestamp, profile: ExecutionProfile, action_type="execution_diagnostic"):
    """Translate small question facts to the existing prospective RML schema."""
    run_version(version)
    tid, cost = trajectory_id, "cost-" + trajectory_id + "-planned"
    run_id, attempt = run + ":" + arm["arm_id"] + ":downstream", arm["arm_id"] + "-" + version
    q = question
    return {"trajectory": {"trajectory_id": tid, "record_mode": "prospective", "owner": profile.owner, "track": profile.track,
        "family_id": q.policy_id, "question": q.question,
        "hypothesis": {"hypothesis_id": "H-" + tid, "observed_deficiency": q.deficiency,
            "supporting_evidence_ids": list(q.supporting_evidence), "alternative_explanations": list(q.alternatives),
            "changed_mechanism": q.changed_mechanism,
            "cheapest_falsifier": profile.cheapest_falsifier,
            "related_closed_family_ids": list(q.closed_families), "expected_native_cost_ref": cost,
            "decision_changed_if_positive": q.positive_decision, "decision_changed_if_negative": q.negative_decision,
            "historical_unknowns": []},
        "state_at_start": {"source_commit": source_commit, "source_config_identity": canonical_fingerprint(arm),
            "contract_refs": list(contract_refs), "reference_ids": list(q.reference_ids),
            "parent_trajectory_ids": [], "prior_trajectory_ids": list(q.prior_trajectories),
            "prior_evidence_ids": list(q.supporting_evidence), "role_snapshot_refs": [role_ref],
            "budget_snapshot_ref": authority_ref},
        "actions": [{"action_id": "A001", "type": action_type, "source_commit": source_commit,
            "run_ids": [run_id], "attempt_ids": [attempt], "evidence_refs": [contract_ref], "cost_event_ids": [cost]}],
        "result": {"evidence_ids": [], "evidence_refs": []}, "decision": {"decision_ref": decision_ref,
            "outcome": "ACTIVE", "next_allowed_actions": [
                "one authorized execution diagnostic" if action_type == "execution_diagnostic"
                else "one authorized frozen experiment"],
            "reopen_conditions": ["terminal evidence"]}},
        "decision_state": {"known_trajectory_ids": [], "known_evidence_ids": [], "active_reference_ids": list(q.reference_ids),
            "available_actions": [q.action, "DEFER"], "chosen_action": q.action, "policy_id": q.policy_id,
            "policy_version": "v1", "budget_snapshot_ref": authority_ref, "role_snapshot_refs": [role_ref],
            "state_timestamp": timestamp, "source_commit": source_commit},
        "costs": [{"schema": "molgap-cost-event-v1", "cost_event_id": cost, "trajectory_id": tid,
            "action_id": "A001", "run_id": run_id, "attempt_id": attempt, "category": "training", "platform": profile.platform,
            "hardware": profile.hardware, "evidence_ref": authority_ref,
            "measurement": {"device_hours": _estimated(profile.estimated_device_hours),
                "wall_hours": _estimated(profile.estimated_wall_hours),
                "cpu_hours": {"status": "measurement_missing", "value": None},
                "queue_hours": {"status": "measurement_missing", "value": None}}}]}
