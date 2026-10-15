"""Shared declaration plumbing; family owners retain scientific contracts."""
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
import re

from .screen_policy import canonical_fingerprint


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


def comparison_base(spec, *, experiment_id, base):
    declaration = spec.to_dict()
    if declaration.get("experiment_id") != experiment_id:
        raise ValueError("Not the frozen execution comparison")
    arms, prospective = declaration.get("arms"), declaration.get("prospective")
    ids = ["original", "optimized"]
    if (type(arms) is not list or [a.get("arm_id") for a in arms] != ids
            or type(prospective) is not dict or type(prospective.get("arms")) is not list
            or [a.get("arm_id") for a in prospective["arms"]] != ids):
        raise ValueError("Comparison Spec must declare original and optimized prospective arms")
    bases = []
    for item, arm_id in zip(prospective["arms"], ids):
        path = item.get("plan_spec_ref")
        safe_comparison_base(path, base=base, require_under_base=False)
        suffix = "/" + arm_id + "/plan_input.json"
        if not path.endswith(suffix):
            raise ValueError("Comparison plan reference has an unexpected arm suffix")
        bases.append(safe_comparison_base(path[:-len(suffix)], base=base))
    if bases[0] != bases[1]:
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


def execution_proposal(*, question, arm, trajectory_id, source_commit, run, version,
                       contract_ref, contract_refs, authority_ref, role_ref, decision_ref,
                       timestamp, estimated_device_hours=7, estimated_wall_hours=3.5):
    """Translate small question facts to the existing prospective RML schema."""
    tid, cost = trajectory_id, "cost-" + trajectory_id + "-planned"
    run_id, attempt = run + ":" + arm["arm_id"] + ":downstream", arm["arm_id"] + "-" + version
    q = question
    return {"trajectory": {"trajectory_id": tid, "record_mode": "prospective", "owner": "server", "track": "C",
        "family_id": q.policy_id, "question": q.question,
        "hypothesis": {"hypothesis_id": "H-" + tid, "observed_deficiency": q.deficiency,
            "supporting_evidence_ids": list(q.supporting_evidence), "alternative_explanations": list(q.alternatives),
            "changed_mechanism": q.changed_mechanism,
            "cheapest_falsifier": "one matched seed42 T4x2 execution diagnostic with actual operator capture",
            "related_closed_family_ids": list(q.closed_families), "expected_native_cost_ref": cost,
            "decision_changed_if_positive": q.positive_decision, "decision_changed_if_negative": q.negative_decision,
            "historical_unknowns": []},
        "state_at_start": {"source_commit": source_commit, "source_config_identity": canonical_fingerprint(arm),
            "contract_refs": list(contract_refs), "reference_ids": list(q.reference_ids),
            "parent_trajectory_ids": [], "prior_trajectory_ids": list(q.prior_trajectories),
            "prior_evidence_ids": list(q.supporting_evidence), "role_snapshot_refs": [role_ref],
            "budget_snapshot_ref": authority_ref},
        "actions": [{"action_id": "A001", "type": "execution_diagnostic", "source_commit": source_commit,
            "run_ids": [run_id], "attempt_ids": [attempt], "evidence_refs": [contract_ref], "cost_event_ids": [cost]}],
        "result": {"evidence_ids": [], "evidence_refs": []}, "decision": {"decision_ref": decision_ref,
            "outcome": "ACTIVE", "next_allowed_actions": ["one authorized execution diagnostic"],
            "reopen_conditions": ["terminal evidence"]}},
        "decision_state": {"known_trajectory_ids": [], "known_evidence_ids": [], "active_reference_ids": list(q.reference_ids),
            "available_actions": [q.action, "DEFER"], "chosen_action": q.action, "policy_id": q.policy_id,
            "policy_version": "v1", "budget_snapshot_ref": authority_ref, "role_snapshot_refs": [role_ref],
            "state_timestamp": timestamp, "source_commit": source_commit},
        "costs": [{"schema": "molgap-cost-event-v1", "cost_event_id": cost, "trajectory_id": tid,
            "action_id": "A001", "run_id": run_id, "attempt_id": attempt, "category": "training", "platform": "kaggle3",
            "hardware": "Tesla_T4x2_shared_allocation", "evidence_ref": authority_ref,
            "measurement": {"device_hours": {"status": "estimated", "value": estimated_device_hours},
                "wall_hours": {"status": "estimated", "value": estimated_wall_hours},
                "cpu_hours": {"status": "measurement_missing", "value": None},
                "queue_hours": {"status": "measurement_missing", "value": None}}}]}


def freeze_comparison_declarations(root, *, base, experiment_id, logical_run_id,
                                   arms, bindings, jobs, entries, plan_format,
                                   authority_ref, source_extras, account, kernel,
                                   title, source_dataset, data_datasets):
    """Write shared Spec/acceptance/workflow envelopes from family-owned facts."""
    from .experiment_spec import ExperimentSpec, TERMINAL_PROTOCOL
    from .experiment_source_inventory import registered_source_files
    from .research_memory.trace import atomic_write, json_bytes
    from .training_reproducibility import sha256_file
    base = safe_comparison_base(base, base="experiments")
    here = Path(root) / base
    if any((here / name).exists() for name in ("experiment_spec.json", "acceptance_plan.json", "workflow_plan.json")):
        raise ValueError("Comparison envelopes already frozen; preserve their bytes")
    spec = ExperimentSpec({"schema_version": "molgap-experiment-spec-v2", "experiment_id": experiment_id,
        "logical_run_id": logical_run_id, "arms": arms, "platform": {"name": "kaggle", "accelerator": "NvidiaTeslaT4",
            "device_count": 2, "cpu_cores": 4, "memory_gib": 29, "atomic_checkpoints": True, "retrievable_chunks": True},
        "prospective": {"arms": bindings}, "evidence": {"policy": {"name": "molgap-v5", "version": "1",
            "sha256": canonical_fingerprint({"contract": "MOLGAP-COMMON-V5-FINAL"})},
            "required_artifacts": ["v5_evidence", "costs", "roles", "trace_manifest", "terminal_artifact"]},
        "terminal_protocol": TERMINAL_PROTOCOL})
    spec.write(here / "experiment_spec.json")
    acceptance = base + "/acceptance_plan.json"
    atomic_write(here / "acceptance_plan.json", json_bytes({"format": plan_format, "spec_identity": spec.identity,
        "authority": {"path": authority_ref, "sha256": sha256_file(Path(root) / authority_ref)}, "arms": entries}))
    sources = registered_source_files(spec, repo_root=root, auto_dependencies=True)
    atomic_write(here / "workflow_plan.json", json_bytes({"format": "molgap-experiment-workflow-v1", "spec_identity": spec.identity,
        "source_files": sources + [*source_extras, acceptance], "arms": jobs, "acceptance_plan": acceptance,
        "kaggle": {"account": account, "kernel": kernel, "title": title, "source_dataset": source_dataset,
            "accelerator": "NvidiaTeslaT4", "datasets": [source_dataset, *data_datasets]}}))
    return {"spec_identity": spec.identity, "kernel": kernel}
