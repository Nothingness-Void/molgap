"""Compatibility profile for historical T4 execution comparisons.

New questions use molgap.common.declarations with explicit facts. Frozen
question defaults below remain outside the host-neutral mechanism.
"""
from pathlib import Path

from .screen_policy import canonical_fingerprint
from .common.declarations import (
    ExecutionQuestion, ExecutionProfile, build_execution_proposal,
    comparison_base as _comparison_base, safe_comparison_base, run_version,
)


def comparison_base(spec, *, experiment_id, base):
    return _comparison_base(spec, experiment_id=experiment_id, base=base,
                            arm_ids=("original", "optimized"))


def execution_proposal(*, question, arm, trajectory_id, source_commit, run, version,
                       contract_ref, contract_refs, authority_ref, role_ref, decision_ref,
                       timestamp, owner, track, platform,
                       estimated_device_hours=7, estimated_wall_hours=3.5):
    """Retain the historical T4 profile; common API has no experimental defaults."""
    if platform not in {"kaggle", "kaggle1", "kaggle2", "kaggle3"}:
        raise ValueError("This declaration envelope requires an explicit Kaggle platform")
    profile = ExecutionProfile(
        owner=owner, track=track, platform=platform,
        hardware="Tesla_T4x2_shared_allocation",
        cheapest_falsifier="one matched seed42 T4x2 execution diagnostic with actual operator capture",
        estimated_device_hours=estimated_device_hours, estimated_wall_hours=estimated_wall_hours)
    return build_execution_proposal(
        question=question, arm=arm, trajectory_id=trajectory_id, source_commit=source_commit,
        run=run, version=version, contract_ref=contract_ref, contract_refs=contract_refs,
        authority_ref=authority_ref, role_ref=role_ref, decision_ref=decision_ref,
        timestamp=timestamp, profile=profile)

def freeze_comparison_declarations(root, *, base, experiment_id, logical_run_id,
                                   arms, bindings, jobs, entries, plan_format,
                                   authority_ref, source_extras, account, kernel,
                                   title, source_dataset, data_datasets, same_run_replay=None,
                                   dataset_manifests=None):
    """Write shared Spec/acceptance/workflow envelopes from family-owned facts."""
    from .experiment_spec import ExperimentSpec, TERMINAL_PROTOCOL
    from .experiment_source_inventory import registered_source_files
    from .research_memory.trace import atomic_write, json_bytes
    from .training_reproducibility import sha256_file
    base = safe_comparison_base(base, base="experiments")
    here = Path(root) / base
    if any((here / name).exists() for name in ("experiment_spec.json", "acceptance_plan.json", "workflow_plan.json")):
        raise ValueError("Comparison envelopes already frozen; preserve their bytes")
    if plan_format not in {"molgap-family-acceptance-plan-v1", "molgap-family-same-run-acceptance-plan-v1"}:
        raise ValueError("Use a registered desktop acceptance plan format")
    prospective = {"arms": bindings}
    if same_run_replay is not None:
        prospective["same_run_replay"] = same_run_replay
    # Authority travels in the frozen source, not as an extra acceptance-schema field.
    sha256_file(Path(root) / safe_comparison_base(authority_ref, base="experiments"))
    spec = ExperimentSpec({"schema_version": "molgap-experiment-spec-v2", "experiment_id": experiment_id,
        "logical_run_id": logical_run_id, "arms": arms, "platform": {"name": "kaggle", "accelerator": "NvidiaTeslaT4",
            "device_count": 2, "cpu_cores": 4, "memory_gib": 29, "atomic_checkpoints": True, "retrievable_chunks": True},
        "prospective": prospective, "evidence": {"policy": {"name": "molgap-v5", "version": "1",
            "sha256": canonical_fingerprint({"contract": "MOLGAP-COMMON-V5-FINAL"})},
            "required_artifacts": ["v5_evidence", "costs", "roles", "trace_manifest", "terminal_artifact"]},
        "terminal_protocol": TERMINAL_PROTOCOL})
    atomic_write(here / "experiment_spec.json", spec.to_json().encode("utf-8"))
    acceptance = base + "/acceptance_plan.json"
    atomic_write(here / "acceptance_plan.json", json_bytes({"format": plan_format, "spec_identity": spec.identity,
        "arms": entries}))
    sources = registered_source_files(spec, repo_root=root, auto_dependencies=True)
    workflow = {"format": "molgap-experiment-workflow-v1", "spec_identity": spec.identity,
        "source_files": sorted(set(sources + [authority_ref, *source_extras, acceptance])),
        "arms": jobs, "acceptance_plan": acceptance,
        "kaggle": {"account": account, "kernel": kernel, "title": title, "source_dataset": source_dataset,
            "accelerator": "NvidiaTeslaT4", "datasets": [source_dataset, *data_datasets]}}
    if dataset_manifests is not None:
        workflow["dataset_manifests"] = dict(dataset_manifests)
    atomic_write(here / "workflow_plan.json", json_bytes(workflow))
    return {"spec_identity": spec.identity, "kernel": kernel}
