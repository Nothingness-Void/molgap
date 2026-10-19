"""Shared declaration plumbing; family owners retain scientific contracts."""
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
                       timestamp, estimated_device_hours=7, estimated_wall_hours=3.5):
    """Preserve the frozen server profile outside the host-neutral mechanism."""
    profile = ExecutionProfile(owner="server", track="C", platform="kaggle3",
        hardware="Tesla_T4x2_shared_allocation",
        cheapest_falsifier="one matched seed42 T4x2 execution diagnostic with actual operator capture",
        estimated_device_hours=estimated_device_hours, estimated_wall_hours=estimated_wall_hours)
    return build_execution_proposal(question=question, arm=arm, trajectory_id=trajectory_id,
        source_commit=source_commit, run=run, version=version, contract_ref=contract_ref,
        contract_refs=contract_refs, authority_ref=authority_ref, role_ref=role_ref,
        decision_ref=decision_ref, timestamp=timestamp, profile=profile)


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


def execution_extension(spec, source_root):
    """Select trusted experiment hooks here; no callback names are read from JSON."""
    from .experiment_execution import ExecutionExtension
    from .family_workflow_trial import is_trial, execution_extension as trial_extension
    if is_trial(spec):
        return trial_extension(spec, source_root)
    arms = spec.to_dict()["arms"]
    from .modern2d_training_release import is_formal_screen
    if is_formal_screen(spec):
        from .modern2d_training_release import execution_extension as formal_extension
        return formal_extension(spec, source_root)
    from .gptrans_infrastructure_comparison import is_comparison, comparison_base
    if is_comparison(spec):
        from .gptrans_execution_diagnostic import read_diagnostic, execution_extension as probe_extension
        record = read_diagnostic(source_root, comparison_base(spec))
        if record is not None:
            return probe_extension(record)
    return ExecutionExtension()
