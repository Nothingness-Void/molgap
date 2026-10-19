"""Reviewed source inventory shared by the registered screen adapters.

Explicit tracked paths; update once when an adapter gains a dependency.
This is not runtime dependency discovery. Extra addon paths remain explicit.
"""

COMMON_SOURCE_FILES = (
    'src/molgap/family_training_state.py',
    'src/molgap/gptrans_bond_local.py',
    'src/molgap/model_substrate.py',
    'src/molgap/pcqm_conjugated_state.py',
    'src/molgap/pcqm_directed_spectral.py',
    'src/molgap/pcqm_gap_architecture.py',
    'src/molgap/pcqm_gptrans_v4.py',
    'src/molgap/pcqm_hop_path.py',
    'src/molgap/pcqm_local_statistics.py',
    'src/molgap/pcqm_relative_value.py',
    'src/molgap/qm9_data.py',
    'src/molgap/qm9_gape.py',
    'src/molgap/qm9_local_hierarchy.py',
    'src/molgap/qm9_neural_atom.py',
    'src/molgap/shared_model_adapter.py',
    'src/molgap/source_dependencies.py',
    "src/molgap/training/__init__.py",
    "src/molgap/training/core.py",
    "src/molgap/training/state.py",
    "src/molgap/training/session.py",
    "src/molgap/models/__init__.py",
    "src/molgap/models/gptrans.py",
    "src/molgap/models/substrate.py",
    "src/molgap/data/__init__.py",
    "src/molgap/data/packed_graph.py",
    "src/molgap/data/fixed_records.py",
    "src/molgap/compat/__init__.py",
    "src/molgap/compat/legacy_constants.py",
    "src/molgap/common/io.py",
    "src/molgap/common/paths.py",
    "src/molgap/common/spec.py",
    "src/molgap/common/wire.py",
    "src/molgap/experiment_catalog.py",
    "src/molgap/family_training_policy.py",
    "src/molgap/family_model_ops.py",
    "src/molgap/substrate_scale_addon.py",
    "src/molgap/common/__init__.py",
    "src/molgap/common/declarations.py",
    "src/molgap/common/identity.py",
    'platforms/kaggle/run_experiment.py',
    'src/molgap/__init__.py',
    'src/molgap/comparison_readiness.py',
    'src/molgap/constants.py',
    'src/molgap/evidence_pointers.py',
    'src/molgap/experiment_execution.py',
    'src/molgap/execution_comparison.py',
    'src/molgap/experiment_launch_config.py',
    'src/molgap/experiment_allocation.py',
    'src/molgap/experiment_resume.py',
    'src/molgap/experiment_retention.py',
    'src/molgap/experiment_family_artifacts.py',
    'src/molgap/experiment_family_workflow.py',
    'src/molgap/experiment_launch.py',
    'src/molgap/experiment_package.py',
    'src/molgap/experiment_preflight.py',
    'src/molgap/experiment_prospective.py',
    'src/molgap/experiment_source_inventory.py',
    'src/molgap/experiment_spec.py',
    'src/molgap/experiment_staging.py',
    'src/molgap/experiment_staging.py',
    'src/molgap/experiment_terminal.py',
    'src/molgap/experiment_training_worker.py',
    'src/molgap/experiment_workflow.py',
    'src/molgap/experiment_workflow_resume.py',
    'src/molgap/family_training_core.py',
    'src/molgap/family_execution_profile.py',
    'src/molgap/family_training_data.py',
    'src/molgap/family_training_kernel.py',
    'src/molgap/family_training_loaders.py',
    'src/molgap/family_training_outputs.py',
    'src/molgap/family_training_recipe.py',
    'src/molgap/family_training_runner.py',
    'src/molgap/family_training_session.py',
    'src/molgap/gps.py',
    'src/molgap/gptrans.py',
    'src/molgap/gptrans_execution_backend.py',
    'src/molgap/gptrans_execution_hotpath.py',
    'src/molgap/gptrans_execution_tuning.py',
    'src/molgap/foreach_ema.py',
    'src/molgap/gptrans_memory.py',
    'src/molgap/gptrans_screen_workflow.py',
    'src/molgap/gptrans_variants.py',
    'src/molgap/k1_joint_aggregation.py',
    'src/molgap/k1_clean_second.py',
    'src/molgap/k1_screen_training.py',
    'src/molgap/k1_execution_layout.py',
    'src/molgap/k1_execution_backend.py',
    'src/molgap/k1_execution_aggregation.py',
    'src/molgap/k1_execution_qualification.py',
    'src/molgap/packed_graph_dataset.py',
    'src/molgap/pcqm_fixed_records.py',
    'src/molgap/pcqm_wedge.py',
    'src/molgap/kaggle_pair_runtime.py',
    'src/molgap/kaggle_workflow.py',
    'src/molgap/research_memory/__init__.py',
    'src/molgap/research_memory/backtest.py',
    'src/molgap/research_memory/compiler.py',
    'src/molgap/research_memory/cost.py',
    'src/molgap/research_memory/derived.py',
    'src/molgap/research_memory/discovery.py',
    'src/molgap/research_memory/finalize.py',
    'src/molgap/research_memory/paired.py',
    'src/molgap/research_memory/paths.py',
    'src/molgap/research_memory/pipeline.py',
    'src/molgap/research_memory/plan.py',
    'src/molgap/research_memory/policy.py',
    'src/molgap/research_memory/recovery.py',
    'src/molgap/research_memory/references.py',
    'src/molgap/research_memory/replay.py',
    'src/molgap/research_memory/roles.py',
    'src/molgap/research_memory/schemas.py',
    'src/molgap/research_memory/summary.py',
    'src/molgap/research_memory/terminal_wiring.py',
    'src/molgap/research_memory/trace.py',
    'src/molgap/research_memory/validate.py',
    'src/molgap/screen_policy.py',
    'src/molgap/training_reproducibility.py',
    'src/molgap/training_logging.py',
    'src/molgap/v4_bundle.py',
    'src/molgap/v4_runtime.py',
    'src/molgap/v5_common.py',
)

# The retained shared-session implementation needs these family owners. A new
# generic graph family has its own reviewed model/input owner and must not
# inherit a legacy model/trainer merely because both use the lifecycle.
_FAMILY_SOURCE_FILES = frozenset(path for path in COMMON_SOURCE_FILES if path.startswith((
    "src/molgap/family_", "src/molgap/gptrans_", "src/molgap/k1_")) or path in {
    "src/molgap/gps.py", "src/molgap/gptrans.py", "src/molgap/pcqm_gptrans_v4.py",
    "src/molgap/pcqm_gap_architecture.py", "src/molgap/qm9_gape.py",
    "src/molgap/qm9_local_hierarchy.py", "src/molgap/qm9_neural_atom.py",
    "src/molgap/pcqm_conjugated_state.py", "src/molgap/pcqm_directed_spectral.py",
    "src/molgap/pcqm_hop_path.py", "src/molgap/pcqm_local_statistics.py",
    "src/molgap/pcqm_relative_value.py", "src/molgap/qm9_data.py",
})


def registered_source_files(spec, extras=(), *, repo_root=None, auto_dependencies=False):
    """Select reviewed roots, optionally following static import dependencies."""
    from .experiment_execution import training_adapter, validate_training_registry
    from .experiment_package import _allowlist
    from .experiment_spec import ADDONS
    validate_training_registry()
    arms = spec.to_dict()["arms"]
    adapters = [training_adapter(arm) for arm in arms]
    names = set(COMMON_SOURCE_FILES)
    if all(adapter.model_factory is not None for adapter in adapters):
        names.difference_update(_FAMILY_SOURCE_FILES)
    names.update(_allowlist(extras) if extras else ())
    for arm, adapter in zip(arms, adapters):
        names.update(adapter.source_files)
        names.update("src/" + module.replace(".", "/") + ".py"
                     for module in adapter.serialization_modules)
        for addon in arm["addons"]:
            execution = adapter.addon(addon["name"], addon["version"])
            declaration = ADDONS[(addon["name"], addon["version"])]
            names.add("src/" + declaration.source_module.replace(".", "/") + ".py")
            names.update(execution.extra_source_files)
            if execution.policy_hook is not None:
                from .experiment_execution import _hook_source
                names.add(_hook_source(execution.policy_hook))
    selected = _allowlist(sorted(names))
    if auto_dependencies:
        if repo_root is None:
            raise ValueError("Automatic dependency discovery requires the selected repository root")
        from .source_dependencies import discover_source_dependencies
        return list(discover_source_dependencies(repo_root, selected).source_files)
    return selected


def _all_registered_sources():
    from .experiment_execution import TRAINING_ADAPTERS
    from .experiment_spec import ADDONS
    names = set(COMMON_SOURCE_FILES)
    for adapter in TRAINING_ADAPTERS.values():
        names.update(adapter.source_files)
        names.update("src/" + module.replace(".", "/") + ".py"
                     for module in adapter.serialization_modules)
        for addon in adapter.addons:
            names.add("src/" + ADDONS[(addon.name, addon.version)].source_module.replace(".", "/") + ".py")
            names.update(addon.extra_source_files)
            if addon.policy_hook is not None:
                names.add("src/" + addon.policy_hook.split(":")[0].replace(".", "/") + ".py")
    return tuple(sorted(names))


# Retain the full reviewed inventory for compatibility and inventory tests.
SHARED_SOURCE_FILES = _all_registered_sources()


# This is a distribution boundary, not an experiment-specific dependency list.
# Host credentials, controllers and retrieval never belong to worker payloads.
_HOST_ONLY = frozenset({"src/molgap/common/kaggle.py", "src/molgap/common/transport.py",
    "src/molgap/kaggle_credentials.py",
    "src/molgap/local_runtime.py", "src/molgap/experiment_input.py", "src/molgap/kaggle_accelerator_push.py",
    "src/molgap/artifact_inventory.py", "src/molgap/experiment_run.py", "src/molgap/experiment_timings.py",
    "src/molgap/kaggle_output_retrieval.py", "src/molgap/kaggle_submission.py",
    "src/molgap/kaggle_workflow.py", "src/molgap/server_control.py", "src/molgap/server_acceptance.py"})


def runtime_source_files(repo_root, records=()):
    """Fixed tracked Python distribution plus actual frozen record inputs.

    AST discovery remains available for diagnostics, not this preparation path.
    Legacy registered_source_files inventories remain readable and unchanged.
    Closed archive modules require explicit selection by historical callers.
    """
    from .v4_bundle import _tracked_paths
    from .experiment_package import _allowlist
    tracked = _tracked_paths(repo_root)
    # Only closed-experiment reproduction imports this subtree. Keep the
    # exclusion here so explicit frozen package inventories/readers still work.
    names = {p for p in tracked if p.startswith("src/molgap/") and p.endswith(".py")
             and not p.startswith("src/molgap/archive/")
             and p not in _HOST_ONLY}
    if "platforms/kaggle/run_experiment.py" in tracked:
        names.add("platforms/kaggle/run_experiment.py")
    names.update(records)
    if names - tracked:
        raise ValueError("Runtime distribution has untracked record/source inputs")
    return _allowlist(sorted(names))
