"""Reviewed source inventory shared by the registered screen adapters.

Explicit tracked paths; update once when an adapter gains a dependency.
This is not runtime dependency discovery. Extra addon paths remain explicit.
"""

COMMON_SOURCE_FILES = (
    'src/molgap/common/__init__.py',
    'src/molgap/common/declarations.py',
    'src/molgap/common/identity.py',
    'src/molgap/common/io.py',
    'src/molgap/common/paths.py',
    'src/molgap/common/spec.py',
    'src/molgap/common/wire.py',
    'platforms/kaggle/run_experiment.py',
    'src/molgap/__init__.py',
    'src/molgap/comparison_readiness.py',
    'src/molgap/constants.py',
    'src/molgap/evidence_pointers.py',
    'src/molgap/experiment_execution.py',
    'src/molgap/execution_comparison.py',
    'src/molgap/experiment_launch_config.py',
    'src/molgap/experiment_inspection.py',
    'src/molgap/experiment_resume.py',
    'src/molgap/experiment_allocation.py',
    'src/molgap/experiment_retention.py',
    'src/molgap/experiment_family_artifacts.py',
    'src/molgap/experiment_family_workflow.py',
    'src/molgap/experiment_launch.py',
    'src/molgap/experiment_package.py',
    'src/molgap/experiment_preflight.py',
    'src/molgap/experiment_prospective.py',
    'src/molgap/experiment_source_inventory.py',
    'src/molgap/experiment_spec.py',
    'src/molgap/experiment_catalog.py',
    'src/molgap/experiment_staging.py',
    'src/molgap/experiment_terminal.py',
    'src/molgap/experiment_training_worker.py',
    'src/molgap/experiment_workflow.py',
    'src/molgap/experiment_workflow_resume.py',
    'src/molgap/family_training_core.py',
    'src/molgap/family_execution_profile.py',
    'src/molgap/family_pair_release.py',
    'src/molgap/family_pair_acceptance.py',
    'src/molgap/gptrans_campaign_release.py',
    'src/molgap/family_cycle_audit.py',
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
    'src/molgap/gptrans_legacy_execution_profile.py',
    'src/molgap/gptrans_infrastructure_comparison.py',
    'src/molgap/gptrans_variants.py',
    'src/molgap/k1_joint_aggregation.py',
    'src/molgap/k1_clean_second.py',
    'src/molgap/k1_screen_training.py',
    'src/molgap/k1_execution_layout.py',
    'src/molgap/k1_execution_backend.py',
    'src/molgap/k1_execution_aggregation.py',
    'src/molgap/k1_execution_qualification.py',
    'src/molgap/k1_legacy_execution_profile.py',
    'src/molgap/k1_infrastructure_comparison.py',
    'src/molgap/packed_graph_dataset.py',
    'src/molgap/pcqm_fixed_records.py',
    'src/molgap/pcqm_wedge.py',
    'src/molgap/kaggle_pair_runtime.py',
    'src/molgap/kaggle_python_environment.py',
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
    'src/molgap/research_memory/run_binding.py',
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
    "src/molgap/family_", "src/molgap/gptrans_", "src/molgap/k1_")) or path == "src/molgap/gps.py")


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
        for addon in arm["addons"]:
            execution = adapter.addon(addon["name"], addon["version"])
            declaration = ADDONS[(addon["name"], addon["version"])]
            names.add("src/" + declaration.source_module.replace(".", "/") + ".py")
            names.update(execution.extra_source_files)
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
        for addon in adapter.addons:
            names.add("src/" + ADDONS[(addon.name, addon.version)].source_module.replace(".", "/") + ".py")
            names.update(addon.extra_source_files)
    return tuple(sorted(names))


# Retain the full reviewed inventory for compatibility and inventory tests.
SHARED_SOURCE_FILES = _all_registered_sources()
