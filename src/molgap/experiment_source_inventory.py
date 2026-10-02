"""Reviewed source inventory shared by the registered screen adapters.

Explicit tracked paths; update once when an adapter gains a dependency.
This is not runtime dependency discovery. Extra addon paths remain explicit.
"""

COMMON_SOURCE_FILES = (
    'platforms/kaggle/run_experiment.py',
    'src/molgap/__init__.py',
    'src/molgap/comparison_readiness.py',
    'src/molgap/constants.py',
    'src/molgap/evidence_pointers.py',
    'src/molgap/experiment_execution.py',
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
    'src/molgap/experiment_staging.py',
    'src/molgap/experiment_terminal.py',
    'src/molgap/experiment_training_worker.py',
    'src/molgap/experiment_workflow.py',
    'src/molgap/experiment_workflow_resume.py',
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
    'src/molgap/research_memory/run_binding.py',
    'src/molgap/research_memory/schemas.py',
    'src/molgap/research_memory/summary.py',
    'src/molgap/research_memory/terminal_wiring.py',
    'src/molgap/research_memory/trace.py',
    'src/molgap/research_memory/validate.py',
    'src/molgap/screen_policy.py',
    'src/molgap/training_reproducibility.py',
    'src/molgap/v4_bundle.py',
    'src/molgap/v4_runtime.py',
    'src/molgap/v5_common.py',
)


def registered_source_files(spec, extras=()):
    """Select reviewed common, family and addon paths; never scan a checkout."""
    from .experiment_execution import training_adapter, validate_training_registry
    from .experiment_package import _allowlist
    from .experiment_spec import ADDONS
    validate_training_registry()
    names = set(COMMON_SOURCE_FILES)
    names.update(_allowlist(extras) if extras else ())
    for arm in spec.to_dict()["arms"]:
        adapter = training_adapter(arm)
        names.update(adapter.source_files)
        for addon in arm["addons"]:
            execution = adapter.addon(addon["name"], addon["version"])
            declaration = ADDONS[(addon["name"], addon["version"])]
            names.add("src/" + declaration.source_module.replace(".", "/") + ".py")
            names.update(execution.extra_source_files)
    return _allowlist(sorted(names))


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
