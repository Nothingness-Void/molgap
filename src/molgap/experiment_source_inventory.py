"""Reviewed source inventory shared by the registered screen adapters.

Explicit tracked paths; update once when an adapter gains a dependency.
This is not runtime dependency discovery. Extra addon paths remain explicit.
"""

COMMON_SOURCE_FILES = (
    "src/molgap/modern2d_execution.py",
    "src/molgap/modern2d_execution_context.py",
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
    "src/molgap/data/indexed_inputs.py",
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
    'src/molgap/family_workflow_trial.py',
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
    'src/molgap/native_proof_storage.py',
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
    'src/molgap/gptrans_execution_diagnostic.py',
    'src/molgap/gptrans_execution_analysis.py',
    'src/molgap/execution_analysis.py',
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
    'src/molgap/research_memory/portability.py',
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
    elif all(adapter.module != "molgap.family_training_runner" for adapter in adapters):
        # Keep shared execution/session/profiling primitives: the legacy
        # GPTrans update path also uses update_scope. Remove only unselected
        # model mechanisms/trainers, restored by the declared adapter below.
        names.difference_update({"src/molgap/k1_screen_training.py",
            "src/molgap/k1_joint_aggregation.py", "src/molgap/k1_clean_second.py",
            "src/molgap/gptrans_memory.py"})
    names.update(_allowlist(extras) if extras else ())
    for arm, adapter in zip(arms, adapters):
        names.update(adapter.source_files)
        for addon in arm["addons"]:
            execution = adapter.addon(addon["name"], addon["version"])
            declaration = ADDONS[(addon["name"], addon["version"])]
            names.add("src/" + declaration.source_module.replace(".", "/") + ".py")
            names.update(execution.extra_source_files)
            if execution.policy_hook is not None:
                names.add("src/" + execution.policy_hook.split(":")[0].replace(".", "/") + ".py")
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
            if addon.policy_hook is not None:
                names.add("src/" + addon.policy_hook.split(":")[0].replace(".", "/") + ".py")
    return tuple(sorted(names))


# Retain the full reviewed inventory for compatibility and inventory tests.
SHARED_SOURCE_FILES = _all_registered_sources()


# This is a distribution boundary, not an experiment-specific dependency list.
# Host credentials, controllers and retrieval never belong to worker payloads.
_HOST_ONLY = frozenset({"src/molgap/common/kaggle.py", "src/molgap/common/transport.py",
    "src/molgap/experiment_attribution.py", "src/molgap/experiment_run_acceptance.py",
    "src/molgap/experiment_terminal_staging.py", "src/molgap/experiment_presets.py",
    "src/molgap/modern2d_training_prepare.py", "src/molgap/modern2d_qualification_prepare.py",
    "src/molgap/modern2d_research_prepare.py",
    "src/molgap/kaggle_credentials.py",
    "src/molgap/local_runtime.py", "src/molgap/local_testing.py", "src/molgap/local_storage.py",
    "src/molgap/experiment_input.py", "src/molgap/kaggle_accelerator_push.py",
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


# These are compact inputs read by the existing local flow tests. Tests create
# molecular data, model states and Git repositories under temporary directories.
# Retain all test Python so shared fixtures do not need a second import catalog.
_TEST_SUPPORT_FILES = frozenset({
    "experiments/.gitattributes",
    "research_memory/policies/.gitattributes",
    "pyproject.toml",
    "platforms/local/run.py",
    "platforms/local/accept_run.py",
    "platforms/kaggle/run_experiment.py",
    "platforms/kaggle/submit_workflow.py",
    "platforms/kaggle/retrieve_family_outputs.py",
    "experiments/pcqm_gptrans_t_100k_v4/training_contract.json",
    "research_memory/policies/k1-relation-resolution-bounded-research.v1.json",
    "docs/operations/EXPERIMENT_ADDON_GUIDE.md",
    "docs/operations/EXPERIMENT_CLI.md",
})


def testing_source_files(repo_root):
    """Explicit host test distribution; never added to worker runtime-v1.

    All tracked active package/test Python is retained, plus the compact inputs
    needed by local_testing's scopes. Other historical tests may still require
    their own repository records/assets. No weights, datasets or host accounts
    are discovered or copied. The dependency interpreter supplies pytest and
    the project's normal dependencies.
    """
    from .experiment_package import _allowlist
    from .v4_bundle import _tracked_paths
    tracked = _tracked_paths(repo_root)
    missing = _TEST_SUPPORT_FILES - tracked
    if missing:
        raise ValueError("Test distribution is missing tracked support files: " + str(sorted(missing)))
    names = {p for p in tracked if p.endswith(".py") and (
        p.startswith("tests/") or (p.startswith("src/molgap/") and
                                  not p.startswith("src/molgap/archive/")))}
    return _allowlist(sorted(names | _TEST_SUPPORT_FILES))


def build_testing_source_package(repo_root, output_dir):
    """Reuse the committed V4 source bundle without inventing a scientific Spec."""
    from pathlib import Path
    from .common.io import _safe_local
    from .experiment_package import _git, _source
    from .v4_bundle import build_v4_source_bundle
    root, output = Path(repo_root).resolve(), Path(output_dir).absolute()
    _safe_local(root)
    _safe_local(output)
    if output == root or root.is_relative_to(output):
        raise ValueError("Test package output must not contain the selected checkout")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Test package output must be absent or empty")
    names = testing_source_files(root)
    for name in names:
        _source(root, name)
    commit = _git(root, "rev-parse", "HEAD").decode().strip()
    return build_v4_source_bundle(repo_root=root, relative_paths=names,
                                 output_dir=output, source_commit=commit)
