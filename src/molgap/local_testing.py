"""Run fixed local regression journeys; these do not qualify a cloud submission."""
import argparse
import json
from pathlib import Path
import sys


# Select behaviours, not whole files: unrelated tests must not expand a journey.
_TRAIN_RECOVERY = "tests/test_family_training_runner.py::test_actual_session_output_translation_and_cpu_mechanical_inspection"
_PLATFORM_ROUNDTRIP = "tests/test_reform_daily_platform.py::test_offline_daily_roundtrip_preserves_original_run[observed]"
SCOPES = {
    "runtime": (
        "tests/test_reform_runtime.py::test_two_workspaces_select_their_own_sources_not_inherited_editable",
        "tests/test_fast_sandbox.py::test_baseline_presets_reuse_factory_and_owning_forward",
    ),
    "package": (
        "tests/test_experiment_package.py::test_binding_and_reproducibility",
        "tests/test_local_testing.py::test_extracted_test_package_collects_all_flows_and_executes_its_own_tests",
    ),
    "prepare": (
        "tests/test_scientific_input.py::test_only_one_human_parameter_edit_compiles_and_recovers_actual_single_entry",
    ),
    "train": (_TRAIN_RECOVERY,),
    "submit": (
        "tests/test_kaggle_submission.py::test_real_package_release_post_receipt_reuses_only_exact_content",
    ),
    "accept": (
        "tests/test_run_acceptance_integration.py::test_top_level_execute_restores_frozen_generated_metadata",
        _PLATFORM_ROUNDTRIP,
    ),
    "resume": (
        _TRAIN_RECOVERY,
        "tests/test_shared_resume_transport.py::test_bundle_binds_captured_allocation_sidecar_during_atomic_refresh",
    ),
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=SCOPES, action="append", default=[])
    parser.add_argument("--list", action="store_true", help="Show selections without importing pytest")
    parser.add_argument("--package", type=Path, metavar="OUTPUT",
                        help="Bundle tracked test sources and flow fixtures using the existing source packager")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd(),
                        help="Selected source checkout for --package")
    parser.add_argument("pytest_args", nargs=argparse.REMAINDER, help="Optional pytest flags after --")
    args = parser.parse_args(argv)
    if args.package is not None:
        if args.scope or args.pytest_args:
            parser.error("--package distributes test assets; run --scope separately after extraction")
        from .experiment_source_inventory import testing_source_files, build_testing_source_package
        report = (testing_source_files(args.repo_root) if args.list else
                  build_testing_source_package(args.repo_root, args.package))
        print(json.dumps(report, indent=2))
        return 0
    if not args.scope:
        if args.list:
            print(json.dumps(SCOPES, indent=2))
            return 0
        parser.error("select affected --scope values; no full-suite default")
    nodes = list(dict.fromkeys(node for scope in args.scope for node in SCOPES[scope]))
    if args.list:
        print(json.dumps(nodes, indent=2))
        return 0
    flags = args.pytest_args[1:] if args.pytest_args[:1] == ["--"] else args.pytest_args
    import pytest
    print("Local fixture journeys only; candidate-package execution and native runtime remain unverified.",
          file=sys.stderr)
    return int(pytest.main([*nodes, *flags]))


if __name__ == "__main__":
    raise SystemExit(main())
