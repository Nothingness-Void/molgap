"""Run the existing tests for selected infrastructure flows in one pytest session."""
import argparse
import json
from pathlib import Path


# These are entry points into existing tests, not another acceptance policy.
SCOPES = {
    "runtime": ("test_reform_runtime", "test_local_testing", "test_local_storage", "test_machine_paths"),
    "package": ("test_experiment_package", "test_reform_package", "test_reform_distribution",
                "test_infra_package_audit"),
    "prepare": ("test_reform_input", "test_scientific_input", "test_reform_workflow",
                "test_reform_catalog", "test_reform_review", "test_experiment_run",
                "test_infra_failure_publication", "test_modern2d_role_discovery",
                "test_modern2d_training_prepare", "test_modern2d_qualification_prepare",
                "test_scientific_input_ergonomics", "test_scientific_input_templates",
                "test_scientific_pair_input"),
    "submit": ("test_reform_workflow", "test_kaggle_submission",
               "test_reform_daily_platform", "test_kaggle_bootstrap_server", "test_infra_launch_audit"),
    "accept": ("test_reform_retrieval", "test_experiment_family_workflow",
               "test_kaggle_family_retrieval", "test_experiment_run", "test_reform_daily_platform",
               "test_run_acceptance_resolution", "test_run_acceptance_cli", "test_run_acceptance_integration",
               "test_experiment_terminal_staging", "test_experiment_attribution", "test_run_attribution",
               "test_experiment_cli", "test_modern2d_training_records",
               "test_modern2d_qualification_records", "test_modern2d_configured_outputs"),
    "resume": ("test_experiment_workflow_resume", "test_family_recovery_guards",
               "test_experiment_run", "test_reform_daily_platform", "test_experiment_resume",
               "test_infra_recovery_audit"),
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
    files = list(dict.fromkeys(
        f"tests/{name}.py" for scope in args.scope for name in SCOPES[scope]))
    if args.list:
        print(json.dumps(files, indent=2))
        return 0
    flags = args.pytest_args[1:] if args.pytest_args[:1] == ["--"] else args.pytest_args
    import pytest
    return int(pytest.main([*files, *flags]))


if __name__ == "__main__":
    raise SystemExit(main())
