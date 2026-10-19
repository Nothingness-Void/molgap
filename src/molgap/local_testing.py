"""Run the existing tests for selected infrastructure flows in one pytest session."""
import argparse
import json


# These are entry points into existing tests, not another acceptance policy.
SCOPES = {
    "runtime": ("test_reform_runtime", "test_local_testing"),
    "package": ("test_experiment_package", "test_reform_package"),
    "prepare": ("test_reform_input", "test_scientific_input", "test_reform_workflow"),
    "submit": ("test_reform_workflow", "test_kaggle_submission", "test_kaggle_bootstrap"),
    "accept": ("test_reform_retrieval", "test_experiment_family_workflow"),
    "resume": ("test_experiment_workflow_resume", "test_family_recovery_guards"),
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=SCOPES, action="append", default=[])
    parser.add_argument("--list", action="store_true", help="Show selections without importing pytest")
    parser.add_argument("pytest_args", nargs=argparse.REMAINDER, help="Optional pytest flags after --")
    args = parser.parse_args(argv)
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
