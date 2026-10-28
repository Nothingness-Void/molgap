"""Inspect, diagnose and draft RML inputs for an already retrieved frozen run.

This entry reads local saved artifacts. Closing canonical RML requires explicit
--execute plus reviewed terminal inputs, or explicit outcome/decision inputs.
"""
from pathlib import Path
import sys


_CLOSURE_REPLACES = {"--outcome", "--arm-outcome", "--decision-ref", "--arm-decision-ref", "--role-audit"}


def _expand_closure_input(arguments):
    """Turn --closure-input FILE into the existing per-arm accept-workflow options.

    The file is {"arms": {arm: {"outcome": ..., "decision_ref": ...}}, "role_audit": path?}.
    It only carries human decisions; the acceptance owner still validates them.
    """
    from molgap.experiment_terminal_staging import load_closure_input

    index = next((i for i, a in enumerate(arguments)
                  if a == "--closure-input" or a.startswith("--closure-input=")), None)
    if index is None:
        return arguments
    if arguments[index] == "--closure-input":
        if index + 1 >= len(arguments):
            raise ValueError("--closure-input requires a file")
        path, rest = arguments[index + 1], arguments[:index] + arguments[index + 2:]
    else:
        path, rest = arguments[index].split("=", 1)[1], arguments[:index] + arguments[index + 1:]
    if any(a.split("=", 1)[0] in _CLOSURE_REPLACES for a in rest):
        raise ValueError("--closure-input replaces the individual outcome, decision and role-audit options")
    arms, role_audit = load_closure_input(path)
    expanded = []
    for arm, item in arms.items():
        expanded += ["--arm-outcome", f"{arm}={item['outcome']}",
                     "--arm-decision-ref", f"{arm}={item['decision_ref']}"]
    if role_audit is not None:
        expanded += ["--role-audit", role_audit]
    return [*rest, *expanded]


def main(argv=None):
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "src"))
    from molgap.experiment_cli import main as experiment_main

    arguments = list(sys.argv[1:] if argv is None else argv)
    try:
        arguments = _expand_closure_input(arguments)
    except (OSError, ValueError) as exc:
        import json
        print(json.dumps({"status": "ERROR", "error": str(exc)}))
        return 2
    if "--repo-root" not in arguments and not any(a.startswith("--repo-root=") for a in arguments):
        arguments += ["--repo-root", str(root)]
    reviewed_locations = any(a == "--locations" or a.startswith("--locations=") for a in arguments)
    strict = "--strict" in arguments or any(a == "--strict-inputs" or a.startswith("--strict-inputs=") for a in arguments)
    read_only = reviewed_locations and strict and not any(
        a in {"--execute", "--stage-terminal", "--outcome", "--arm-outcome"}
        or a.startswith(("--outcome=", "--arm-outcome=")) for a in arguments)
    defaults = ["accept-workflow"]
    new_science = any(a in {"--outcome", "--arm-outcome", "--decision-ref", "--arm-decision-ref", "--role-audit", "--stage-terminal"}
        or a.startswith(("--outcome=", "--arm-outcome=", "--decision-ref=", "--arm-decision-ref=", "--role-audit=")) for a in arguments)
    if not read_only and not (strict and not new_science):
        defaults.append("--diagnose")
    # Reviewed packages already carry continuation and qualification bindings.
    # An explicitly requested --stage-terminal still reaches the owning stager.
    if not reviewed_locations:
        defaults.append("--stage-terminal")
    return experiment_main([*defaults, *arguments])


if __name__ == "__main__":
    raise SystemExit(main())
