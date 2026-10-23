"""Inspect, diagnose and draft RML inputs for an already retrieved frozen run.

This entry reads local saved artifacts. Closing canonical RML requires explicit
--execute plus reviewed terminal inputs, or explicit outcome/decision inputs.
"""
from pathlib import Path
import sys


def main(argv=None):
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "src"))
    from molgap.experiment_cli import main as experiment_main

    arguments = list(sys.argv[1:] if argv is None else argv)
    if "--repo-root" not in arguments and not any(a.startswith("--repo-root=") for a in arguments):
        arguments += ["--repo-root", str(root)]
    reviewed_locations = any(a == "--locations" or a.startswith("--locations=") for a in arguments)
    strict = "--strict" in arguments or any(a == "--strict-inputs" or a.startswith("--strict-inputs=") for a in arguments)
    read_only = reviewed_locations and strict and not any(
        a in {"--execute", "--stage-terminal", "--outcome"} or a.startswith("--outcome=") for a in arguments)
    defaults = ["accept-workflow"]
    new_science = any(a in {"--outcome", "--decision-ref", "--role-audit"}
        or a.startswith(("--outcome=", "--decision-ref=", "--role-audit=")) for a in arguments)
    if not read_only and not (strict and not new_science):
        defaults.append("--diagnose")
    # Reviewed packages already carry continuation and qualification bindings.
    # An explicitly requested --stage-terminal still reaches the owning stager.
    if not reviewed_locations:
        defaults.append("--stage-terminal")
    return experiment_main([*defaults, *arguments])


if __name__ == "__main__":
    raise SystemExit(main())
