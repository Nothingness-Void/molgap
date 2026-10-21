"""Inspect, diagnose and draft RML inputs for an already retrieved frozen run.

This entry performs local saved-artifact analysis only. Closing canonical RML
requires an explicit outcome, decision reference and --execute.
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
    return experiment_main(["accept-workflow", "--diagnose", "--stage-terminal", *arguments])


if __name__ == "__main__":
    raise SystemExit(main())
