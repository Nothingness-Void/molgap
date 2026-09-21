"""CLI Wrapper for MolGap Preflight Replay-Ready Verification."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from molgap.preflight import PreflightError, run_triple_check


def main() -> None:
    parser = argparse.ArgumentParser(description="MolGap Preflight Replay-Ready Verification")
    parser.add_argument("--experiment", type=Path, required=True, help="Path to experiment dir")
    parser.add_argument("--package", type=Path, default=None, help="Path to staged kernel package")
    args = parser.parse_args()

    try:
        run_triple_check(args.experiment.resolve(), args.package.resolve() if args.package else None)
    except PreflightError as e:
        print(f"\n[FAIL CLOSED] Pre-flight verification failed:\n  ERROR: {e}\n", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
