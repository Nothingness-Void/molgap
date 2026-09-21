"""Universal terminal closure runner for this multi-arm experiment.

Executes the native RML terminal closure pipeline across all candidate arms:
  raw arm trace -> canonical trace -> manifest -> readiness -> evidence ->
  costs/roles -> terminal package -> atomic finalize -> validate -> rebuild ->
  check --frozen -> replay admission (REPLAY_READY / REPLAY_EXCLUDED).

No custom per-experiment adapter wiring is needed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from molgap.constants import REPO_ROOT
from molgap.research_memory.native import close_native_bundle


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Execute native RML terminal closure for this experiment."
    )
    default_dir = Path(__file__).parent.relative_to(REPO_ROOT).as_posix()
    parser.add_argument(
        "--receipt",
        default=f"{default_dir}/prelaunch_receipt.json",
        help="Repository-relative pointer to the prelaunch receipt.",
    )
    parser.add_argument(
        "--launch",
        default=f"{default_dir}/launch_binding.json",
        help="Repository-relative pointer to the launch binding.",
    )
    args = parser.parse_args(argv)

    result = close_native_bundle(
        repo_root=REPO_ROOT,
        receipt_ref=args.receipt,
        launch_ref=args.launch,
    )
    print(json.dumps(result, indent=2, sort_keys=True))

    failed = any(
        c.get("pipeline", {}).get("pipeline_status") != "COMPLETE"
        or c.get("closure_complete") is not True
        for c in result.get("candidates", [])
    )
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
