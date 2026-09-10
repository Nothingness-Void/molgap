from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

from molgap.pcqm_gptrans_scale import (
    GPTransScaleConfig,
    atomic_json,
    run_preflight,
    run_worker,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("preflight", "train"))
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit")
    args = parser.parse_args()
    config = GPTransScaleConfig()
    try:
        if args.command == "preflight":
            result = run_preflight(args.cache_root, args.output, config)
        else:
            if args.source_commit is None:
                parser.error("train requires --source-commit")
            result = run_worker(
                args.cache_root,
                args.output,
                config,
                source_commit=args.source_commit,
            )
    except Exception as error:
        atomic_json(
            args.output / "failure.json",
            {
                "complete": False,
                "command": args.command,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        raise
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
