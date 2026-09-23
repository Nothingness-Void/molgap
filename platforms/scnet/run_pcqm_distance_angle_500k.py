from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

from molgap.pcqm_distance_angle_scale import (
    ARMS,
    DistanceAngleScaleConfig,
    accept_pair,
    accept_subset,
    build_subset_manifest,
    run_preflight,
    run_worker,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", choices=("manifest", "accept", "preflight", "train", "accept-results")
    )
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--parent-acceptance", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--role", choices=ARMS)
    parser.add_argument("--source-commit")
    args = parser.parse_args()
    config = DistanceAngleScaleConfig()
    try:
        if args.command == "manifest":
            if args.parent_acceptance is None or args.output is None:
                parser.error("manifest requires --parent-acceptance and --output")
            result = build_subset_manifest(args.parent_acceptance, args.output)
        elif args.command == "accept":
            if args.cache_root is None:
                parser.error("accept requires --cache-root")
            result = accept_subset(args.cache_root)
        elif args.command == "preflight":
            if args.cache_root is None or args.output is None:
                parser.error("preflight requires --cache-root and --output")
            result = run_preflight(args.cache_root, args.output, config)
        elif args.command == "train":
            if None in (args.cache_root, args.output, args.role, args.source_commit):
                parser.error("train requires cache, output, role, and source commit")
            result = run_worker(
                args.role,
                args.cache_root,
                args.output,
                config,
                source_commit=args.source_commit,
            )
        else:
            if args.output is None:
                parser.error("accept-results requires --output")
            result = accept_pair(args.output)
    except Exception as error:
        if args.output is not None:
            from molgap.pcqm_distance_angle_scale import atomic_json

            atomic_json(
                args.output / "failure.json",
                {
                    "complete": False,
                    "command": args.command,
                    "role": args.role,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                },
            )
        raise
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
