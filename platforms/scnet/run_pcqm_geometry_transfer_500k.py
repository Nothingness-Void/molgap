from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

from molgap.pcqm_geometry_transfer import MODEL_IDS
from molgap.pcqm_geometry_transfer_runner import (
    GeometryTransferConfig,
    atomic_json,
    run_fusion,
    run_preflight,
    run_training,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("preflight", "train", "fusion"))
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", choices=MODEL_IDS)
    parser.add_argument("--source-commit")
    args = parser.parse_args()
    try:
        if args.command == "fusion":
            result = run_fusion(args.output)
        else:
            if args.cache_root is None or args.model_id is None:
                parser.error("preflight/train require --cache-root and --model-id")
            config = GeometryTransferConfig(model_id=args.model_id)
            if args.command == "preflight":
                result = run_preflight(args.cache_root, args.output, config)
            else:
                if args.source_commit is None:
                    parser.error("train requires --source-commit")
                result = run_training(
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
                "model_id": args.model_id,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        raise
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
