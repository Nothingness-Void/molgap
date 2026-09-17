"""Thin CLI for the Xi'an reduction determinism audit."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit")
    audit.add_argument("--output", type=Path, required=True)
    audit.add_argument("--strict", action="store_true")
    audit.add_argument("--repetitions", type=int, default=12)

    accept = subparsers.add_parser("accept")
    accept.add_argument("--default", type=Path, required=True)
    accept.add_argument("--strict", type=Path, required=True)
    accept.add_argument("--strict-repeat", type=Path, required=True)
    accept.add_argument("--output", type=Path, required=True)

    model_audit = subparsers.add_parser("model-audit")
    model_audit.add_argument("--model-source", type=Path, required=True)
    model_audit.add_argument("--cache-root", type=Path, required=True)
    model_audit.add_argument("--output", type=Path, required=True)
    model_audit.add_argument("--repetitions", type=int, default=3)
    model_audit.add_argument("--batch-size", type=int, default=128)

    model_accept = subparsers.add_parser("model-accept")
    model_accept.add_argument("--first", type=Path, required=True)
    model_accept.add_argument("--second", type=Path, required=True)
    model_accept.add_argument("--output", type=Path, required=True)

    training_accept = subparsers.add_parser("training-accept")
    training_accept.add_argument("--first-root", type=Path, required=True)
    training_accept.add_argument("--second-root", type=Path, required=True)
    training_accept.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    from molgap.reproducibility_audit import (
        accept_model_replays,
        accept_operator_audits,
        accept_short_training_retests,
        run_model_replay_audit,
        run_operator_audit,
    )

    if args.command == "audit":
        result = run_operator_audit(
            output=args.output,
            strict=args.strict,
            repetitions=args.repetitions,
        )
    elif args.command == "accept":
        result = accept_operator_audits(
            args.default,
            args.strict,
            args.strict_repeat,
            args.output,
        )
    elif args.command == "model-audit":
        result = run_model_replay_audit(
            model_source=args.model_source,
            cache_root=args.cache_root,
            output=args.output,
            repetitions=args.repetitions,
            batch_size=args.batch_size,
        )
    elif args.command == "model-accept":
        result = accept_model_replays(args.first, args.second, args.output)
    else:
        result = accept_short_training_retests(
            args.first_root,
            args.second_root,
            args.output,
        )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
