"""Local metadata/packaging/acceptance entry; never runs a model."""
import argparse
import json
from pathlib import Path
from molgap.k1_relation_intervention_records import package_and_plan, accept_and_analyze, bind_submission, prepare_terminal

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("package-plan", "accept-analyze", "bind-submission", "prepare-terminal"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--finalized-at")
    args = parser.parse_args()
    if args.operation == "package-plan":
        if args.output is None:
            parser.error("--output required for immutable package")
        result = package_and_plan(args.output)
    elif args.operation == "accept-analyze":
        result = accept_and_analyze()
    elif args.operation == "bind-submission":
        result = bind_submission()
    else:
        if not args.finalized_at:
            parser.error("--finalized-at required for immutable terminal identity")
        result = prepare_terminal(args.finalized_at)
    print(json.dumps(result, indent=2))
