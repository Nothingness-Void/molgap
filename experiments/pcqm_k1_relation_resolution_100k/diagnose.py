"""Local metadata/packaging/acceptance entry; never runs a model."""
import argparse
import json
from pathlib import Path
from molgap.k1_relation_intervention_records import package_and_plan, accept_and_analyze, bind_submission

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("package-plan", "accept-analyze", "bind-submission"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.operation == "package-plan":
        if args.output is None:
            parser.error("--output required for immutable package")
        result = package_and_plan(args.output)
    elif args.operation == "accept-analyze":
        result = accept_and_analyze()
    else:
        result = bind_submission()
    print(json.dumps(result, indent=2))
