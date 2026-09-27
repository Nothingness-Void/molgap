"""Local metadata/packaging/acceptance entry; never runs a model."""
import argparse
import json
from pathlib import Path
from molgap.k1_relation_intervention_records import package_and_plan, accept_and_analyze

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("package-plan", "accept-analyze"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.operation == "package-plan":
        if args.output is None:
            parser.error("--output required for immutable package")
        result = package_and_plan(args.output)
    else:
        result = accept_and_analyze()
    print(json.dumps(result, indent=2))
