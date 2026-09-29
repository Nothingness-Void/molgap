"""Saved-output CLI: does not retrieve data or execute a model."""
import argparse
from pathlib import Path
from molgap.k1_representation_records import write_reports

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--plan", type=Path, default=Path("experiments/pcqm_k1_explainability_audit/representation"))
    parser.add_argument("--scheduler", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_reports(args.raw, args.plan, args.scheduler, args.output)
