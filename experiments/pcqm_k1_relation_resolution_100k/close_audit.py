"""Saved-tensor analysis and metadata-only audit terminal preparation."""
import argparse
import json
from molgap.k1_relation_audit_records import analyze, prepare

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("analyze", "prepare"))
    parser.add_argument("--finalized-at")
    args = parser.parse_args()
    if args.operation == "prepare" and not args.finalized_at:
        parser.error("--finalized-at is required for immutable terminal binding")
    result = analyze() if args.operation == "analyze" else prepare(args.finalized_at)
    print(json.dumps(result, indent=2))
