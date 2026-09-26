"""Package accepted frozen models and already accepted K1 predictions."""
import argparse
import json
from pathlib import Path
from molgap.k1_relation_audit import package

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = package(args.output)
    print(json.dumps({"run_id": spec["run_id"], "input_archive_sha256": spec["input_archive_sha256"]}))
