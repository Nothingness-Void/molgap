"""Package the frozen source through the shared deterministic bundle builder."""
import argparse
import json
from pathlib import Path
from molgap.k1_joint_study_records import package


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(package(args.output), default=str))
