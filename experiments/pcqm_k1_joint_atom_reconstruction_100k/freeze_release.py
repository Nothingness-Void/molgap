"""Freeze real-reference-bound plans only; no remote execution."""
import json
from molgap.k1_joint_study_records import freeze


if __name__ == "__main__":
    print(json.dumps(freeze(), default=str))
