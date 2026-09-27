"""Local receipt verification and existing-Luna binding; no submission."""
import argparse
import json
from molgap.k1_joint_study_ops import bind

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-package", required=True)
    parser.add_argument("--records", required=True)
    args = parser.parse_args()
    print(json.dumps(bind(args.source_package, args.records), indent=2))
