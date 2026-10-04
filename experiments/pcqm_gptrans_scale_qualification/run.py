"""Invoke the shared qualification owner in a single-GPU fresh environment."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from molgap.experiment_package import verify_experiment_source_package
from molgap.gptrans_scale_profile import profile

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-package", type=Path, required=True)
    args = parser.parse_args()
    contract = json.loads(Path(__file__).with_name("contract.json").read_text())
    print(json.dumps(profile(args.inputs, args.output, contract,
        source_identity=verify_experiment_source_package(args.source_package))))
