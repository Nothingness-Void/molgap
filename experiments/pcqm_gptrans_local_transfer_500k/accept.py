"""Thin saved-output acceptance and shared RML closure; no local model execution."""
import argparse
import json
from pathlib import Path
from molgap.constants import REPO_ROOT
from molgap.gptrans_scale_local_acceptance import accept_local_outputs, close_local_outputs
from molgap.training_reproducibility import atomic_json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    args = parser.parse_args()
    output = REPO_ROOT / "experiments/pcqm_gptrans_local_transfer_500k/gpu/results/acceptance.json"
    atomic_json(output, accept_local_outputs(REPO_ROOT, args.records, args.package))
    print(json.dumps(close_local_outputs(REPO_ROOT, args.records, output)))
