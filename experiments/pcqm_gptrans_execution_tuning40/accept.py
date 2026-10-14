"""Saved-artifact acceptance only; never model execution."""
import argparse
import json
from pathlib import Path
from molgap.constants import REPO_ROOT
from molgap.gptrans_execution_acceptance import accept

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--receipt-ref", required=True)
    args = parser.parse_args()
    print(json.dumps(accept(REPO_ROOT, args.records, args.package,
                           receipt_ref=args.receipt_ref,
                           base_ref="experiments/pcqm_gptrans_execution_tuning40/gpu"), indent=2))
