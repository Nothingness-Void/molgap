"""Saved-artifact CLI; never executes a model."""
import argparse
from pathlib import Path
from molgap.constants import REPO_ROOT
from molgap.gptrans_execution_acceptance import accept
from molgap.training_reproducibility import atomic_json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt-ref", required=True)
    args = parser.parse_args()
    atomic_json(args.output, accept(REPO_ROOT, args.records, args.package, receipt_ref=args.receipt_ref))
