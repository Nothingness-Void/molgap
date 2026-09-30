"""Thin saved-artifact acceptance entry; no training or model inference."""
import argparse
from pathlib import Path

from molgap.gptrans_author_acceptance import accept_prepared_inputs
from molgap.training_reproducibility import atomic_json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--staged-package", type=Path, required=True)
    parser.add_argument("--acceptance-output", type=Path, required=True)
    args = parser.parse_args()
    result = accept_prepared_inputs(args.output_root, args.staged_package)
    atomic_json(args.acceptance_output, result)
    print("CPU input acceptance PASS; separate GPU admission required")
