"""Mechanical no-inference training acceptance, independently per notebook."""
import argparse
from pathlib import Path
from molgap.k1_relation_study_records import accept_training, save

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--slot", choices=("dual", "rrwp"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    save(args.output, accept_training(args.reference_root, args.candidate_root,
        args.source_commit, args.archive_sha256, args.slot))
