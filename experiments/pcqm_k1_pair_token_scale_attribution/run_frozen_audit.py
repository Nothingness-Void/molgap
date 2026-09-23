"""Thin CLI for Round 3 frozen PairToken inference interventions."""
from __future__ import annotations

import argparse
from pathlib import Path

from molgap.pcqm_k1_pair_token_scale_audit import run_audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference-payload", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-archive-sha256", required=True)
    args = parser.parse_args()
    result = run_audit(args.cache_root, args.checkpoint, args.reference_payload,
                       args.output_root, audit_source_commit=args.source_commit,
                       source_archive_sha256=args.source_archive_sha256)
    print(result["format"], result["complete"], result["development_rows"])


if __name__ == "__main__":
    main()
