"""Thin CLI for the no-training PCQM specialist/Oracle audit."""

from __future__ import annotations

import argparse
from pathlib import Path

from molgap.pcqm_specialist_oracle import run_specialist_audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--source-csv", type=Path, required=True)
    parser.add_argument("--record-root", type=Path, required=True)
    parser.add_argument("--packaged-payload-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--matrix-output", type=Path, required=True)
    args = parser.parse_args()
    result = run_specialist_audit(
        cache_root=args.cache_root,
        source_csv=args.source_csv,
        record_root=args.record_root,
        packaged_payload_root=args.packaged_payload_root,
        output_root=args.output_root,
        matrix_output=args.matrix_output,
    )
    print(result["decision"]["verdict"])


if __name__ == "__main__":
    main()

