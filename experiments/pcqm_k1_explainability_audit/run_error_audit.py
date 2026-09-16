"""Thin CLI for the frozen K1 V4 error-attribution audit."""

from __future__ import annotations

import argparse
from pathlib import Path

from molgap.pcqm_k1_explainability import run_error_audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--payload-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = run_error_audit(args.cache_root, args.payload_root, args.output_root)
    print(result["format"], result["complete"], result["development_rows"])


if __name__ == "__main__":
    main()

