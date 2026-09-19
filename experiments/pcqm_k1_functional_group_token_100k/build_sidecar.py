"""Thin CPU-only CLI for the functional-group sidecar."""
from __future__ import annotations

import argparse
from pathlib import Path

from molgap.pcqm_functional_group_sidecar import (
    accept_functional_group_sidecar,
    build_functional_group_sidecar,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    build_functional_group_sidecar(
        args.fixed_root.resolve(),
        args.output_root.resolve(),
        source_commit=args.source_commit,
    )
    accept_functional_group_sidecar(
        args.output_root.resolve(), expected_source_commit=args.source_commit
    )


if __name__ == "__main__":
    main()
