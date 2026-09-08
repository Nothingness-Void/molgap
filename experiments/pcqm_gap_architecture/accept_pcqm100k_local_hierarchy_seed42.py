"""CLI for no-inference acceptance of paired PCQM local-hierarchy training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from molgap.pcqm_local_hierarchy import accept_paired_screen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--label-sha256", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            accept_paired_screen(
                args.root,
                expected_source_commit=args.source_commit,
                expected_label_sha256=args.label_sha256,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
