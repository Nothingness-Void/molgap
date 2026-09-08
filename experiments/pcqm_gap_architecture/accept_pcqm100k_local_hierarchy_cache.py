"""CLI for no-model acceptance of the PCQM local-hierarchy sidecar."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from molgap.pcqm_local_hierarchy import accept_local_label_cache


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            accept_local_label_cache(
                args.root, expected_source_commit=args.source_commit
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
