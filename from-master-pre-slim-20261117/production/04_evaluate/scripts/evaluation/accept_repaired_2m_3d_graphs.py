"""Accept an immutable repaired-2M 3D graph cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from molgap.artifact_acceptance import accept_repaired_3d_graphs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument(
        "--source-csv",
        type=Path,
        default=Path("data/raw/phase8_repaired_2m.csv"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-shards", type=int, default=100)
    parser.add_argument("--pattern", default="graphs_*.pt")
    args = parser.parse_args()
    print(
        json.dumps(
            accept_repaired_3d_graphs(
                args.shard_dir,
                args.source_csv,
                args.output,
                expected_shards=args.expected_shards,
                pattern=args.pattern,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
