"""Accept an independently seeded secondary repaired-2M 3D graph cache.

The primary acceptance CLI cannot be reused: secondary shards are keyed to the
primary view's per-shard graph counts, not to source-index spans, so their
sidecars carry no `start`/`stop`. This entrypoint additionally proves the two
views are the same molecule set and that the coordinates really differ.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from molgap.artifact_acceptance import accept_repaired_3d_secondary_graphs
from molgap.constants import RAW_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument(
        "--source-csv",
        type=Path,
        default=RAW_DIR / "phase8_repaired_2m.csv",
    )
    parser.add_argument("--primary-manifest", type=Path, required=True)
    parser.add_argument(
        "--primary-shard-dir",
        type=Path,
        default=None,
        help="Defaults to <primary manifest parent>/graph_shards.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-shards", type=int, default=100)
    parser.add_argument("--expected-seed", type=int, default=314159)
    parser.add_argument("--pattern", default="graphs_*.pt")
    parser.add_argument("--min-distinct-fraction", type=float, default=0.999)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        json.dumps(
            accept_repaired_3d_secondary_graphs(
                args.shard_dir,
                args.source_csv,
                args.primary_manifest,
                args.output,
                expected_shards=args.expected_shards,
                expected_seed=args.expected_seed,
                pattern=args.pattern,
                min_distinct_fraction=args.min_distinct_fraction,
                primary_shard_dir=args.primary_shard_dir,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
