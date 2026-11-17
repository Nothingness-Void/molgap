"""Build the resumable RWSE graph cache for the architecture screen."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from molgap.structural_encoding import build_rwse_graph_cache


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--progress-dir", type=Path, required=True)
    parser.add_argument("--walk-length", type=int, default=16)
    parser.add_argument("--shard-size", type=int, default=10_000)
    parser.add_argument("--max-graphs", type=int)
    args = parser.parse_args()
    report = build_rwse_graph_cache(
        args.input,
        args.output,
        args.progress_dir,
        walk_length=args.walk_length,
        shard_size=args.shard_size,
        max_graphs=args.max_graphs,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
