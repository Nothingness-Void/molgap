"""Build or resume the repaired-2M primary ETKDG graph shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from molgap.repaired_2m_3d_colab import build_graph_shards


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repaired-csv", type=Path, required=True)
    parser.add_argument("--original-csv", type=Path, required=True)
    parser.add_argument("--original-graphs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-size", type=int, default=20_000)
    parser.add_argument("--workers", type=int, default=28)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-repaired-sha256", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            build_graph_shards(
                repaired_csv=args.repaired_csv,
                original_csv=args.original_csv,
                original_graph_cache=args.original_graphs,
                output_dir=args.output_dir,
                shard_size=args.shard_size,
                workers=args.workers,
                seed=args.seed,
                verify_repaired_sha256=not args.skip_repaired_sha256,
            ),
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
