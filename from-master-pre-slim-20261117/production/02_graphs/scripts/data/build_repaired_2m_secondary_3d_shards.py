"""Build, resume, and validate repaired-2M secondary ETKDG graph shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from molgap.repaired_2m_3d_colab import (
    build_secondary_graph_shards,
    validate_graph_shards,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repaired-csv", type=Path, required=True)
    parser.add_argument("--primary-graph-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=28)
    parser.add_argument("--seed", type=int, default=314_159)
    parser.add_argument("--skip-repaired-sha256", action="store_true")
    args = parser.parse_args()

    build = build_secondary_graph_shards(
        repaired_csv=args.repaired_csv,
        primary_graph_dir=args.primary_graph_dir,
        output_dir=args.output_dir,
        workers=args.workers,
        seed=args.seed,
        verify_repaired_sha256=not args.skip_repaired_sha256,
    )
    validation = validate_graph_shards(args.output_dir)
    print(
        json.dumps(
            {"build": build, "validation": validation},
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
