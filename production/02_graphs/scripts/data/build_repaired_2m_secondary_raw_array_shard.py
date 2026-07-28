"""Build one framework-neutral repaired-2M secondary ETKDG array shard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from molgap.etkdg_array import build_secondary_raw_shard


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--workers", type=int, default=14)
    args = parser.parse_args()
    result = build_secondary_raw_shard(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        shard_index=args.shard_index,
        workers=args.workers,
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
