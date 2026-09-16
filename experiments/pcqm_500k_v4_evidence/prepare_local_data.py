"""Download and accept the exact fixed-500K V4 cache for local execution."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from molgap.pcqm_v4_local import accept_fixed_500k_cache, download_fixed_500k_cache


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--credentials", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    credentials = json.loads(args.credentials.read_text(encoding="utf-8"))
    download_fixed_500k_cache(
        args.output_root,
        username=credentials["username"],
        key=credentials["key"],
        workers=args.workers,
    )
    result = accept_fixed_500k_cache(args.output_root, deep=True)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
