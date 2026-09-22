from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-archive-sha256", required=True)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    sys.path.insert(0, str(source_root / "src"))
    from molgap.pcqm_k1_matched60_500k import run

    os.environ["MOLGAP_FIXED_500K_CACHE_ROOT"] = str(args.cache_root.resolve())
    run(
        args.output_root.resolve(),
        source_commit=args.source_commit,
        source_archive_sha256=args.source_archive_sha256,
        resume=args.resume_from.resolve() if args.resume_from else None,
        preflight_only=args.preflight_only,
    )


if __name__ == "__main__":
    main()
