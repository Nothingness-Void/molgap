"""Thin Kunshan entry point for the GPTrans shortest-path profile."""
from __future__ import annotations

import argparse
import json
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
    parser.add_argument("--allocation-seconds", type=float, required=True)
    parser.add_argument("--slurm-job-id", required=True)
    args = parser.parse_args()
    source_root = args.source_root.resolve()
    cache_root = args.cache_root.resolve()
    if not (source_root / "src/molgap/pcqm_gptrans_shortest_path_profile.py").is_file():
        raise FileNotFoundError("Immutable profiling source is incomplete")
    if not (cache_root / "manifest.json").is_file():
        raise FileNotFoundError("Accepted fixed 500K cache is incomplete")
    os.environ["MOLGAP_FIXED_500K_CACHE_ROOT"] = str(cache_root)
    sys.path.insert(0, str(source_root / "src"))
    from molgap.pcqm_gptrans_shortest_path_profile import run

    result = run(
        args.output_root,
        source_commit=args.source_commit,
        source_archive_sha256=args.source_archive_sha256,
        allocation_seconds=args.allocation_seconds,
        slurm_job_id=args.slurm_job_id,
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
