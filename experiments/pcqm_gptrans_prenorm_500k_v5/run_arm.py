"""Thin SCNet entry point for one frozen GPTrans 500K V5 arm."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("reference", "pair_prenorm"), required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-archive-sha256", required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--reuse-runtime-certificate", action="store_true")
    parser.add_argument("--execution-source-commit")
    parser.add_argument("--execution-source-archive-sha256")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    cache_root = args.cache_root.resolve()
    if not (source_root / "src/molgap/pcqm_gptrans_prenorm_500k.py").is_file():
        raise FileNotFoundError("Immutable GPTrans Pair PreNorm source is incomplete")
    if not (cache_root / "manifest.json").is_file():
        raise FileNotFoundError("Accepted fixed 500K cache is incomplete")
    os.environ["MOLGAP_FIXED_500K_CACHE_ROOT"] = str(cache_root)
    os.environ["MOLGAP_PLATFORM_ID"] = "scnet-kunshan"
    sys.path.insert(0, str(source_root / "src"))

    from molgap.pcqm_gptrans_prenorm_500k import run

    run(
        args.output_root,
        mode=args.mode,
        source_commit=args.source_commit,
        source_archive_sha256=args.source_archive_sha256,
        resume=args.resume,
        preflight_only=args.preflight_only,
        reuse_runtime_certificate=args.reuse_runtime_certificate,
        execution_source_commit=args.execution_source_commit,
        execution_source_archive_sha256=args.execution_source_archive_sha256,
    )


if __name__ == "__main__":
    main()
