"""Thin entry point for K1 sparse-pair profiling and training."""
from __future__ import annotations

import argparse
from pathlib import Path

from molgap.pcqm_k1_sparse_pair_100k import run_profile, run_training


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("profile", "train"))
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--source-archive-sha256", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--platform-id", required=True)
    parser.add_argument("--profile", type=Path)
    args = parser.parse_args()
    common = {
        "dataset_root": args.dataset_root,
        "manifest_path": args.manifest,
        "source_archive": args.source_archive,
        "source_archive_sha256": args.source_archive_sha256,
        "source_commit": args.source_commit,
        "output": args.output,
        "platform_id": args.platform_id,
    }
    if args.mode == "profile":
        run_profile(**common)
        return
    if args.profile is None:
        parser.error("train mode requires --profile")
    run_training(profile_path=args.profile, **common)


if __name__ == "__main__":
    main()
