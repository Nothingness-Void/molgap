"""Thin CLI for the frozen cross-scale diagnostic."""

from __future__ import annotations

import argparse
from pathlib import Path

from molgap.pcqm_k1_cross_scale_diagnostic import run_diagnostic


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in (
        "cache-100k", "cache-500k", "k1-model", "k1-payload",
        "pair-model", "pair-payload", "target-transform", "output-root",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-archive-sha256", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    run_diagnostic(
        cache_100k=args.cache_100k,
        cache_500k=args.cache_500k,
        model_paths={"k1": args.k1_model, "pair_token": args.pair_model},
        payload_paths={"k1": args.k1_payload, "pair_token": args.pair_payload},
        transform_path=args.target_transform,
        output_root=args.output_root,
        source_commit=args.source_commit,
        source_archive_sha256=args.source_archive_sha256,
        preflight_only=args.preflight_only,
    )


if __name__ == "__main__":
    main()
