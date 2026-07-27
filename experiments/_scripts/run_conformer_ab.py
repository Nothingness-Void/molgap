"""Run the local 50K bare-ETKDG versus ETKDGv3+MMFF Route B A/B."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from molgap.constants import EXPERIMENTS_DIR, PLATFORMS_DIR, REPO_ROOT
from molgap.conformer_ab import (
    PROTOCOLS,
    analyze_tradeoff,
    build_protocol_cache,
    evaluate_frozen_route_b,
    select_aligned_rows,
    write_selection,
)

EXPERIMENT = EXPERIMENTS_DIR / "conformer_protocol" / "results"
CACHE = REPO_ROOT / "data" / "cache" / "phase8" / "conformer_protocol_50k"
PC100K = EXPERIMENTS_DIR / "pubchemqc100k_architecture" / "results"
ACCEPTED = PC100K / "schnet_v2_acceptance"
KAGGLE = PLATFORMS_DIR / "_records" / "kaggle" / "pubchemqc100k_architecture"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", choices=("build", "evaluate", "all"), default="all"
    )
    parser.add_argument("--output-dir", type=Path, default=EXPERIMENT)
    parser.add_argument("--cache-dir", type=Path, default=CACHE)
    parser.add_argument("--train", type=int, default=40_000)
    parser.add_argument("--validation", type=int, default=5_000)
    parser.add_argument("--test", type=int, default=5_000)
    parser.add_argument(
        "--workers", type=int, default=max(1, min(12, (os.cpu_count() or 2) - 2))
    )
    parser.add_argument("--shard-size", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selection_path = args.output_dir / "selection.json"
    if selection_path.exists():
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        rows = selection["items"]
        expected = args.train + args.validation + args.test
        if len(rows) != expected:
            raise ValueError(
                f"existing selection has {len(rows)} rows, requested {expected}"
            )
    else:
        rows = select_aligned_rows(
            PC100K / "split.csv",
            ACCEPTED / "gps11_160_embeddings_accepted.pt",
            {
                "train": args.train,
                "validation": args.validation,
                "test": args.test,
            },
        )
        write_selection(selection_path, rows)

    summaries = {}
    if args.stage in {"build", "all"}:
        for protocol in PROTOCOLS:
            summaries[protocol] = build_protocol_cache(
                rows,
                protocol,
                args.cache_dir / protocol,
                workers=args.workers,
                shard_size=args.shard_size,
                base_seed=args.seed,
            )
    else:
        summaries = {
            protocol: json.loads(
                (args.cache_dir / protocol / "summary.json").read_text(
                    encoding="utf-8"
                )
            )
            for protocol in PROTOCOLS
        }
    if args.stage == "build":
        return

    evaluation = evaluate_frozen_route_b(
        {protocol: args.cache_dir / protocol for protocol in PROTOCOLS},
        gps9_payload=ACCEPTED / "gps9_embeddings_accepted.pt",
        gps11_payload=ACCEPTED / "gps11_160_embeddings_accepted.pt",
        accepted_primary_payload=ACCEPTED / "primary_embeddings_accepted.pt",
        accepted_augmented_payload=(
            ACCEPTED / "augmented_primary_view_embeddings_accepted.pt"
        ),
        primary_checkpoint=(
            KAGGLE
            / "light_schnet_primary_v2_complete"
            / "light_schnet_primary"
            / "best.pt"
        ),
        augmented_checkpoint=(
            KAGGLE
            / "light_schnet_augmented_v2_complete"
            / "light_schnet_augmented"
            / "best.pt"
        ),
        head_checkpoints=[
            PC100K
            / "route_b_residual_scale_ab"
            / f"scale_0p10_seed{seed}"
            / "best.pt"
            for seed in (42, 43, 44)
        ],
        output_path=args.output_dir / "evaluation.json",
        batch_size=args.batch_size,
    )
    tradeoff = analyze_tradeoff(
        summaries, evaluation, args.output_dir / "tradeoff.json"
    )
    print(json.dumps(tradeoff, indent=2), flush=True)


if __name__ == "__main__":
    main()
