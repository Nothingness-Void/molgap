"""Thin CLI for the GPTrans Noisy Nodes 100K experiment."""
from __future__ import annotations

import argparse
from pathlib import Path

from molgap.noisy_nodes import run_training_noisy_nodes


def main() -> None:
    parser = argparse.ArgumentParser(description="GPTrans Noisy Nodes 100K runner")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--source-archive-sha256", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--platform-id", required=True)
    parser.add_argument("--initial-state", type=Path, required=True)
    parser.add_argument("--noise-std", type=float, default=0.15)
    parser.add_argument("--loss-weight", type=float, default=0.1)

    args = parser.parse_args()
    result = run_training_noisy_nodes(
        dataset_root=args.dataset_root,
        manifest_path=args.manifest,
        preflight_path=args.preflight,
        source_archive=args.source_archive,
        source_archive_sha256=args.source_archive_sha256,
        source_commit=args.source_commit,
        output=args.output,
        platform_id=args.platform_id,
        initial_state_path=args.initial_state,
        noise_std=args.noise_std,
        loss_weight=args.loss_weight,
    )
    print(result, flush=True)


if __name__ == "__main__":
    main()
