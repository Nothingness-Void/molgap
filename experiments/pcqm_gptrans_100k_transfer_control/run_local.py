"""Thin Windows entry point for the existing GPTrans V4 training APIs."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("reference-preflight", "reference-train", "joint-train"))
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--initial-state", type=Path, required=True)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--source-archive-sha256", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight", type=Path)
    args = parser.parse_args()

    from molgap import pcqm_gptrans_v4 as v4

    # The V4 dataset class is local to _load_datasets and cannot be pickled by
    # Windows multiprocessing spawn. Zero workers changes loading only.
    v4.LOADER_WORKERS = 0
    common = {
        "dataset_root": args.dataset_root,
        "manifest_path": args.dataset_root / "manifest.json",
        "source_archive": args.source_archive,
        "source_archive_sha256": args.source_archive_sha256,
        "source_commit": args.source_commit,
        "output": args.output,
        "platform_id": "local-rtx5060",
        "initial_state_path": args.initial_state,
    }
    started = time.perf_counter()
    if args.action == "reference-preflight":
        result = v4.run_preflight(**common)
    elif args.action == "reference-train":
        if args.preflight is None:
            parser.error("reference-train requires --preflight")
        result = v4.run_training(**common, preflight_path=args.preflight)
    else:
        from molgap.noisy_nodes import run_training_noisy_nodes

        result = run_training_noisy_nodes(
            **common,
            preflight_path=args.preflight,
            pair_update_norm=True,
            noise_std=0.15,
            loss_weight=0.1,
        )
    print(json.dumps({"action": args.action, "elapsed_seconds": time.perf_counter() - started,
                      "result": result}, default=str), flush=True)


if __name__ == "__main__":
    main()
