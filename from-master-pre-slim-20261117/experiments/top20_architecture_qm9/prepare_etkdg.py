"""CPU-side, resumable ETKDG cache builder for the top-20 QM9 screen."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from molgap.qm9_screen import (
    build_etkdg_cache,
    fixed_split,
    load_qm9_records,
    target_stats,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-size", type=int, default=100_000)
    parser.add_argument("--validation-size", type=int, default=10_000)
    parser.add_argument("--test-size", type=int, default=10_000)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--geometry-seed", type=int, default=42)
    args = parser.parse_args()

    records = load_qm9_records(args.cache_dir)
    split = fixed_split(
        len(records),
        args.train_size,
        args.validation_size,
        args.test_size,
        args.split_seed,
    )
    mean, std = target_stats(records, split.train)
    _, report = build_etkdg_cache(
        records,
        split.all_indices,
        mean,
        std,
        cache_dir=args.cache_dir,
        seed=args.geometry_seed,
    )
    expected_index_hash = hashlib.sha256(
        split.all_indices.astype(np.int64).tobytes()
    ).hexdigest()
    if report["index_sha256"] != expected_index_hash:
        raise RuntimeError("ETKDG cache index identity differs from requested split")

    report_candidates = sorted(
        (args.cache_dir / "etkdg").glob(f"graphs_*_seed{args.geometry_seed}.json")
    )
    matching_reports = [
        path for path in report_candidates
        if json.loads(path.read_text(encoding="utf-8"))["index_sha256"]
        == expected_index_hash
    ]
    if len(matching_reports) != 1:
        raise RuntimeError(f"Expected one matching ETKDG report, found {matching_reports}")
    report_path = matching_reports[0]
    cache_path = report_path.with_suffix(".pt")
    shard_dir = args.cache_dir / "etkdg" / "shards" / report_path.stem.removeprefix("graphs_")
    # The shard directory name is the graph key plus the seed; the report file
    # adds only the ``graphs_`` prefix to that identity.
    shards = sorted(shard_dir.glob("*.pt"))
    if not cache_path.exists() or not shards:
        raise RuntimeError("ETKDG final cache or independent shards are missing")

    payload = {
        "format": "molgap-top20-qm9-etkdg-acceptance-v1",
        "status": "complete",
        "protocol": {
            "train_size": args.train_size,
            "validation_size": args.validation_size,
            "test_size": args.test_size,
            "split_seed": args.split_seed,
            "geometry_seed": args.geometry_seed,
            "geometry": "ETKDGv3+MMFF200",
        },
        "report": report,
        "cache": {"path": str(cache_path), "sha256": sha256_file(cache_path)},
        "report_file": {"path": str(report_path), "sha256": sha256_file(report_path)},
        "independent_shards": [
            {"path": str(path), "sha256": sha256_file(path)} for path in shards
        ],
    }
    atomic_json(args.output_dir / "cache_acceptance.json", payload)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
