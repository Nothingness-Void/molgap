"""Split contiguous PyG graph caches into durable, resumable training shards."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--expected-count", type=int, action="append", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--shard-size", type=int, default=100_000)
    return parser.parse_args()


def atomic_torch_save(value: object, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def atomic_json_write(value: dict, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def source_idx(graph) -> int:
    return int(graph.source_idx.view(-1)[0])


def validate_shard(path: Path, count: int, start: int) -> None:
    graphs = torch.load(path, weights_only=False)
    if len(graphs) != count:
        raise ValueError(f"{path}: expected {count:,} graphs, found {len(graphs):,}")
    for offset, graph in enumerate(graphs):
        actual = source_idx(graph)
        expected = start + offset
        if actual != expected:
            raise ValueError(f"{path}: source_idx[{offset}]={actual}, expected {expected}")


def main() -> None:
    args = parse_args()
    if len(args.input) != len(args.expected_count):
        raise ValueError("--input and --expected-count must be paired")
    if args.shard_size <= 0:
        raise ValueError("--shard-size must be positive")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)

    entries: list[dict] = []
    global_start = 0
    shard_number = 0
    for cache_path, expected_count in zip(args.input, args.expected_count):
        if not cache_path.is_file():
            raise FileNotFoundError(cache_path)
        expected_parts = []
        for local_start in range(0, expected_count, args.shard_size):
            count = min(args.shard_size, expected_count - local_start)
            path = args.out_dir / f"graphs-{shard_number:03d}-{global_start + local_start:09d}.pt"
            expected_parts.append((path, count, global_start + local_start, local_start))
            shard_number += 1

        missing = [part for part in expected_parts if not part[0].is_file()]
        graphs = None
        if missing:
            print(f"Loading {cache_path} for {len(missing)} missing shard(s)", flush=True)
            graphs = torch.load(cache_path, weights_only=False)
            if len(graphs) != expected_count:
                raise ValueError(
                    f"{cache_path}: expected {expected_count:,} graphs, found {len(graphs):,}"
                )

        for path, count, start, local_start in expected_parts:
            if path.is_file():
                validate_shard(path, count, start)
                print(f"Reused {path} ({count:,})", flush=True)
            else:
                shard = graphs[local_start:local_start + count]
                for offset, graph in enumerate(shard):
                    actual = source_idx(graph)
                    expected = start + offset
                    if actual != expected:
                        raise ValueError(
                            f"{cache_path}: source_idx[{local_start + offset}]={actual}, expected {expected}"
                        )
                atomic_torch_save(shard, path)
                print(f"Saved {path} ({count:,})", flush=True)
            entries.append({
                "path": path.as_posix(),
                "n_graphs": count,
                "source_idx_start": start,
                "source_idx_end": start + count,
                "bytes": path.stat().st_size,
            })
            atomic_json_write({
                "format": "molgap-pyg-shards-v1",
                "complete": False,
                "total_graphs": global_start + local_start + count,
                "shards": entries,
            }, args.manifest)
        del graphs
        global_start += expected_count

    manifest = {
        "format": "molgap-pyg-shards-v1",
        "complete": True,
        "total_graphs": global_start,
        "shard_size": args.shard_size,
        "source_caches": [
            {"path": path.as_posix(), "expected_count": count}
            for path, count in zip(args.input, args.expected_count)
        ],
        "shards": entries,
    }
    atomic_json_write(manifest, args.manifest)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
