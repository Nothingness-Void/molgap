"""Atomically append a contiguous 2D graph cache to a frozen base cache."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--append", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--expected-base", type=int, required=True)
    parser.add_argument("--expected-append", type=int, required=True)
    parser.add_argument("--delete-append-after-success", action="store_true")
    return parser.parse_args()


def source_indices(graphs) -> torch.Tensor:
    return torch.tensor([int(graph.source_idx.view(-1)[0]) for graph in graphs], dtype=torch.long)


def atomic_json(value: dict, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    for path in (args.base, args.append):
        if not path.is_file():
            raise FileNotFoundError(path)
    base = torch.load(args.base, weights_only=False)
    append = torch.load(args.append, weights_only=False)
    if len(base) != args.expected_base or len(append) != args.expected_append:
        raise ValueError(
            f"Unexpected cache sizes: {len(base):,} + {len(append):,}; expected "
            f"{args.expected_base:,} + {args.expected_append:,}"
        )
    base_idx = source_indices(base)
    append_idx = source_indices(append)
    if not torch.equal(base_idx, torch.arange(args.expected_base)):
        raise ValueError("Base source_idx is not contiguous from zero")
    if not torch.equal(
        append_idx,
        torch.arange(args.expected_base, args.expected_base + args.expected_append),
    ):
        raise ValueError("Append source_idx does not continue the base range")

    base.extend(append)
    del append
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name(f".{args.out.name}.tmp")
    torch.save(base, temporary)
    os.replace(temporary, args.out)
    expected_total = args.expected_base + args.expected_append
    report = {
        "base": str(args.base),
        "append": str(args.append),
        "out": str(args.out),
        "base_graphs": args.expected_base,
        "append_graphs": args.expected_append,
        "total_graphs": expected_total,
        "source_idx_min": int(base_idx.min()),
        "source_idx_max": int(append_idx.max()),
        "contiguous_source_idx": True,
        "output_bytes": args.out.stat().st_size,
    }
    atomic_json(report, args.report)
    if args.delete_append_after_success:
        args.append.unlink()
        report["append_cache_deleted_after_success"] = True
        atomic_json(report, args.report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
