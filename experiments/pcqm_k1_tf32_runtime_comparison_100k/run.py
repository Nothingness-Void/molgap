"""Thin CLI for the frozen IMS K1 FP32/TF32 two-arm diagnostic."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from molgap.pcqm_k1_tf32_comparison import run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--source-archive-sha256", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    digest = hashlib.sha256(args.source_archive.read_bytes()).hexdigest()
    if digest != args.source_archive_sha256:
        raise RuntimeError("Source archive SHA-256 mismatch")
    source_root = args.source_archive.parent
    inventory = json.loads((source_root / "SOURCE_FILES.json").read_text(encoding="utf-8"))["files"]
    if not inventory:
        raise RuntimeError("Source inventory is empty")
    for item in inventory:
        relative = Path(item["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError("Source inventory contains an unsafe path")
        path = source_root / relative
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise RuntimeError(f"Source inventory SHA-256 mismatch: {relative}")
    run(
        cache_root=args.cache_root,
        output_root=args.output_root,
        source_commit=args.source_commit,
        source_archive_sha256=digest,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
