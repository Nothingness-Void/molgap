"""Kaggle1 CPU entry point for the deterministic PCQM MoSE cache."""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path


def find_one(name: str) -> Path:
    matches = list(Path("/kaggle/input").rglob(name))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one {name}, found {matches}")
    return matches[0]


def source_root() -> Path:
    archive = find_one("source_payload.bin")
    root = Path("/kaggle/working/_molgap_mose_source")
    if not root.exists():
        shutil.unpack_archive(archive, root, format="gztar")
    return root


def verify_source(root: Path) -> None:
    archive = find_one("source_payload.bin")
    expected = find_one("SOURCE_ARCHIVE_SHA256.txt").read_text().strip()
    if hashlib.sha256(archive.read_bytes()).hexdigest() != expected:
        raise RuntimeError("Source archive hash mismatch")
    inventory = json.loads(find_one("SOURCE_FILES.json").read_text())["files"]
    for item in inventory:
        path = root / item["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise RuntimeError(f"Source file hash mismatch: {item['path']}")


def main() -> None:
    root = source_root()
    verify_source(root)
    sys.path.insert(0, str(root / "src"))
    from molgap.pcqm_mose import build_mose_cache

    manifest = build_mose_cache(
        Path("/kaggle/working/molgap-pcqm-k1-mose-cache-v1"),
        workers=4,
    )
    print(json.dumps({"event": "cache_complete", **manifest}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

