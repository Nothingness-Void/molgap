"""Materialize four bounded Kaggle conformer-preparation kernels."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "prep_second_conformer.py"


def main() -> None:
    for shard_id in range(4):
        target = ROOT / "packages" / f"r{shard_id}"
        target.mkdir(parents=True, exist_ok=True)
        source = SOURCE.read_text(encoding="utf-8")
        packaged = source.replace("SHARD_ID = None", f"SHARD_ID = {shard_id}", 1)
        if packaged == source:
            raise RuntimeError("Could not embed SHARD_ID in packaged kernel")
        (target / SOURCE.name).write_text(
            packaged,
            encoding="utf-8",
        )
        variant_path = target / "variant.json"
        if variant_path.exists():
            variant_path.unlink()
        metadata = {
            "id": f"nothingnessvoid/molgap-pc100k-conformer-r{shard_id}",
            "title": f"MolGap PC100K Conformer R{shard_id}",
            "code_file": SOURCE.name,
            "language": "python",
            "kernel_type": "script",
            "is_private": "true",
            "enable_gpu": "false",
            "enable_internet": "true",
            "dataset_sources": [
                "nothingnessvoid/molgap-pubchemqc100k-arch-split-20260725"
            ],
            "competition_sources": [],
            "kernel_sources": [],
            "model_sources": [],
        }
        (target / "kernel-metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
