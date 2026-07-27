"""Materialize primary and augmented lightweight SchNet Kaggle kernels."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "run_light_schnet.py"
DATASETS = [
    "nothingnessvoid/1m-full",
    "nothingnessvoid/molgap-pc100k-second-conformer-v3-20260725",
    "nothingnessvoid/molgap-pc100k-schnet-code-20260725",
]


def main() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    for variant in ("primary", "augmented"):
        target = ROOT / "packages" / variant
        target.mkdir(parents=True, exist_ok=True)
        packaged = source.replace("VARIANT = None", f'VARIANT = "{variant}"', 1)
        if packaged == source:
            raise RuntimeError("Could not embed VARIANT")
        (target / SOURCE.name).write_text(packaged, encoding="utf-8")
        metadata = {
            "id": f"nothingnessvoid/molgap-pc100k-light-schnet-{variant}",
            "title": f"MolGap PC100K Light SchNet {variant.title()}",
            "code_file": SOURCE.name,
            "language": "python",
            "kernel_type": "script",
            "is_private": "true",
            "enable_gpu": "true",
            "enable_internet": "true",
            "dataset_sources": DATASETS,
            "competition_sources": [],
            "kernel_sources": [],
            "model_sources": [],
        }
        (target / "kernel-metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
