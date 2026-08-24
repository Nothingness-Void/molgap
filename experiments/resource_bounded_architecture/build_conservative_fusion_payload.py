"""Build the compact Colab input for conservative 2D/3D fusion."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from molgap.conservative_fusion_payload import build_training_payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-2d", type=Path, required=True)
    parser.add_argument("--scaffold-manifest", type=Path, required=True)
    parser.add_argument("--primary-embeddings", type=Path, required=True)
    parser.add_argument("--augmented-embeddings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    report = build_training_payload(
        frozen_2d_path=args.frozen_2d,
        scaffold_manifest_path=args.scaffold_manifest,
        primary_embeddings=args.primary_embeddings,
        augmented_embeddings=args.augmented_embeddings,
        output_path=args.output,
        manifest_path=args.manifest,
    )
    print(json.dumps({key: report[key] for key in ("status", "rows", "split")}))


if __name__ == "__main__":
    main()
