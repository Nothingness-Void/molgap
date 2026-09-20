"""Thin remote CLI for the frozen chemistry-conditioned PairToken candidate."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


MODE = "neural_atom_k1_chem_typed_pair_token"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--sidecar-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-archive-sha256", required=True)
    args = parser.parse_args()
    source_root = args.source_root.resolve()
    if not (source_root / "src/molgap/k1_chem_typed_pair_token.py").is_file():
        raise FileNotFoundError("Immutable chemistry-conditioned source is incomplete")
    os.environ["MOLGAP_FIXED_CACHE_ROOT"] = str(args.cache_root.resolve())
    os.environ["MOLGAP_FUNCTIONAL_GROUP_SIDECAR_ROOT"] = str(
        args.sidecar_root.resolve()
    )
    os.environ["MOLGAP_PLATFORM_ID"] = os.environ.get(
        "MOLGAP_PLATFORM_ID", "kaggle2"
    )
    sys.path.insert(0, str(source_root / "src"))

    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Screen requires exactly one visible accelerator")
    from molgap.pcqm_k1_variants_runner import train_arm

    train_arm(
        MODE,
        args.output_root,
        source_commit=args.source_commit,
        source_archive_sha256=args.source_archive_sha256,
    )


if __name__ == "__main__":
    main()
