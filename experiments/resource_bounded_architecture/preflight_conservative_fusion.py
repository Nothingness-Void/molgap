"""Validate accepted 2D/3D fusion inputs and the conservative identity head."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from molgap.hierarchical_fusion import (
    ConservativeFusionConfig,
    ConservativeHierarchicalResidualHead,
)
from molgap.structural_encoding import sha256


def _atomic_json(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _embedding_parts(directory: Path) -> list[Path]:
    paths = sorted(directory.glob("embeddings_*.pt"))
    if len(paths) != 100:
        raise ValueError(f"Expected 100 embedding parts under {directory}, found {len(paths)}")
    return paths


def _check_embedding(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    source_idx = payload.get("source_idx")
    embeddings = payload.get("embeddings")
    targets = payload.get("targets")
    if not all(isinstance(value, torch.Tensor) for value in (source_idx, embeddings, targets)):
        raise TypeError(f"{path} misses embedding tensors")
    rows = source_idx.numel()
    if embeddings.shape != (rows, 176) or targets.shape != (rows, 3):
        raise ValueError(f"{path} has an invalid tensor shape")
    if source_idx.unique().numel() != rows:
        raise ValueError(f"{path} has duplicate source_idx")
    if not torch.isfinite(embeddings).all() or not torch.isfinite(targets).all():
        raise ValueError(f"{path} contains non-finite tensors")
    return {"path": path.name, "rows": rows, "sha256": sha256(path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-2d", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--primary-embeddings", type=Path, required=True)
    parser.add_argument("--augmented-embeddings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    frozen = torch.load(args.frozen_2d, map_location="cpu", weights_only=False)
    if frozen.get("format") != "molgap-three-gps-frozen-2d-test-v1":
        raise ValueError("Frozen 2D payload format differs")
    rows = int(frozen["source_idx"].numel())
    if rows < 100_000 or frozen["targets"].shape != (rows, 3):
        raise ValueError("Frozen 2D payload shape differs")
    with np.load(args.manifest, allow_pickle=False) as manifest:
        manifest_rows = len(manifest["manifest_row"])
    if manifest_rows != 2_000_000:
        raise ValueError("Scaffold manifest row count differs")

    primary = _embedding_parts(args.primary_embeddings)
    augmented = _embedding_parts(args.augmented_embeddings)
    sampled_parts = [
        _check_embedding(path)
        for path in (primary[0], primary[-1], augmented[0], augmented[-1])
    ]

    config = ConservativeFusionConfig()
    model = ConservativeHierarchicalResidualHead(
        32,
        np.zeros(32, dtype=np.float32),
        np.ones(32, dtype=np.float32),
        config,
    )
    base = torch.randn(8, 3)
    prediction, correction, confidence = model(base, torch.randn(8, 32) * 1_000)
    if not torch.equal(prediction, base) or torch.count_nonzero(correction):
        raise RuntimeError("Conservative head is not exact identity at initialization")
    if not torch.allclose(confidence, torch.full_like(confidence, config.gate_init)):
        raise RuntimeError("Conservative confidence prior differs")

    report = {
        "status": "accepted",
        "frozen_2d_rows": rows,
        "scaffold_manifest_rows": manifest_rows,
        "primary_parts": len(primary),
        "augmented_parts": len(augmented),
        "sampled_parts": sampled_parts,
        "frozen_2d_sha256": sha256(args.frozen_2d),
        "manifest_sha256": sha256(args.manifest),
        "initial_prediction_exact_identity": True,
        "initial_gate": config.gate_init,
        "maximum_correction_eV": config.correction_scale_eV,
    }
    _atomic_json(report, args.output)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
