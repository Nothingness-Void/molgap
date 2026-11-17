"""Validate fixed repaired-2M fusion inputs before a scheduled training run."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from molgap.multi2d_router_fusion import metric_block
from molgap.router import paired_bootstrap_mean


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def embedding_probe(directory: Path) -> dict:
    paths = sorted(directory.glob("embeddings_*.pt"))
    if len(paths) != 100:
        raise RuntimeError(f"Expected 100 embedding parts in {directory}")
    probes = []
    for path in (paths[0], paths[-1]):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        source_idx = payload["source_idx"]
        embeddings = payload["embeddings"]
        targets = payload["targets"]
        if (
            source_idx.ndim != 1
            or embeddings.shape != (len(source_idx), 176)
            or targets.shape != (len(source_idx), 3)
            or not torch.isfinite(embeddings).all()
            or not torch.isfinite(targets).all()
        ):
            raise RuntimeError(f"Invalid embedding probe: {path}")
        probes.append(
            {
                "path": path.name,
                "rows": len(source_idx),
                "source_min": int(source_idx.min()),
                "source_max": int(source_idx.max()),
            }
        )
    return {"parts": len(paths), "probes": probes}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-2d", type=Path, required=True)
    parser.add_argument("--frozen-sha256", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--primary-embeddings", type=Path, required=True)
    parser.add_argument("--augmented-embeddings", type=Path, required=True)
    parser.add_argument("--embedding-acceptance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if sha256(args.frozen_2d) != args.frozen_sha256:
        raise RuntimeError("Frozen 2D SHA256 differs")
    if sha256(args.manifest) != args.manifest_sha256:
        raise RuntimeError("Repaired-2M manifest SHA256 differs")
    acceptance = json.loads(
        args.embedding_acceptance.read_text(encoding="utf-8")
    )
    if (
        acceptance.get("status") != "accepted"
        or int(acceptance.get("parts_per_variant", -1)) != 100
        or int(acceptance.get("rows", -1)) != 1_989_116
    ):
        raise RuntimeError("Dual-SchNet embedding acceptance differs")

    frozen = torch.load(
        args.frozen_2d, map_location="cpu", weights_only=False
    )
    if (
        frozen.get("format") != "molgap-three-gps-frozen-2d-test-v1"
        or frozen["targets"].ndim != 2
        or frozen["targets"].shape[1] != 3
        or not torch.isfinite(frozen["targets"]).all()
    ):
        raise RuntimeError("Frozen 2D payload contract differs")
    if args.manifest.suffix != ".npz":
        raise RuntimeError("Remote preflight requires a numeric NPZ manifest")
    with np.load(args.manifest, allow_pickle=False) as payload:
        manifest_row = payload["manifest_row"].astype(np.int64)
        scaffold_group = payload["scaffold_group"].astype(np.int64)
    if (
        len(manifest_row) != 2_000_000
        or scaffold_group.shape != manifest_row.shape
        or not np.array_equal(
            manifest_row,
            np.arange(len(manifest_row), dtype=np.int64),
        )
    ):
        raise RuntimeError("Repaired-2M manifest row count differs")

    result = {
        "status": "complete",
        "training_imports": {
            "metric_block": callable(metric_block),
            "paired_bootstrap_mean": callable(paired_bootstrap_mean),
        },
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "frozen_2d": {
            "rows": len(frozen["source_idx"]),
            "sha256": args.frozen_sha256,
        },
        "manifest": {
            "rows": len(manifest_row),
            "scaffold_groups": int(np.unique(scaffold_group).size),
            "sha256": args.manifest_sha256,
        },
        "primary": embedding_probe(args.primary_embeddings),
        "augmented": embedding_probe(args.augmented_embeddings),
        "embedding_acceptance_sha256": sha256(
            args.embedding_acceptance
        ),
    }
    atomic_json(result, args.output)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
