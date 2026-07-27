"""Add leakage-safe embedding PCA and expansion500k prototype distances."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from molgap.constants import RESULTS_DIR
from molgap.pubchemqc import sha256_file
from molgap.router import EmbeddingRouterFeatures
from molgap.utils import ensure_dirs, load_aligned_encoder_embeddings


PHASE8 = RESULTS_DIR / "phase8"
OUT_DIR = PHASE8 / "archive" / "archive-r02-pubchemqc-router"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=OUT_DIR / "router_development_dataset.parquet")
    parser.add_argument("--embeddings", type=Path, default=OUT_DIR / "router_development_embeddings.npz")
    parser.add_argument("--reference-2d", type=Path, default=PHASE8 / "gps_expansion_500k_embeddings.pt")
    parser.add_argument("--reference-3d", type=Path, default=PHASE8 / "schnet_expansion_500k_embeddings.pt")
    parser.add_argument("--reference-graphs", type=Path, default=PHASE8 / "pyg_3d_graphs_etkdg_expansion_500k.pt")
    parser.add_argument("--out", type=Path, default=OUT_DIR / "router_development_dataset_r5.parquet")
    parser.add_argument("--projector-out", type=Path, default=OUT_DIR / "embedding_projector.pkl")
    parser.add_argument("--manifest-out", type=Path, default=OUT_DIR / "embedding_feature_manifest.json")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    table = pd.read_parquet(args.dataset)
    with np.load(args.embeddings) as raw:
        if not np.array_equal(raw["probe_idx"], table.probe_idx.to_numpy()):
            raise ValueError("Development embeddings are not aligned with the table")
        h2 = raw["h2"].copy()
        h3 = raw["h3"].copy()
    print("Loading aligned expansion500k reference embeddings", flush=True)
    reference_2d, reference_3d, _, _ = load_aligned_encoder_embeddings(
        [args.reference_2d], args.reference_3d, args.reference_graphs
    )
    train = table.split.eq("train").to_numpy()
    projector = EmbeddingRouterFeatures(
        n_components=16, n_clusters=64, max_reference_samples=100_000,
        random_state=args.seed,
    ).fit(
        h2[train], h3[train], reference_2d.numpy(), reference_3d.numpy()
    )
    for name, values in projector.transform(h2, h3).items():
        table[name] = values
    ensure_dirs(args.out.parent)
    table.to_parquet(args.out, index=False)
    joblib.dump(projector, args.projector_out)
    manifest = {
        "fit_rows": int(train.sum()),
        "fit_split": "train only",
        "reference": "aligned expansion500k GPS/SchNet embeddings",
        "projector": projector.manifest(),
        "dataset_sha256": sha256_file(args.out),
    }
    args.manifest_out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
