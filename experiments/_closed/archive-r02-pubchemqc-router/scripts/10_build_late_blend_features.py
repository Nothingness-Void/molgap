"""Build post-Expert 7/9-GPS and Fusion features without rerunning ETKDG."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

from molgap.constants import RESULTS_DIR
from molgap.inference import encode_smiles_batch_dual_gps_2d, load_routed_dual_gps_hybrid
from molgap.pubchemqc import sha256_file
from molgap.utils import ensure_dirs


OUT_DIR = RESULTS_DIR / "phase8" / "archive" / "archive-r02-pubchemqc-router"
TARGETS = ("homo", "lumo", "gap")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=OUT_DIR / "router_development_dataset_r5.parquet")
    parser.add_argument("--embeddings", type=Path, default=OUT_DIR / "router_development_embeddings.npz")
    parser.add_argument("--out", type=Path, default=OUT_DIR / "late_blend_features.parquet")
    parser.add_argument("--arrays-out", type=Path, default=OUT_DIR / "late_blend_embeddings.npz")
    parser.add_argument("--manifest-out", type=Path, default=OUT_DIR / "late_blend_feature_manifest.json")
    parser.add_argument("--chunk-size", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    denominator = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return np.sum(a * b, axis=1) / np.maximum(denominator, 1e-12)


def main() -> None:
    args = parse_args()
    table = pd.read_parquet(args.dataset)
    with np.load(args.embeddings) as raw:
        if not np.array_equal(raw["probe_idx"], table.probe_idx.to_numpy()):
            raise ValueError("Stored Base embeddings do not align with development rows")
        stored_base = raw["h2"].copy()
        h3 = raw["h3"].copy()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = load_routed_dual_gps_hybrid(device)
    base_rows, extra_rows, base_pred_rows, extra_pred_rows = [], [], [], []
    for start in range(0, len(table), args.chunk_size):
        stop = min(start + args.chunk_size, len(table))
        valid, base, extra, base_pred, extra_pred = encode_smiles_batch_dual_gps_2d(
            table.canonical_smiles.iloc[start:stop].tolist(), models=models
        )
        if len(valid) != stop - start or not np.array_equal(valid, np.arange(stop - start)):
            raise ValueError(f"2D encoding failed in rows {start}:{stop}")
        base_rows.append(base); extra_rows.append(extra)
        base_pred_rows.append(base_pred); extra_pred_rows.append(extra_pred)
        print(f"2D {stop:,}/{len(table):,}", flush=True)
    base = np.concatenate(base_rows)
    extra = np.concatenate(extra_rows)
    base_gps_pred = np.concatenate(base_pred_rows)
    extra_gps_pred = np.concatenate(extra_pred_rows)
    max_base_difference = float(np.max(np.abs(base - stored_base)))
    if max_base_difference > 1e-5:
        raise ValueError(f"Recomputed 7-layer embedding mismatch: {max_base_difference}")

    base_fusion, expert_fusion = [], []
    with torch.no_grad():
        for start in range(0, len(table), 4096):
            stop = min(start + 4096, len(table))
            base_tensor = torch.from_numpy(base[start:stop]).to(device)
            extra_tensor = torch.from_numpy(extra[start:stop]).to(device)
            h3_tensor = torch.from_numpy(h3[start:stop]).to(device)
            base_fusion.append(models["base_fusion"].encode(base_tensor, h3_tensor).cpu().numpy())
            expert_fusion.append(
                models["dual_fusion"].encode(
                    torch.cat([base_tensor, extra_tensor], dim=-1), h3_tensor
                ).cpu().numpy()
            )
    base_fusion = np.concatenate(base_fusion)
    expert_fusion = np.concatenate(expert_fusion)
    gps_difference = extra - base
    fusion_difference = expert_fusion - base_fusion
    train = table.split.eq("train").to_numpy()
    gps_pca = PCA(n_components=16, svd_solver="randomized", random_state=args.seed).fit(gps_difference[train])
    fusion_pca = PCA(n_components=16, svd_solver="randomized", random_state=args.seed).fit(fusion_difference[train])
    gps_z = gps_pca.transform(gps_difference)
    fusion_z = fusion_pca.transform(fusion_difference)

    features = pd.DataFrame({"probe_idx": table.probe_idx})
    for index, target in enumerate(TARGETS):
        features[f"gps9_{target}"] = extra_gps_pred[:, index]
        features[f"gps9_minus_gps7_{target}"] = extra_gps_pred[:, index] - base_gps_pred[:, index]
        features[f"abs_gps9_minus_gps7_{target}"] = np.abs(features[f"gps9_minus_gps7_{target}"])
    for index in range(16):
        features[f"gps7_9_diff_pca_{index + 1:02d}"] = gps_z[:, index]
        features[f"fusion_diff_pca_{index + 1:02d}"] = fusion_z[:, index]
    features["gps7_9_diff_norm"] = np.linalg.norm(gps_difference, axis=1)
    features["gps7_9_cosine"] = cosine(base, extra)
    features["fusion_diff_norm"] = np.linalg.norm(fusion_difference, axis=1)
    features["fusion_cosine"] = cosine(base_fusion, expert_fusion)
    ensure_dirs(args.out.parent)
    features.to_parquet(args.out, index=False)
    np.savez_compressed(
        args.arrays_out, probe_idx=table.probe_idx.to_numpy(), extra_h2=extra,
        base_fusion=base_fusion, expert_fusion=expert_fusion,
    )
    manifest = {
        "n": len(features),
        "geometry_rerun": False,
        "max_recomputed_base_embedding_abs_difference": max_base_difference,
        "gps_difference_pca_explained_variance": float(gps_pca.explained_variance_ratio_.sum()),
        "fusion_difference_pca_explained_variance": float(fusion_pca.explained_variance_ratio_.sum()),
        "artifacts_sha256": {
            args.out.name: sha256_file(args.out),
            args.arrays_out.name: sha256_file(args.arrays_out),
        },
    }
    args.manifest_out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
