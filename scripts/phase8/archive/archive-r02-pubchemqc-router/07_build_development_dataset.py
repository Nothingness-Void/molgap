"""Build the 49k archive-r02 Router development table and scaffold split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from molgap.constants import RESULTS_DIR
from molgap.pubchemqc import sha256_file
from molgap.router import DEFAULT_TARGET_WEIGHTS, router_descriptor_row
from molgap.router_sampling import compute_scaffold_keys
from molgap.utils import ensure_dirs


OUT_DIR = RESULTS_DIR / "phase8" / "archive" / "archive-r02-pubchemqc-router"
TARGETS = ("homo", "lumo", "gap")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle-labels", type=Path, default=OUT_DIR / "oracle_probe_gain_labels.parquet")
    parser.add_argument("--expansion-predictions", type=Path, default=OUT_DIR / "development_expansion_predictions.parquet")
    parser.add_argument("--oracle-chunks", type=Path, default=OUT_DIR / "oracle_probe_chunks")
    parser.add_argument("--expansion-chunks", type=Path, default=OUT_DIR / "development_expansion_chunks")
    parser.add_argument("--sealed-random", type=Path, default=OUT_DIR / "sealed_random_inputs.parquet")
    parser.add_argument("--sealed-hard", type=Path, default=OUT_DIR / "sealed_hard_inputs.parquet")
    parser.add_argument("--out", type=Path, default=OUT_DIR / "router_development_dataset.parquet")
    parser.add_argument("--embeddings-out", type=Path, default=OUT_DIR / "router_development_embeddings.npz")
    parser.add_argument("--manifest-out", type=Path, default=OUT_DIR / "router_development_manifest.json")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_embeddings(directories: list[Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ids, h2, h3 = [], [], []
    for directory in directories:
        for path in sorted(directory.glob("chunk-*.npz")):
            with np.load(path) as chunk:
                ids.append(chunk["probe_idx"])
                h2.append(chunk["h2"])
                h3.append(chunk["h3"])
    return np.concatenate(ids), np.concatenate(h2), np.concatenate(h3)


def main() -> None:
    args = parse_args()
    if any(path.exists() for path in (args.out, args.embeddings_out, args.manifest_out)) and not args.overwrite:
        raise FileExistsError("development dataset already exists; pass --overwrite intentionally")
    old = pd.read_parquet(args.oracle_labels)
    new = pd.read_parquet(args.expansion_predictions)
    new = new[new.prediction_success].copy()
    table = pd.concat([old, new], ignore_index=True, sort=False)
    table = table.sort_values("probe_idx").drop_duplicates("probe_idx").reset_index(drop=True)

    for target in TARGETS:
        table[f"y_{target}"] = table[target]
    y = table[[f"y_{target}" for target in TARGETS]].to_numpy(dtype=np.float64)
    base = table[[f"base_{target}" for target in TARGETS]].to_numpy(dtype=np.float64)
    expert = table[[f"expert_{target}" for target in TARGETS]].to_numpy(dtype=np.float64)
    absolute_delta = np.abs(base - y) - np.abs(expert - y)
    table["gain"] = absolute_delta[:, 2]
    table["gain_gap"] = absolute_delta[:, 2]
    table["gain_weighted"] = absolute_delta @ DEFAULT_TARGET_WEIGHTS
    table["downside"] = np.maximum(-table.gain, 0.0)
    table["expert_wins"] = table.gain > 0
    table["expert_meaningful_win_0.002"] = table.gain > 0.002
    table["expert_meaningful_win_0.005"] = table.gain > 0.005
    table["fixed_route_flag"] = table.base_gap < 4.0
    table["fixed_route_margin"] = 4.0 - table.base_gap
    table["gap_consistency_signed"] = table.base_gap - (table.base_lumo - table.base_homo)
    table["gap_consistency_abs"] = table.gap_consistency_signed.abs()
    for target in TARGETS:
        table[f"abs_gps_schnet_{target}"] = (
            table[f"gps_{target}"] - table[f"schnet_{target}"]
        ).abs()

    print(f"Descriptors for {len(table):,} rows", flush=True)
    descriptors = pd.DataFrame(
        [router_descriptor_row(value) for value in table.canonical_smiles],
        index=table.index,
    )
    for column in descriptors:
        table[column] = descriptors[column]
    table["scaffold"] = compute_scaffold_keys(
        table.canonical_smiles.tolist(), workers=args.workers
    )

    indices = np.arange(len(table))
    groups = table.scaffold.to_numpy(dtype=object)
    outer = GroupShuffleSplit(n_splits=1, test_size=0.10, random_state=args.seed)
    train_val, test = next(outer.split(indices, groups=groups))
    inner = GroupShuffleSplit(n_splits=1, test_size=2 / 9, random_state=args.seed + 1)
    train_rel, val_rel = next(inner.split(train_val, groups=groups[train_val]))
    train, validation = train_val[train_rel], train_val[val_rel]
    split = np.full(len(table), "", dtype=object)
    split[train], split[validation], split[test] = "train", "validation", "dev_test"
    table["split"] = split

    raw_weight = table.sampling_weight.to_numpy(dtype=np.float64)
    cap = float(np.quantile(raw_weight[train], 0.99))
    clipped = np.minimum(raw_weight, cap)
    table["training_weight"] = clipped / clipped[train].mean()

    embedding_ids, h2, h3 = load_embeddings([args.oracle_chunks, args.expansion_chunks])
    order = pd.Series(np.arange(len(embedding_ids)), index=embedding_ids)
    positions = order.loc[table.probe_idx].to_numpy(dtype=np.int64)
    h2, h3 = h2[positions], h3[positions]
    ensure_dirs(args.out.parent)
    table.to_parquet(args.out, index=False)
    np.savez_compressed(
        args.embeddings_out, probe_idx=table.probe_idx.to_numpy(), h2=h2, h3=h3
    )

    sealed_random = pd.read_parquet(args.sealed_random)
    sealed_hard = pd.read_parquet(args.sealed_hard)
    dev_scaffolds = set(table.scaffold)
    overlap = {
        "development_sealed_random": len(dev_scaffolds & set(sealed_random.scaffold)),
        "development_sealed_hard": len(dev_scaffolds & set(sealed_hard.scaffold)),
        "sealed_random_hard": len(set(sealed_random.scaffold) & set(sealed_hard.scaffold)),
    }
    split_scaffolds = {
        name: set(table.loc[table.split == name, "scaffold"])
        for name in ("train", "validation", "dev_test")
    }
    split_overlap = {
        "train_validation": len(split_scaffolds["train"] & split_scaffolds["validation"]),
        "train_dev_test": len(split_scaffolds["train"] & split_scaffolds["dev_test"]),
        "validation_dev_test": len(split_scaffolds["validation"] & split_scaffolds["dev_test"]),
    }
    manifest = {
        "n": len(table),
        "valid_by_source": table.sampling_source.value_counts().to_dict(),
        "split": table.split.value_counts().to_dict(),
        "unique_scaffolds": {
            name: len(values) for name, values in split_scaffolds.items()
        },
        "split_scaffold_overlap": split_overlap,
        "sealed_scaffold_overlap": overlap,
        "gain_target": "Gap absolute-error reduction: |base_gap-y| - |expert_gap-y|",
        "sampling_weight": {
            "used_for_train_validation_only": True,
            "clip_quantile": 0.99,
            "clip_value": cap,
        },
        "artifacts_sha256": {
            args.out.name: sha256_file(args.out),
            args.embeddings_out.name: sha256_file(args.embeddings_out),
        },
    }
    args.manifest_out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
