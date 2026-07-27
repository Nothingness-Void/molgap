"""Build the leakage-controlled archive-r01 Router development table."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit

from molgap.constants import MODELS_DIR, PARAMS_GPS_2D, PARAMS_SCHNET_300K, RESULTS_DIR, SEED
from molgap.gps import GPSWrapper
from molgap.router import EmbeddingRouterFeatures
from molgap.schnet import SchNetWrapper
from molgap.utils import ensure_dirs, load_aligned_encoder_embeddings, murcko_scaffold_smiles


PHASE8 = RESULTS_DIR / "phase8"
OUT_DIR = PHASE8 / "archive" / "archive-r01-learned-router"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--oracle-predictions", type=Path, default=OUT_DIR / "oracle_predictions.parquet"
    )
    parser.add_argument(
        "--emb-base", type=Path, default=PHASE8 / "gps_expansion_500k_embeddings.pt"
    )
    parser.add_argument(
        "--emb-extra", type=Path, default=PHASE8 / "gps_arch_depth9_embeddings.pt"
    )
    parser.add_argument(
        "--emb-3d", type=Path, default=PHASE8 / "schnet_expansion_500k_embeddings.pt"
    )
    parser.add_argument(
        "--graphs-3d", type=Path,
        default=PHASE8 / "pyg_3d_graphs_etkdg_expansion_500k.pt",
    )
    parser.add_argument(
        "--gps-model", type=Path, default=MODELS_DIR / "phase8_gps_expansion_500k.pt"
    )
    parser.add_argument(
        "--schnet-model", type=Path,
        default=MODELS_DIR / "phase8_schnet_expansion_500k.pt",
    )
    parser.add_argument("--pca-components", type=int, default=16)
    parser.add_argument("--prototype-clusters", type=int, default=64)
    parser.add_argument("--prototype-samples", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out", type=Path, default=OUT_DIR / "router_dataset.parquet")
    parser.add_argument("--split-out", type=Path, default=OUT_DIR / "router_split.npz")
    parser.add_argument(
        "--projector-out", type=Path, default=OUT_DIR / "embedding_projector.pkl"
    )
    parser.add_argument(
        "--manifest-out", type=Path, default=OUT_DIR / "router_dataset_manifest.json"
    )
    return parser.parse_args()


def scaffold_split(smiles: list[str], seed: int):
    scaffolds = []
    for i, value in enumerate(smiles, start=1):
        scaffold = murcko_scaffold_smiles(value) or "INVALID"
        # Empty Murcko scaffolds cannot be one giant group without losing most
        # acyclic chemistry from two of the three splits.
        if scaffold == "NO_SCAFFOLD":
            scaffold = f"NO_SCAFFOLD::{value}"
        scaffolds.append(scaffold)
        if i % 10_000 == 0:
            print(f"Scaffolds: {i}/{len(smiles)}", flush=True)
    groups = np.asarray(scaffolds, dtype=object)
    indices = np.arange(len(smiles))
    outer = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
    train_val, test = next(outer.split(indices, groups=groups))
    inner = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    train_rel, val_rel = next(inner.split(train_val, groups=groups[train_val]))
    train, val = train_val[train_rel], train_val[val_rel]
    return train, val, test, groups


@torch.no_grad()
def branch_predictions(h2: torch.Tensor, h3: torch.Tensor, args, device):
    gps = GPSWrapper(**PARAMS_GPS_2D).to(device)
    gps.load_state_dict(torch.load(args.gps_model, weights_only=True, map_location=device))
    gps.eval()
    schnet = SchNetWrapper(**PARAMS_SCHNET_300K, use_charges=True).to(device)
    schnet.load_state_dict(
        torch.load(args.schnet_model, weights_only=True, map_location=device)
    )
    schnet.eval()

    gps_pred, schnet_pred = [], []
    for start in range(0, len(h2), 4096):
        gps_pred.append(gps.head(h2[start:start + 4096, :192].to(device)).float().cpu())
        schnet_pred.append(schnet.head(h3[start:start + 4096].to(device)).float().cpu())
    return torch.cat(gps_pred).numpy(), torch.cat(schnet_pred).numpy()


def main() -> None:
    args = parse_args()
    ensure_dirs(args.out.parent, args.split_out.parent, args.projector_out.parent)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    oracle = pd.read_parquet(args.oracle_predictions)
    table = oracle.loc[oracle["dataset"].eq("internal")].copy().reset_index(drop=True)
    h2_all, h3_all, _, source_idx_all = load_aligned_encoder_embeddings(
        [args.emb_base, args.emb_extra], args.emb_3d, args.graphs_3d
    )
    permutation = np.random.RandomState(SEED).permutation(len(h2_all))
    router_pool_idx = permutation[int(0.9 * len(permutation)):]
    h2, h3 = h2_all[router_pool_idx], h3_all[router_pool_idx]
    source_idx = source_idx_all[router_pool_idx].numpy()
    if not np.array_equal(source_idx, table["source_idx"].to_numpy(dtype=np.int64)):
        raise ValueError("Oracle rows do not match the frozen internal held-out split")

    smiles = table["canonical_smiles"].fillna(table["smiles"]).astype(str).tolist()
    train, val, test, scaffolds = scaffold_split(smiles, args.seed)
    split = np.full(len(table), "", dtype=object)
    split[train], split[val], split[test] = "train", "validation", "test"
    table["scaffold"] = scaffolds
    table["split"] = split

    print("Computing frozen 2D/3D branch predictions", flush=True)
    gps_pred, schnet_pred = branch_predictions(h2, h3, args, device)
    for i, target in enumerate(("homo", "lumo", "gap")):
        table[f"gps_{target}"] = gps_pred[:, i]
        table[f"schnet_{target}"] = schnet_pred[:, i]
        table[f"abs_gps_schnet_{target}"] = np.abs(gps_pred[:, i] - schnet_pred[:, i])
    table["gap_consistency_signed"] = (
        table["base_gap"] - (table["base_lumo"] - table["base_homo"])
    )
    table["gap_consistency_abs"] = table["gap_consistency_signed"].abs()
    table["fixed_route_flag"] = table["base_gap"] < 4.0
    table["fixed_route_margin"] = 4.0 - table["base_gap"]

    print("Fitting PCA and expansion500k training prototypes", flush=True)
    reference_idx = permutation[:int(0.8 * len(permutation))]
    projector = EmbeddingRouterFeatures(
        n_components=args.pca_components,
        n_clusters=args.prototype_clusters,
        max_reference_samples=args.prototype_samples,
        random_state=args.seed,
    ).fit(
        h2[train, :192].numpy(),
        h3[train].numpy(),
        h2_all[reference_idx, :192].numpy(),
        h3_all[reference_idx].numpy(),
    )
    embedding_features = projector.transform(h2[:, :192].numpy(), h3.numpy())
    for name, values in embedding_features.items():
        table[name] = values

    table.to_parquet(args.out, index=False)
    np.savez(
        args.split_out,
        train_idx=train,
        validation_idx=val,
        test_idx=test,
        source_idx=source_idx,
    )
    joblib.dump(projector, args.projector_out)
    manifest = {
        "source": str(args.oracle_predictions),
        "n": int(len(table)),
        "split": {"train": int(len(train)), "validation": int(len(val)), "test": int(len(test))},
        "unique_scaffolds": {
            name: int(len(set(scaffolds[idx])))
            for name, idx in (("train", train), ("validation", val), ("test", test))
        },
        "scaffold_overlap": {
            "train_validation": int(len(set(scaffolds[train]) & set(scaffolds[val]))),
            "train_test": int(len(set(scaffolds[train]) & set(scaffolds[test]))),
            "validation_test": int(len(set(scaffolds[val]) & set(scaffolds[test]))),
        },
        "expert_win_rate": {
            name: float(table.iloc[idx]["expert_wins"].mean())
            for name, idx in (("train", train), ("validation", val), ("test", test))
        },
        "embedding_projector": projector.manifest(),
        "feature_columns": [
            name for name in table.columns
            if name.startswith(("gps_", "schnet_", "abs_gps_schnet_"))
            or name in {
                "base_homo", "base_lumo", "base_gap", "gap_consistency_signed",
                "gap_consistency_abs", "fixed_route_flag", "fixed_route_margin",
                "mw", "heavy_atoms", "ring_count", "aromatic_rings", "rotatable_bonds",
                "tpsa", "logp", "fraction_csp3", "hbd", "hba", "formal_charge",
                "conjugated_bonds", "aromatic_atom_fraction", "n_N", "n_O", "n_S",
                "n_F", "n_Cl", "n_Br", "n_B", "n_P", "n_Si",
            }
        ],
        "artifacts": {
            "dataset": str(args.out),
            "split": str(args.split_out),
            "projector": str(args.projector_out),
        },
    }
    args.manifest_out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest["split"], indent=2), flush=True)
    print(f"Dataset -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
