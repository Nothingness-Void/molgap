"""
Phase 8 final audit: compare Phase 7, replacement300k, and expansion500k on the same PCQM4Mv2
validation proxy used for the P8.1 coverage diagnosis.

This is NOT an OGB leaderboard submission. It is a leakage-filtered, in-domain
subset/sampling of PCQM4Mv2 valid, used as a coverage stress test:
  - official valid split only;
  - drop molecules in Phase 7 / replacement300k / expansion500k training CSVs;
  - keep CHONSFCl, MW 200-1000;
  - deterministic sample;
  - evaluate both hybrid models on the identical molecules.

Usage:
  .venv\\Scripts\\python.exe scripts/phase8/eval_pcqm4mv2_proxy.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Descriptors
from sklearn.metrics import mean_absolute_error

from molgap.constants import RAW_DIR, RESULTS_DIR
from molgap.inference import load_hybrid, predict_smiles_batch_hybrid
from molgap.utils import canonicalize_smiles

DATA_CSV = RAW_DIR / "pcqm4m-v2" / "raw" / "data.csv.gz"
SPLIT_PT = RAW_DIR / "pcqm4m-v2" / "split_dict.pt"
P7_TRAIN_CSV = RAW_DIR / "phase7_chonsfcl_mw200_1000_300k.csv"
P8_TRAIN_CSV = RAW_DIR / "phase8_replacement_300k.csv"
P8_EXPANSION_TRAIN_CSV = RAW_DIR / "phase8_expansion_500k.csv"
TAIL_PROBE_TRAIN_CSV = RAW_DIR / "phase8_tail_probe_30k.csv"
OUT_METRICS = RESULTS_DIR / "phase8" / "pcqm4mv2_proxy_p7_vs_p8_metrics.json"
OUT_PREDICTIONS = RESULTS_DIR / "phase8" / "pcqm4mv2_proxy_p7_vs_p8_predictions.csv"
OUT_METRICS_3WAY = RESULTS_DIR / "phase8" / "pcqm4mv2_proxy_p7_v2_v3_metrics.json"
OUT_PREDICTIONS_3WAY = RESULTS_DIR / "phase8" / "pcqm4mv2_proxy_p7_v2_v3_predictions.csv"

ALLOWED = {"C", "H", "O", "N", "S", "F", "Cl"}
MW_MIN, MW_MAX = 200.0, 1000.0
SIM_BINS = [(0.0, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 1.01)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-csv", type=Path, default=DATA_CSV)
    parser.add_argument("--split-pt", type=Path, default=SPLIT_PT)
    parser.add_argument("--p7-train-csv", type=Path, default=P7_TRAIN_CSV)
    parser.add_argument("--p8-train-csv", type=Path, default=P8_TRAIN_CSV)
    parser.add_argument("--p8-expansion-train-csv", type=Path, default=P8_EXPANSION_TRAIN_CSV)
    parser.add_argument("--tail-train-csv", type=Path, default=TAIL_PROBE_TRAIN_CSV)
    parser.add_argument("--n-sample", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--p7-key", default="phase7_hybrid")
    parser.add_argument("--p8-key", default="phase8_replacement_hybrid")
    parser.add_argument("--v3-key", default="phase8_expansion_hybrid")
    parser.add_argument("--tail-key", default="",
                        help="optional fourth model key, e.g. phase8_tail_probe_hybrid")
    parser.add_argument("--metrics-out", type=Path, default=OUT_METRICS_3WAY)
    parser.add_argument("--predictions-out", type=Path, default=OUT_PREDICTIONS_3WAY)
    parser.add_argument("--skip-similarity", action="store_true")
    return parser.parse_args()


def in_dist(smiles: str) -> bool:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    elements = {a.GetSymbol() for a in mol.GetAtoms()}
    if any(a.GetTotalNumHs() for a in mol.GetAtoms()):
        elements.add("H")
    return not (elements - ALLOWED) and MW_MIN <= Descriptors.MolWt(mol) <= MW_MAX


def fp_of(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)


def load_train_canon(paths: list[Path]) -> tuple[set[str], dict[str, int]]:
    all_smiles: set[str] = set()
    counts: dict[str, int] = {}
    for path in paths:
        df = pd.read_csv(path, usecols=["canonical_smiles"])
        smiles = set(df["canonical_smiles"].dropna().astype(str))
        counts[str(path)] = len(smiles)
        all_smiles.update(smiles)
    return all_smiles, counts


def build_subset(args: argparse.Namespace, train_canon: set[str]) -> tuple[pd.DataFrame, dict]:
    if not args.data_csv.exists() or not args.split_pt.exists():
        raise SystemExit(
            "PCQM4Mv2 local files not found. Run scripts/phase9/benchmark_pcqm4mv2.py "
            "once if the dataset needs to be downloaded."
        )

    df = pd.read_csv(args.data_csv)
    split = torch.load(args.split_pt, weights_only=False)
    valid = df.iloc[np.array(split["valid"])].reset_index(drop=True)

    rows = []
    n_overlap = 0
    n_ood = 0
    n_bad = 0
    for _, row in valid.iterrows():
        smi = row["smiles"]
        can = canonicalize_smiles(smi)
        if can is None:
            n_bad += 1
            continue
        if can in train_canon:
            n_overlap += 1
            continue
        if not in_dist(smi):
            n_ood += 1
            continue
        rows.append({"pcqm_idx": int(row["idx"]), "smiles": smi, "canonical_smiles": can,
                     "gap_true": float(row["homolumogap"])})

    subset = pd.DataFrame(rows)
    n_usable_before_sample = len(subset)
    if len(subset) > args.n_sample:
        subset = subset.sample(
            args.n_sample, random_state=np.random.RandomState(args.seed)
        ).reset_index(drop=True)

    stats = {
        "n_valid_total": int(len(valid)),
        "n_bad_smiles": int(n_bad),
        "n_train_overlap_union": int(n_overlap),
        "n_ood": int(n_ood),
        "n_usable_before_sample": int(n_usable_before_sample),
        "n_sample_requested": int(args.n_sample),
        "n_sampled": int(len(subset)),
        "seed": int(args.seed),
    }
    return subset, stats


def predict_key(subset: pd.DataFrame, key: str, label: str) -> pd.DataFrame:
    print(f"Predicting {label}: {key}")
    models = load_hybrid(key=key)
    valid_idx, preds = predict_smiles_batch_hybrid(
        subset["smiles"].tolist(), models=models
    )
    out = subset.iloc[valid_idx].copy().reset_index(drop=True)
    out[f"{label}_homo_pred"] = preds[:, 0]
    out[f"{label}_lumo_pred"] = preds[:, 1]
    out[f"{label}_gap_pred"] = preds[:, 2]
    out[f"{label}_gap_abs_err"] = np.abs(out[f"{label}_gap_pred"] - out["gap_true"])
    return out


def add_p7_similarity(df: pd.DataFrame, train_csv: Path) -> pd.DataFrame:
    print("Computing nearest-neighbor similarity to Phase 7 training set...")
    train = pd.read_csv(train_csv, usecols=["canonical_smiles"])
    train_fps = [
        fp for fp in (fp_of(s) for s in train["canonical_smiles"].dropna().astype(str))
        if fp is not None
    ]
    sims = []
    for smi in df["smiles"]:
        fp = fp_of(smi)
        if fp is None:
            sims.append(np.nan)
        else:
            sims.append(float(max(DataStructs.BulkTanimotoSimilarity(fp, train_fps))))
    out = df.copy()
    out["p7_train_max_sim"] = sims
    return out


def metrics_for(df: pd.DataFrame, labels: list[str]) -> dict:
    result: dict = {}
    for label in labels:
        result[label] = {
            "n": int(len(df)),
            "gap_mae": float(mean_absolute_error(df["gap_true"], df[f"{label}_gap_pred"])),
            "gap_err_mean": float(df[f"{label}_gap_abs_err"].mean()),
            "gap_err_median": float(df[f"{label}_gap_abs_err"].median()),
        }

    if "p7_gap_abs_err" in df and "p8_gap_abs_err" in df:
        result["delta_p8_minus_p7"] = {
            "gap_mae": result["p8"]["gap_mae"] - result["p7"]["gap_mae"],
            "gap_err_median": result["p8"]["gap_err_median"] - result["p7"]["gap_err_median"],
        }
    if "p7_gap_abs_err" in df and "v3_gap_abs_err" in df:
        result["delta_v3_minus_p7"] = {
            "gap_mae": result["v3"]["gap_mae"] - result["p7"]["gap_mae"],
            "gap_err_median": result["v3"]["gap_err_median"] - result["p7"]["gap_err_median"],
        }
    if "p8_gap_abs_err" in df and "v3_gap_abs_err" in df:
        result["delta_v3_minus_p8"] = {
            "gap_mae": result["v3"]["gap_mae"] - result["p8"]["gap_mae"],
            "gap_err_median": result["v3"]["gap_err_median"] - result["p8"]["gap_err_median"],
        }
    if "v3_gap_abs_err" in df and "tail_gap_abs_err" in df:
        result["delta_tail_minus_v3"] = {
            "gap_mae": result["tail"]["gap_mae"] - result["v3"]["gap_mae"],
            "gap_err_median": result["tail"]["gap_err_median"] - result["v3"]["gap_err_median"],
        }
    if "p7_gap_abs_err" in df and "tail_gap_abs_err" in df:
        result["delta_tail_minus_p7"] = {
            "gap_mae": result["tail"]["gap_mae"] - result["p7"]["gap_mae"],
            "gap_err_median": result["tail"]["gap_err_median"] - result["p7"]["gap_err_median"],
        }

    if "p7_train_max_sim" in df:
        layers = []
        for lo, hi in SIM_BINS:
            mask = (df["p7_train_max_sim"] >= lo) & (df["p7_train_max_sim"] < hi)
            row: dict = {"bin": f"[{lo:.1f},{hi:.1f})", "n": int(mask.sum())}
            for label in labels:
                row[f"{label}_gap_mae"] = (
                    float(df.loc[mask, f"{label}_gap_abs_err"].mean())
                    if row["n"] else None
                )
            if row["n"] and "p7_gap_abs_err" in df and "p8_gap_abs_err" in df:
                row["delta_p8_minus_p7"] = row["p8_gap_mae"] - row["p7_gap_mae"]
            if row["n"] and "p8_gap_abs_err" in df and "v3_gap_abs_err" in df:
                row["delta_v3_minus_p8"] = row["v3_gap_mae"] - row["p8_gap_mae"]
            if row["n"] and "p7_gap_abs_err" in df and "v3_gap_abs_err" in df:
                row["delta_v3_minus_p7"] = row["v3_gap_mae"] - row["p7_gap_mae"]
            if row["n"] and "v3_gap_abs_err" in df and "tail_gap_abs_err" in df:
                row["delta_tail_minus_v3"] = row["tail_gap_mae"] - row["v3_gap_mae"]
            layers.append(row)
        result["by_p7_train_similarity"] = layers
        result["mean_p7_train_max_sim"] = float(df["p7_train_max_sim"].mean())

    return result


def main() -> None:
    args = parse_args()
    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    args.predictions_out.parent.mkdir(parents=True, exist_ok=True)

    train_paths = [args.p7_train_csv, args.p8_train_csv]
    if args.p8_expansion_train_csv.exists():
        train_paths.append(args.p8_expansion_train_csv)
    else:
        print(f"Warning: expansion train CSV not found, not excluding it: {args.p8_expansion_train_csv}")
    if args.tail_train_csv.exists():
        train_paths.append(args.tail_train_csv)
    train_canon, train_counts = load_train_canon(train_paths)
    subset, subset_stats = build_subset(args, train_canon)
    print(f"PCQM4Mv2 proxy subset: {len(subset)} molecules")
    print(
        f"  overlaps removed using train union: {subset_stats['n_train_overlap_union']}; "
        f"OOD removed: {subset_stats['n_ood']}"
    )

    p7 = predict_key(subset, args.p7_key, "p7")
    p8 = predict_key(subset, args.p8_key, "p8")
    v3 = predict_key(subset, args.v3_key, "v3")
    tail = predict_key(subset, args.tail_key, "tail") if args.tail_key else None
    common = p7.merge(
        p8[["pcqm_idx", "p8_homo_pred", "p8_lumo_pred", "p8_gap_pred", "p8_gap_abs_err"]],
        on="pcqm_idx",
        how="inner",
    ).merge(
        v3[["pcqm_idx", "v3_homo_pred", "v3_lumo_pred", "v3_gap_pred", "v3_gap_abs_err"]],
        on="pcqm_idx",
        how="inner",
    )
    if tail is not None:
        common = common.merge(
            tail[["pcqm_idx", "tail_homo_pred", "tail_lumo_pred", "tail_gap_pred", "tail_gap_abs_err"]],
            on="pcqm_idx",
            how="inner",
        )

    if not args.skip_similarity:
        common = add_p7_similarity(common, args.p7_train_csv)

    labels = ["p7", "p8", "v3"] + (["tail"] if tail is not None else [])
    metrics = {
        "note": "PCQM4Mv2 valid proxy audit, not an OGB leaderboard submission.",
        "model_keys": {"p7": args.p7_key, "p8": args.p8_key, "v3": args.v3_key, "tail": args.tail_key},
        "train_exclusion": {
            "csv_counts": train_counts,
            "union_canonical_smiles": len(train_canon),
        },
        "subset": subset_stats,
        "prediction": {
            "n_p7_valid": int(len(p7)),
            "n_p8_valid": int(len(p8)),
            "n_v3_valid": int(len(v3)),
            "n_tail_valid": int(len(tail)) if tail is not None else None,
            "n_common_valid": int(len(common)),
        },
        "metrics": metrics_for(common, labels),
    }
    common.to_csv(args.predictions_out, index=False)
    args.metrics_out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    p7_mae = metrics["metrics"]["p7"]["gap_mae"]
    p8_mae = metrics["metrics"]["p8"]["gap_mae"]
    v3_mae = metrics["metrics"]["v3"]["gap_mae"]
    tail_mae = metrics["metrics"]["tail"]["gap_mae"] if "tail" in metrics["metrics"] else None
    delta_p8 = metrics["metrics"]["delta_p8_minus_p7"]["gap_mae"]
    delta_v3_p7 = metrics["metrics"]["delta_v3_minus_p7"]["gap_mae"]
    delta_v3_p8 = metrics["metrics"]["delta_v3_minus_p8"]["gap_mae"]
    print("\nPCQM4Mv2 proxy Gap MAE")
    print(f"  P7: {p7_mae:.6f} eV")
    print(f"  v2: {p8_mae:.6f} eV")
    print(f"  v3: {v3_mae:.6f} eV")
    if tail_mae is not None:
        print(f"  tail: {tail_mae:.6f} eV")
    print(f"  delta v2-P7: {delta_p8:+.6f} eV")
    print(f"  delta v3-P7: {delta_v3_p7:+.6f} eV")
    print(f"  delta v3-v2: {delta_v3_p8:+.6f} eV")
    if "delta_tail_minus_v3" in metrics["metrics"]:
        print(f"  delta tail-v3: {metrics['metrics']['delta_tail_minus_v3']['gap_mae']:+.6f} eV")
    print(f"Saved metrics: {args.metrics_out}")
    print(f"Saved predictions: {args.predictions_out}")


if __name__ == "__main__":
    main()
