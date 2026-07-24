"""
Phase 8.2c: assemble a fixed-size replacement training CSV.

This is the control-variable dataset constructor:

  Phase 7 control:  old 300k
  Phase 8 variant:  old 300k - N common/easy rows + N targeted hard rows

The total training size stays fixed at 300k. The only intended variable is
coverage distribution, not more data.

Usage:
  .venv\\Scripts\\python.exe scripts/phase8/archive/legacy/data_coverage/assemble_replacement_dataset.py ^
    --candidate-csv data/raw/phase8_targeted_topup_rare_probe.csv ^
    --candidate-csv data/raw/phase8_targeted_topup_balanced_probe.csv ^
    --out-csv data/raw/phase8_replacement_300k_probe.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from molgap.constants import RAW_DIR, RESULTS_DIR
from molgap.utils import canonicalize_smiles, ensure_dirs

TRAIN_CSV = RAW_DIR / "phase7_chonsfcl_mw200_1000_300k.csv"
DESC_CACHE = RESULTS_DIR / "phase8" / "training_gap_descriptors.csv"
OUT_CSV = RAW_DIR / "phase8_replacement_300k.csv"
REPORT_JSON = RESULTS_DIR / "phase8" / "replacement_dataset_report.json"
REPORT_MD = RESULTS_DIR / "phase8" / "replacement_dataset_report.md"

TARGET_COLS = ["homo", "lumo", "gap"]
TRAIN_COLS = ["cid", "mw", "formula", "smiles", "homo", "lumo", "gap", "canonical_smiles"]


def load_old_with_descriptors(train_csv: Path, desc_cache: Path) -> pd.DataFrame:
    old = pd.read_csv(train_csv)
    desc = pd.read_csv(desc_cache)
    if len(old) != len(desc):
        raise ValueError(f"descriptor cache length mismatch: old={len(old)} desc={len(desc)}")
    old = old.reset_index(drop=True).copy()
    desc = desc.reset_index(drop=True)
    if "canonical_smiles" not in old:
        old["canonical_smiles"] = desc["canonical_smiles"]
    for col in [
        "valid_rdkit", "heavy_atoms", "fragments", "ring_count", "aromatic_rings",
        "aromatic_atom_fraction", "rotatable_bonds", "conjugated_bonds",
        "has_s", "has_cl", "has_f", "has_s_or_cl",
    ]:
        if col in desc.columns:
            old[col] = desc[col]
    for col in ["mw", *TARGET_COLS]:
        old[col] = pd.to_numeric(old[col], errors="coerce")
    return old


def normalize_candidates(paths: list[Path], old_smiles: set[str], old_cids: set[int]) -> pd.DataFrame:
    dfs = []
    for path in paths:
        df = pd.read_csv(path)
        if len(df) == 0:
            continue
        if "canonical_smiles" not in df:
            df["canonical_smiles"] = df["smiles"].apply(canonicalize_smiles)
        df["source_file"] = path.name
        dfs.append(df)
    if not dfs:
        return pd.DataFrame(columns=TRAIN_COLS + ["bucket", "source_file"])

    cand = pd.concat(dfs, ignore_index=True)
    for col in ["mw", *TARGET_COLS]:
        cand[col] = pd.to_numeric(cand[col], errors="coerce")
    cand["cid_num"] = pd.to_numeric(cand.get("cid"), errors="coerce")
    cand = cand.dropna(subset=["canonical_smiles", "smiles", "mw", *TARGET_COLS])
    cand = cand[cand["gap"] > 0].copy()
    cand = cand[~cand["canonical_smiles"].isin(old_smiles)].copy()
    cand = cand[~cand["cid_num"].dropna().astype("Int64").isin(old_cids).reindex(cand.index, fill_value=False)]
    cand = cand.drop_duplicates(subset=["canonical_smiles"]).reset_index(drop=True)
    if "bucket" not in cand:
        cand["bucket"] = "targeted"
    for col in TRAIN_COLS:
        if col not in cand:
            cand[col] = np.nan
    return cand


def add_hard_flags(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    has_s_or_cl = ((x.get("has_s", 0) == 1) | (x.get("has_cl", 0) == 1))
    aromatic_edge = (
        (x.get("aromatic_rings", 0) >= 5)
        | (x.get("aromatic_atom_fraction", 0) >= 0.85)
    )
    high_aromatic = (
        (x.get("aromatic_rings", 0) >= 4)
        | (x.get("aromatic_atom_fraction", 0) >= 0.70)
    )
    x["p8_low_gap"] = x["gap"] < 3.2
    x["p8_large"] = x["mw"] >= 500
    x["p8_aromatic_edge"] = aromatic_edge
    x["p8_scl_hard"] = has_s_or_cl & ((x["gap"] < 3.5) | high_aromatic)
    x["p8_flexible_hard"] = (x.get("rotatable_bonds", 0) >= 8) & ((x["gap"] < 3.5) | high_aromatic)
    x["p8_any_hard"] = (
        x["p8_low_gap"] | x["p8_large"] | x["p8_aromatic_edge"]
        | x["p8_scl_hard"] | x["p8_flexible_hard"]
    )
    return x


def choose_rows_to_remove(old: pd.DataFrame, n_remove: int, seed: int) -> pd.DataFrame:
    x = add_hard_flags(old)
    rng = np.random.RandomState(seed)
    x["_tie"] = rng.rand(len(x))
    score = np.zeros(len(x), dtype=float)
    score += ((x["gap"] >= 4.0) & (x["gap"] < 5.5)).astype(float) * 4.0
    score += ((x["mw"] >= 200) & (x["mw"] < 500)).astype(float) * 2.0
    score += (x.get("aromatic_rings", 99) <= 2).astype(float) * 2.0
    score += (x.get("aromatic_atom_fraction", 1.0) < 0.75).astype(float) * 1.0
    score += (x.get("rotatable_bonds", 99) <= 6).astype(float) * 1.0
    score += (~((x.get("has_s", 0) == 1) | (x.get("has_cl", 0) == 1))).astype(float) * 0.5
    score -= x["p8_any_hard"].astype(float) * 100.0
    x["_remove_score"] = score
    return x.sort_values(["_remove_score", "_tie"], ascending=[False, True]).head(n_remove)


def summarize(df: pd.DataFrame) -> dict[str, object]:
    x = add_hard_flags(df)
    out: dict[str, object] = {"n": int(len(x))}
    for name in [
        "p8_low_gap", "p8_large", "p8_aromatic_edge",
        "p8_scl_hard", "p8_flexible_hard", "p8_any_hard",
    ]:
        n = int(x[name].sum())
        out[name] = {"n": n, "fraction": n / len(x) if len(x) else 0.0}
    for col in ["mw", "homo", "lumo", "gap"]:
        vals = pd.to_numeric(x[col], errors="coerce").dropna()
        out[col] = {
            "mean": float(vals.mean()),
            "p1": float(vals.quantile(0.01)),
            "p50": float(vals.quantile(0.50)),
            "p99": float(vals.quantile(0.99)),
        }
    if "bucket" in x:
        out["bucket_counts"] = {str(k): int(v) for k, v in x["bucket"].value_counts().items()}
    return out


def write_markdown(report: dict, path: Path) -> None:
    lines = [
        "# P8 Replacement Dataset Report",
        "",
        f"- old rows: {report['old']['n']:,}",
        f"- targeted candidate rows used: {report['n_replaced']:,}",
        f"- output rows: {report['replacement']['n']:,}",
        f"- output csv: `{report['out_csv']}`",
        "",
        "## Targeted Buckets Used",
        "",
        "| bucket | n |",
        "|---|---:|",
    ]
    for bucket, n in report["candidates_used"].get("bucket_counts", {}).items():
        lines.append(f"| `{bucket}` | {n:,} |")

    lines += [
        "",
        "## Coverage Shift",
        "",
        "| flag | old n | old frac | replacement n | replacement frac | delta n |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for flag in [
        "p8_low_gap", "p8_large", "p8_aromatic_edge",
        "p8_scl_hard", "p8_flexible_hard", "p8_any_hard",
    ]:
        old = report["old"][flag]
        new = report["replacement"][flag]
        lines.append(
            f"| {flag} | {old['n']:,} | {old['fraction']:.2%} | "
            f"{new['n']:,} | {new['fraction']:.2%} | {new['n'] - old['n']:+,} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", type=Path, default=TRAIN_CSV)
    ap.add_argument("--desc-cache", type=Path, default=DESC_CACHE)
    ap.add_argument("--candidate-csv", type=Path, action="append", required=True)
    ap.add_argument("--out-csv", type=Path, default=OUT_CSV)
    ap.add_argument("--report-json", type=Path, default=REPORT_JSON)
    ap.add_argument("--report-md", type=Path, default=REPORT_MD)
    ap.add_argument("--replacement-size", type=int, default=50_000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    old = load_old_with_descriptors(args.train_csv, args.desc_cache)
    old_smiles = set(old["canonical_smiles"].dropna().astype(str).tolist())
    old_cids = set(pd.to_numeric(old["cid"], errors="coerce").dropna().astype(int).tolist())
    candidates = normalize_candidates(args.candidate_csv, old_smiles, old_cids)
    if len(candidates) == 0:
        raise ValueError("No usable candidate rows after dedup/filtering")

    n_replace = min(args.replacement_size, len(candidates), len(old))
    candidates_used = candidates.head(n_replace).copy()
    remove = choose_rows_to_remove(old, n_replace, args.seed)
    keep = old.drop(index=remove.index)

    replacement_full = pd.concat(
        [keep, candidates_used],
        ignore_index=True,
        sort=False,
    ).drop_duplicates(subset=["canonical_smiles"]).reset_index(drop=True)
    replacement = replacement_full[TRAIN_COLS].copy()
    if len(replacement) != len(old):
        raise RuntimeError(
            f"replacement size changed unexpectedly: old={len(old)} replacement={len(replacement)}"
        )

    ensure_dirs(args.out_csv.parent, args.report_json.parent, args.report_md.parent)
    replacement[TRAIN_COLS].to_csv(args.out_csv, index=False, encoding="utf-8")
    report = {
        "train_csv": str(args.train_csv),
        "candidate_csvs": [str(p) for p in args.candidate_csv],
        "out_csv": str(args.out_csv),
        "requested_replacement_size": args.replacement_size,
        "n_replaced": int(n_replace),
        "old": summarize(old),
        "removed_old": summarize(remove),
        "candidates_used": summarize(candidates_used),
        "replacement": summarize(replacement_full),
    }
    args.report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, args.report_md)
    print(f"Old rows: {len(old):,}")
    print(f"Candidates usable: {len(candidates):,}")
    print(f"Replaced rows: {n_replace:,}")
    print(f"Saved CSV: {args.out_csv}")
    print(f"Saved report: {args.report_md}")


if __name__ == "__main__":
    main()
