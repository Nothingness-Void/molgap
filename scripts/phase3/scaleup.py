"""
18_phase3_scaleup.py — Phase 3: Production Scale-Up.

Scales CHONSFCl MW 200-500 from 10k (Phase 2 step4) to 30k-50k,
retrains tuned LightGBM, and compares with Phase 2 baseline.

Outputs:
  data/raw/phase3_chonsfcl_mw200_500_{N}k.csv
  results/phase3/phase3_metrics.json
  results/phase3/phase3_vs_phase2_comparison.csv
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from molgap.utils import (
    METADATA_COLS,
    PROCESSED_DIR,
    RAW_DIR,
    RESULTS_DIR,
    TARGET_COLS,
    build_feature_rows_parallel,
    canonicalize_smiles,
    create_split_indices,
    ensure_dirs,
    regression_metrics,
    save_json,
)

log = logging.getLogger("phase3")

REPO_ROOT = Path(__file__).resolve().parents[2]
PHASE3_DIR = RESULTS_DIR / "phase3"

HF_SUBSET = "b3lyp_pm6_chnopsfclnakmgca500"
HF_RESOLVE = (
    "https://huggingface.co/datasets/molssiai-hub/pubchemqc-b3lyp/"
    f"resolve/main/data/{HF_SUBSET}/train/{{file}}"
)
HF_API_TREE = (
    "https://huggingface.co/api/datasets/molssiai-hub/pubchemqc-b3lyp/"
    f"tree/main/data/{HF_SUBSET}/train"
)

USER_AGENT = "curl/8"
ELEMENTS = {"C", "H", "O", "N", "S", "F", "Cl"}
MW_MIN, MW_MAX = 200, 500
CSV_FIELDS = ["cid", "mw", "formula", "smiles", "homo", "lumo", "gap"]


def http_get_range(url, start, end, timeout=120, retries=3):
    headers = {"User-Agent": USER_AGENT, "Range": f"bytes={start}-{end}"}
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read(), resp.getcode()
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last_err = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"range request failed after {retries} attempts: {last_err}")


def list_hf_files():
    url = HF_API_TREE
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.load(resp)
    return sorted(d["path"].split("/")[-1] for d in data if d.get("type") == "file")


def formula_elements(formula):
    elements = set()
    i, n = 0, len(formula)
    while i < n:
        c = formula[i]
        if c.isupper():
            sym = c
            i += 1
            while i < n and formula[i].islower():
                sym += formula[i]
                i += 1
            elements.add(sym)
        else:
            i += 1
    return elements


def _f(x):
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def iter_records_from_bytes(buf):
    import ijson
    try:
        for obj in ijson.items(io.BytesIO(buf), "item"):
            yield obj
    except Exception:
        pass


# ── Fetch ──────────────────────────────────────────────────

def fetch_data(max_records, chunk_bytes):
    tag = f"{max_records // 1000}k"
    out_path = RAW_DIR / f"phase3_chonsfcl_mw200_500_{tag}.csv"

    if out_path.exists():
        df = pd.read_csv(out_path)
        if len(df) >= max_records * 0.9:
            print(f"Reusing existing {out_path} ({len(df)} rows)")
            return out_path
        print(f"Existing file has only {len(df)} rows, re-fetching")

    print(f"Fetching CHONSFCl MW {MW_MIN}-{MW_MAX}, target {max_records} records...")
    ensure_dirs(out_path.parent)

    files = list_hf_files()
    print(f"Found {len(files)} files in subset")

    total_kept = 0
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()

        for fname in tqdm(files, desc="Fetch phase3", unit="file"):
            url = HF_RESOLVE.format(file=fname)
            try:
                buf, _ = http_get_range(url, 0, chunk_bytes - 1)
            except RuntimeError:
                continue

            for obj in iter_records_from_bytes(buf):
                cid = obj.get("cid")
                mw = _f(obj.get("pubchem-molecular-weight"))
                formula = obj.get("pubchem-molecular-formula")
                smiles = obj.get("pubchem-isomeric-smiles")
                homo = _f(obj.get("energy-alpha-homo"))
                lumo = _f(obj.get("energy-alpha-lumo"))
                gap = _f(obj.get("energy-alpha-gap"))

                if homo is None or lumo is None or gap is None:
                    continue
                if mw is None or not (MW_MIN <= mw <= MW_MAX):
                    continue
                if formula is None:
                    continue
                els = formula_elements(formula)
                if not els or not els.issubset(ELEMENTS):
                    continue

                writer.writerow({
                    "cid": cid, "mw": mw, "formula": formula,
                    "smiles": smiles, "homo": homo, "lumo": lumo, "gap": gap,
                })
                total_kept += 1
                if total_kept >= max_records:
                    print(f"Reached {max_records} records")
                    return out_path

    print(f"Fetched {total_kept} records (wanted {max_records})")
    return out_path


# ── Clean ──────────────────────────────────────────────────

def clean_data(raw_path):
    df = pd.read_csv(raw_path)
    n_raw = len(df)

    for col in ["homo", "lumo", "gap", "mw"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["homo", "lumo", "gap", "smiles"])
    df = df[df["gap"] > 0]

    df["canonical_smiles"] = df["smiles"].apply(canonicalize_smiles)
    df = df.dropna(subset=["canonical_smiles"])
    df = df.drop_duplicates(subset=["canonical_smiles"])

    n_clean = len(df)
    print(f"Clean: {n_raw} → {n_clean}")
    return df.reset_index(drop=True)


# ── Features ───────────────────────────────────────────────

def generate_features(df):
    smiles_list = df["canonical_smiles"].tolist()
    print(f"Generating features for {len(smiles_list)} molecules...")

    results = build_feature_rows_parallel(smiles_list)
    print(f"Features generated: {len(results)}/{len(smiles_list)}")

    feat_rows = []
    valid_indices = []
    for idx, row in results:
        feat_rows.append(row)
        valid_indices.append(idx)

    feat_df = pd.DataFrame(feat_rows)
    meta_df = df.iloc[valid_indices][METADATA_COLS + TARGET_COLS].reset_index(drop=True)
    combined = pd.concat([meta_df, feat_df], axis=1)

    feature_cols = [c for c in combined.columns if c not in set(METADATA_COLS + TARGET_COLS)]
    nunique = combined[feature_cols].nunique()
    const_cols = nunique[nunique <= 1].index.tolist()
    combined = combined.drop(columns=const_cols)

    feature_cols = [c for c in combined.columns if c not in set(METADATA_COLS + TARGET_COLS)]
    for col in feature_cols:
        if combined[col].isna().any():
            combined[col] = combined[col].fillna(combined[col].median())

    n_feat = len([c for c in combined.columns if c not in set(METADATA_COLS + TARGET_COLS)])
    print(f"Final: {len(combined)} rows, {n_feat} features")
    return combined


# ── Train & Evaluate ───────────────────────────────────────

def train_and_evaluate(df, seed=42):
    from lightgbm import LGBMRegressor
    from sklearn.multioutput import MultiOutputRegressor

    required = set(METADATA_COLS + TARGET_COLS)
    feature_cols = [c for c in df.columns if c not in required]
    X = df[feature_cols].values.astype(np.float32)
    y = df[TARGET_COLS].values.astype(np.float32)

    train_idx, valid_idx, test_idx = create_split_indices(len(df), random_state=seed)
    tv_idx = np.concatenate([train_idx, valid_idx])

    model = MultiOutputRegressor(LGBMRegressor(
        n_estimators=800, learning_rate=0.06, num_leaves=39,
        max_depth=10, min_child_samples=23, subsample=0.888,
        colsample_bytree=0.604, reg_alpha=0.00556, reg_lambda=0.00920,
        random_state=seed, n_jobs=-1, verbose=-1,
    ))
    model.fit(X[tv_idx], y[tv_idx])
    pred = model.predict(X[test_idx])
    metrics = regression_metrics(y[test_idx], pred)

    print(f"\nPhase 3 Results:")
    for t in TARGET_COLS:
        m = metrics[t]
        print(f"  {t:5s}: MAE={m['mae']:.4f}  RMSE={m['rmse']:.4f}  R2={m['r2']:.4f}")
    avg = metrics["average"]
    print(f"  avg  : MAE={avg['mae']:.4f}  RMSE={avg['rmse']:.4f}  R2={avg['r2']:.4f}")

    return metrics, len(df), len(feature_cols), len(tv_idx), len(test_idx)


# ── Comparison ─────────────────────────────────────────────

def build_comparison(metrics, n_molecules, n_features):
    phase2_path = RESULTS_DIR / "generalization" / "step4_chonsfcl_mw200_500_metrics.json"
    rows = []

    if phase2_path.exists():
        with open(phase2_path) as f:
            p2 = json.load(f)
        p2m = p2["metrics"]
        rows.append({
            "phase": "Phase 2 (10k)", "n_molecules": p2["n_molecules"],
            "n_features": p2["n_features"],
            **{f"{t}_{k}": p2m[t][k] for t in TARGET_COLS + ["average"] for k in ["mae", "rmse", "r2"]},
        })

    rows.append({
        "phase": f"Phase 3 ({n_molecules // 1000}k)", "n_molecules": n_molecules,
        "n_features": n_features,
        **{f"{t}_{k}": metrics[t][k] for t in TARGET_COLS + ["average"] for k in ["mae", "rmse", "r2"]},
    })

    comp = pd.DataFrame(rows)
    comp_path = PHASE3_DIR / "phase3_vs_phase2_comparison.csv"
    comp.to_csv(comp_path, index=False)
    print(f"\nComparison saved to {comp_path}")

    if len(rows) == 2:
        p2_r2 = rows[0]["average_r2"]
        p3_r2 = rows[1]["average_r2"]
        p2_mae = rows[0]["average_mae"]
        p3_mae = rows[1]["average_mae"]
        print(f"\n{'='*50}")
        print(f"Phase 2 (10k):  avg MAE={p2_mae:.4f}  R2={p2_r2:.4f}")
        print(f"Phase 3 ({n_molecules//1000}k):  avg MAE={p3_mae:.4f}  R2={p3_r2:.4f}")
        delta_mae = p3_mae - p2_mae
        delta_r2 = p3_r2 - p2_r2
        print(f"Delta MAE={delta_mae:+.4f}  Delta R2={delta_r2:+.4f}")
        if delta_r2 > 0:
            print("✓ Scale-up recovered accuracy as hypothesized")
        else:
            print("✗ Scale-up did not improve — may need model retuning")
        print(f"{'='*50}")

    return comp


# ── Main ───────────────────────────────────────────────────

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")

    p = argparse.ArgumentParser(description="Phase 3: Production Scale-Up")
    p.add_argument("--max-records", type=int, default=30000)
    p.add_argument("--chunk-bytes", type=int, default=50_000_000)
    p.add_argument("--raw-csv", type=str, default=None,
                   help="Path to pre-fetched raw CSV; skips the fetch step")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    ensure_dirs(PHASE3_DIR)

    if args.raw_csv:
        raw_path = Path(args.raw_csv)
        print(f"Using pre-fetched data: {raw_path} ({len(pd.read_csv(raw_path))} rows)")
    else:
        raw_path = fetch_data(args.max_records, args.chunk_bytes)
    print(f"\n{'='*50}")
    print(f"[1/4] Cleaning data...")
    print(f"{'='*50}")
    df_clean = clean_data(raw_path)

    print(f"\n{'='*50}")
    print(f"[2/4] Generating features ({len(df_clean)} molecules)...")
    print(f"{'='*50}")
    df_feat = generate_features(df_clean)

    print(f"\n{'='*50}")
    print(f"[3/4] Training & evaluating...")
    print(f"{'='*50}")
    metrics, n_mol, n_feat, n_train, n_test = train_and_evaluate(df_feat, seed=args.seed)

    result = {
        "phase": "phase3_scaleup",
        "elements": sorted(ELEMENTS),
        "mw_range": f"{MW_MIN}-{MW_MAX}",
        "max_records_requested": args.max_records,
        "n_molecules": n_mol,
        "n_features": n_feat,
        "n_train": n_train,
        "n_test": n_test,
        "metrics": metrics,
    }
    save_json(result, PHASE3_DIR / "phase3_metrics.json")

    build_comparison(metrics, n_mol, n_feat)
    print(f"\nAll Phase 3 outputs saved to {PHASE3_DIR}/")


if __name__ == "__main__":
    raise SystemExit(main())
