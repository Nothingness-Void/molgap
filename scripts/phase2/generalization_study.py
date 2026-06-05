"""
16_generalization_study.py — systematic generalization experiment.

Tests model performance as molecular diversity expands step by step.
Each step fetches data, cleans, generates features, trains with fixed
best-known params, and saves standardized results.

Experiment matrix:
  step0: CHON,       MW 200-300  (baseline, reuse existing)
  step1: CHON,       MW 200-500
  step2: CHONS,      MW 200-500
  step3: CHONSF,     MW 200-500
  step4: CHONSFCl,   MW 200-500

Outputs:
  results/generalization/step{N}_{tag}_metrics.json
  results/generalization/generalization_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import os
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
    RESULTS_DIR,
    TARGET_COLS,
    build_feature_rows_parallel,
    canonicalize_smiles,
    create_split_indices,
    ensure_dirs,
    regression_metrics,
    safe_mol,
    save_json,
)

log = logging.getLogger("generalization")

REPO_ROOT = Path(__file__).resolve().parents[2]
GEN_DIR = RESULTS_DIR / "phase2" / "generalization"

# ── HuggingFace config ──────────────────────────────────────

SUBSETS = {
    "chon300": "b3lyp_pm6_chon300nosalt",
    "chon500": "b3lyp_pm6_chon500nosalt",
    "broad500": "b3lyp_pm6_chnopsfclnakmgca500",
}

HF_RESOLVE = (
    "https://huggingface.co/datasets/molssiai-hub/pubchemqc-b3lyp/"
    "resolve/main/data/{subset}/train/{{file}}"
)
HF_API_TREE = (
    "https://huggingface.co/api/datasets/molssiai-hub/pubchemqc-b3lyp/"
    "tree/main/data/{subset}/train"
)

USER_AGENT = "curl/8"
CSV_FIELDS = ["cid", "mw", "formula", "smiles", "homo", "lumo", "gap"]

# ── Experiment steps ────────────────────────────────────────

STEPS = [
    {
        "name": "step0_chon_mw200_300",
        "subset": "chon300",
        "elements": {"C", "H", "O", "N"},
        "mw_min": 200, "mw_max": 300,
        "max_records": 10000,
    },
    {
        "name": "step1_chon_mw200_500",
        "subset": "chon500",
        "elements": {"C", "H", "O", "N"},
        "mw_min": 200, "mw_max": 500,
        "max_records": 10000,
    },
    {
        "name": "step2_chons_mw200_500",
        "subset": "broad500",
        "elements": {"C", "H", "O", "N", "S"},
        "mw_min": 200, "mw_max": 500,
        "max_records": 10000,
    },
    {
        "name": "step3_chonsf_mw200_500",
        "subset": "broad500",
        "elements": {"C", "H", "O", "N", "S", "F"},
        "mw_min": 200, "mw_max": 500,
        "max_records": 10000,
    },
    {
        "name": "step4_chonsfcl_mw200_500",
        "subset": "broad500",
        "elements": {"C", "H", "O", "N", "S", "F", "Cl"},
        "mw_min": 200, "mw_max": 500,
        "max_records": 10000,
    },
]

# ── HTTP helpers (reused from 01_fetch_stream.py) ───────────

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


def list_hf_files(subset_key):
    import json as _json
    url = HF_API_TREE.format(subset=SUBSETS[subset_key])
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = _json.load(resp)
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


# ── ijson parsing ───────────────────────────────────────────

def iter_records_from_bytes(buf):
    import ijson
    try:
        for obj in ijson.items(io.BytesIO(buf), "item"):
            yield obj
    except Exception:
        pass


def _f(x):
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


# ── Fetch step ──────────────────────────────────────────────

def fetch_step(step, chunk_bytes=50_000_000):
    name = step["name"]
    subset = step["subset"]
    elements = step["elements"]
    mw_min, mw_max = step["mw_min"], step["mw_max"]
    max_records = step["max_records"]

    out_path = REPO_ROOT / "data" / "raw" / f"{name}.csv"
    if out_path.exists():
        df = pd.read_csv(out_path)
        if len(df) >= max_records * 0.9:
            print(f"  [{name}] reusing existing {out_path} ({len(df)} rows)")
            return out_path
        print(f"  [{name}] existing file has only {len(df)} rows, re-fetching")

    print(f"  [{name}] fetching from {subset}, elements={elements}, MW={mw_min}-{mw_max}")
    ensure_dirs(out_path.parent)

    files = list_hf_files(subset)
    print(f"  [{name}] found {len(files)} files in subset")

    total_kept = 0
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()

        for fname in tqdm(files, desc=f"Fetch {name}", unit="file"):
            url = HF_RESOLVE.format(subset=SUBSETS[subset]).format(file=fname)
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
                if mw is None or not (mw_min <= mw <= mw_max):
                    continue
                if formula is None:
                    continue
                els = formula_elements(formula)
                if not els or not els.issubset(elements):
                    continue

                writer.writerow({"cid": cid, "mw": mw, "formula": formula,
                                 "smiles": smiles, "homo": homo, "lumo": lumo, "gap": gap})
                total_kept += 1
                if total_kept >= max_records:
                    print(f"  [{name}] reached {max_records} records")
                    return out_path

    print(f"  [{name}] fetched {total_kept} records")
    return out_path


# ── Clean + Feature + Train ─────────────────────────────────

def clean_data(raw_path, step_name):
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
    print(f"  [{step_name}] clean: {n_raw} → {n_clean}")
    return df.reset_index(drop=True)


def generate_features(df, step_name):
    smiles_list = df["canonical_smiles"].tolist()
    print(f"  [{step_name}] generating features for {len(smiles_list)} molecules...")

    results = build_feature_rows_parallel(smiles_list)
    print(f"  [{step_name}] features generated for {len(results)}/{len(smiles_list)}")

    feat_rows = []
    valid_indices = []
    for idx, row in results:
        feat_rows.append(row)
        valid_indices.append(idx)

    feat_df = pd.DataFrame(feat_rows)
    meta_df = df.iloc[valid_indices][METADATA_COLS + TARGET_COLS].reset_index(drop=True)
    combined = pd.concat([meta_df, feat_df], axis=1)

    # Drop constant columns
    feature_cols = [c for c in combined.columns if c not in set(METADATA_COLS + TARGET_COLS)]
    nunique = combined[feature_cols].nunique()
    const_cols = nunique[nunique <= 1].index.tolist()
    combined = combined.drop(columns=const_cols)

    # Fill NaN
    feature_cols = [c for c in combined.columns if c not in set(METADATA_COLS + TARGET_COLS)]
    for col in feature_cols:
        if combined[col].isna().any():
            combined[col] = combined[col].fillna(combined[col].median())

    n_feat = len([c for c in combined.columns if c not in set(METADATA_COLS + TARGET_COLS)])
    print(f"  [{step_name}] final: {len(combined)} rows, {n_feat} features")
    return combined


def train_and_evaluate(df, step_name, seed=42):
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

    print(f"  [{step_name}] results:")
    for t in TARGET_COLS:
        m = metrics[t]
        print(f"    {t:5s}: MAE={m['mae']:.4f}  R2={m['r2']:.4f}")
    avg = metrics["average"]
    print(f"    avg  : MAE={avg['mae']:.4f}  R2={avg['r2']:.4f}")

    return {
        "step": step_name,
        "n_molecules": len(df),
        "n_features": len(feature_cols),
        "n_train": len(tv_idx),
        "n_test": len(test_idx),
        "metrics": metrics,
    }


# ── Main ────────────────────────────────────────────────────

def run(steps_to_run=None, chunk_bytes=50_000_000, seed=42):
    ensure_dirs(GEN_DIR)
    all_results = []

    steps = STEPS if steps_to_run is None else [STEPS[i] for i in steps_to_run]

    for i, step in enumerate(steps):
        name = step["name"]
        print(f"\n{'='*60}")
        print(f"Step {i}: {name}")
        print(f"  elements={step['elements']}, MW={step['mw_min']}-{step['mw_max']}")
        print(f"{'='*60}")

        raw_path = fetch_step(step, chunk_bytes=chunk_bytes)
        df_clean = clean_data(raw_path, name)
        df_feat = generate_features(df_clean, name)
        result = train_and_evaluate(df_feat, name, seed=seed)

        result["elements"] = sorted(step["elements"])
        result["mw_range"] = f"{step['mw_min']}-{step['mw_max']}"

        save_json(result, GEN_DIR / f"{name}_metrics.json")
        all_results.append(result)

    # Summary table
    rows = []
    for r in all_results:
        row = {
            "step": r["step"],
            "elements": ",".join(r["elements"]),
            "mw_range": r["mw_range"],
            "n_molecules": r["n_molecules"],
            "n_features": r["n_features"],
        }
        for target in TARGET_COLS + ["average"]:
            for metric in ["mae", "rmse", "r2"]:
                row[f"{target}_{metric}"] = r["metrics"][target][metric]
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_csv(GEN_DIR / "generalization_summary.csv", index=False)
    print(f"\n{'='*60}")
    print("GENERALIZATION SUMMARY")
    print(f"{'='*60}")
    print(summary[["step", "elements", "mw_range", "n_molecules",
                    "average_mae", "average_r2"]].to_string(index=False))
    print(f"\nSaved to {GEN_DIR}/")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    p = argparse.ArgumentParser(description="Generalization study")
    p.add_argument("--steps", type=str, default=None,
                   help="Comma-separated step indices to run (e.g. '0,1,2'). Default: all")
    p.add_argument("--chunk-bytes", type=int, default=50_000_000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    steps = None
    if args.steps:
        steps = [int(x) for x in args.steps.split(",")]

    run(steps_to_run=steps, chunk_bytes=args.chunk_bytes, seed=args.seed)


if __name__ == "__main__":
    raise SystemExit(main())
