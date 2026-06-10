"""
Phase 7 Step 1: Fetch ~270k additional molecules (MW 200-1000) and merge with
existing Phase 3+6 data to build a 300k dataset, then generate ETKDG 3D graphs.

Usage:
  .venv\Scripts\python.exe scripts/phase7/fetch_300k.py [--target 300000]

Outputs:
  data/raw/phase7_chonsfcl_mw200_1000_300k.csv   (merged deduplicated CSV)
  results/phase7/pyg_3d_graphs_etkdg_300k.pt      (PyG graph cache)
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import re
import time
import urllib.error
import urllib.request

import ijson
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from molgap.constants import RAW_DIR, RESULTS_DIR, TARGET_COLS, DATA_PHASE3, DATA_PHASE6_LARGE
from molgap.utils import ensure_dirs, canonicalize_smiles
from molgap.graphs import build_labeled_graphs

log = logging.getLogger("phase7")

SEED = 42
HF_SUBSET = "b3lyp_pm6"
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
MW_MIN, MW_MAX = 200, 1000
CSV_FIELDS = ["cid", "mw", "formula", "smiles", "homo", "lumo", "gap"]
CHUNK_BYTES = 20_000_000

PHASE7_DIR = RESULTS_DIR / "phase7"
GRAPH_PATH = PHASE7_DIR / "pyg_3d_graphs_etkdg_300k.pt"


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
    raise RuntimeError(f"Failed after {retries} attempts: {last_err}")


def list_hf_files():
    req = urllib.request.Request(HF_API_TREE, headers={"User-Agent": USER_AGENT})
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


def load_existing_data():
    """Load and merge existing Phase 3 + Phase 6 CSVs."""
    dfs = []
    for p in [DATA_PHASE3, DATA_PHASE6_LARGE]:
        if p.exists():
            df = pd.read_csv(p)
            dfs.append(df)
            print(f"  Loaded {p.name}: {len(df)} rows")
    if not dfs:
        return pd.DataFrame(columns=CSV_FIELDS)
    merged = pd.concat(dfs, ignore_index=True)
    merged["canonical_smiles"] = merged["smiles"].apply(canonicalize_smiles)
    merged = merged.dropna(subset=["canonical_smiles"])
    merged = merged.drop_duplicates(subset=["canonical_smiles"])
    print(f"  Existing after dedup: {len(merged)}")
    return merged


def fetch_additional(target_total, existing_cids):
    """Stream from HuggingFace and collect new molecules until we reach target_total."""
    need = target_total - len(existing_cids)
    if need <= 0:
        print(f"  Already have {len(existing_cids)} >= {target_total}, skipping fetch")
        return []

    print(f"  Need {need} more molecules, streaming from HuggingFace...")
    files = list_hf_files()
    print(f"  Found {len(files)} files in {HF_SUBSET}")
    np.random.seed(SEED)
    np.random.shuffle(files)

    new_records = []
    total_scanned = 0

    pbar = tqdm(files, desc="Fetch phase7", unit="file")
    for fname in pbar:
        if len(new_records) >= need:
            break
        url = HF_RESOLVE.format(file=fname)
        try:
            buf, _ = http_get_range(url, 0, CHUNK_BYTES - 1)
        except RuntimeError:
            continue

        file_kept = 0
        try:
            for obj in ijson.items(io.BytesIO(buf), "item"):
                total_scanned += 1
                cid = obj.get("cid")
                if cid is not None and int(cid) in existing_cids:
                    continue

                mw = _f(obj.get("pubchem-molecular-weight"))
                if mw is None or not (MW_MIN <= mw <= MW_MAX):
                    continue

                formula = obj.get("pubchem-molecular-formula")
                if formula is None:
                    continue
                els = formula_elements(formula)
                if not els or not els.issubset(ELEMENTS):
                    continue

                homo = _f(obj.get("energy-alpha-homo"))
                lumo = _f(obj.get("energy-alpha-lumo"))
                gap = _f(obj.get("energy-alpha-gap"))
                if homo is None or lumo is None or gap is None or gap <= 0:
                    continue

                smiles = obj.get("pubchem-isomeric-smiles")
                if not smiles:
                    continue

                new_records.append({
                    "cid": int(cid) if cid else None,
                    "mw": mw, "formula": formula,
                    "smiles": smiles, "homo": homo, "lumo": lumo, "gap": gap,
                })
                if cid is not None:
                    existing_cids.add(int(cid))
                file_kept += 1

                if len(new_records) >= need:
                    break
        except Exception:
            pass

        pbar.set_postfix(new=len(new_records), target=need, file_hit=file_kept)

    print(f"  Fetched {len(new_records)} new molecules (scanned {total_scanned})")
    return new_records


def build_graphs(df):
    """Build ETKDG 3D PyG graphs with Gasteiger charges."""
    smiles_list = df["canonical_smiles"].tolist()
    targets = df[TARGET_COLS].values.astype(np.float32)
    return build_labeled_graphs(smiles_list, targets, use_charges=True)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=300000)
    parser.add_argument("--skip-graphs", action="store_true",
                        help="Only fetch CSV, skip graph generation (for large runs)")
    args = parser.parse_args()

    ensure_dirs(RAW_DIR, PHASE7_DIR)

    # Step 1: load existing + fetch additional
    print("=" * 60)
    print("Step 1: Merge existing data + fetch additional")
    print("=" * 60)

    existing = load_existing_data()
    existing_cids = set()
    if "cid" in existing.columns:
        existing_cids = set(existing["cid"].dropna().astype(int).tolist())

    new_records = fetch_additional(args.target, existing_cids)

    if new_records:
        df_new = pd.DataFrame(new_records)
        df_new["canonical_smiles"] = df_new["smiles"].apply(canonicalize_smiles)
        df_new = df_new.dropna(subset=["canonical_smiles"])
        df_all = pd.concat([existing, df_new], ignore_index=True)
    else:
        df_all = existing

    for col in ["homo", "lumo", "gap", "mw"]:
        df_all[col] = pd.to_numeric(df_all[col], errors="coerce")
    df_all = df_all.dropna(subset=["homo", "lumo", "gap", "canonical_smiles"])
    df_all = df_all[df_all["gap"] > 0]
    df_all = df_all.drop_duplicates(subset=["canonical_smiles"])
    df_all = df_all.reset_index(drop=True)

    tag = f"{len(df_all) // 1000}k"
    out_csv = RAW_DIR / f"phase7_chonsfcl_mw200_1000_{tag}.csv"
    df_all.to_csv(out_csv, index=False, encoding="utf-8")

    print(f"\n  Total: {len(df_all)} molecules")
    print(f"  MW range: {df_all['mw'].min():.1f} - {df_all['mw'].max():.1f}")
    print(f"  Saved to {out_csv}")

    if args.skip_graphs:
        print("\n  --skip-graphs: skipping graph generation")
        return

    # Step 2: build ETKDG 3D graphs
    print(f"\n{'=' * 60}")
    print("Step 2: Build ETKDG 3D graphs")
    print("=" * 60)

    if GRAPH_PATH.exists():
        data_list = torch.load(GRAPH_PATH, weights_only=False)
        if len(data_list) >= len(df_all) * 0.9:
            print(f"  Reusing cached graphs: {len(data_list)}")
            return
        print(f"  Cache has {len(data_list)} but need ~{len(df_all)}, rebuilding")

    data_list = build_graphs(df_all)
    torch.save(data_list, GRAPH_PATH)
    print(f"  Saved {len(data_list)} graphs to {GRAPH_PATH}")
    print(f"  File size: {GRAPH_PATH.stat().st_size / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
