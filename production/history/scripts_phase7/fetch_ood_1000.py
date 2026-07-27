"""
Phase 7: fetch 1000 OOD molecules from PubChemQC (HuggingFace) for the formal
2D vs 3D vs Hybrid comparison. Excludes all training CIDs (true unseen set),
MW 200-1000, CHONSFCl only.

Saves results/phase7/ood_1000/ood_molecules_1000.csv (smiles, cid, mw, formula,
homo, lumo, gap). The comparison script reads this file later.

Usage:
  .venv\\Scripts\\python.exe scripts/phase7/fetch_ood_1000.py
"""
from __future__ import annotations

import io
import json
import re
import time
import urllib.request
import warnings

import ijson
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from molgap.constants import RAW_DIR, RESULTS_DIR, SEED
from molgap.utils import ensure_dirs

OUT_DIR = RESULTS_DIR / "phase7" / "ood_1000"
N_TARGET = 1000
PER_FILE = 30  # molecules sampled per HF chunk (diversity across files)

HF_BASE = (
    "https://huggingface.co/datasets/molssiai-hub/pubchemqc-b3lyp/"
    "resolve/main/data/b3lyp_pm6/train/{file}"
)
HF_API_TREE = (
    "https://huggingface.co/api/datasets/molssiai-hub/pubchemqc-b3lyp/"
    "tree/main/data/b3lyp_pm6/train"
)
USER_AGENT = "curl/8"
ELEMENTS = {"C", "H", "O", "N", "S", "F", "Cl"}
MW_MIN, MW_MAX = 200, 1000
SCAN_CAP = 8000  # max records to stream-scan per file before giving up


def list_files():
    req = urllib.request.Request(HF_API_TREE, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.load(resp)
    return sorted(d["path"].split("/")[-1] for d in data if d.get("type") == "file")


def formula_elements(formula):
    return set(re.findall(r"[A-Z][a-z]?", formula))


def stream_ood(filename, n_want, train_cids, seen_cids):
    """Stream-parse a 6GB HF file from the top, skip training-set molecules
    (they fill the front of each file), collect up to n_want OOD molecules,
    then close the connection. OOD are abundant after the per-file quota."""
    url = HF_BASE.format(file=filename)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    collected = []
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                scanned = 0
                for obj in ijson.items(resp, "item"):
                    scanned += 1
                    if scanned > SCAN_CAP:
                        break
                    cid = obj.get("cid")
                    mw_raw = obj.get("pubchem-molecular-weight")
                    formula = obj.get("pubchem-molecular-formula")
                    smiles = obj.get("pubchem-isomeric-smiles")
                    homo = obj.get("energy-alpha-homo")
                    lumo = obj.get("energy-alpha-lumo")
                    gap = obj.get("energy-alpha-gap")
                    if not all(v is not None for v in
                               [cid, mw_raw, formula, smiles, homo, lumo, gap]):
                        continue
                    cid = int(cid)
                    if cid in train_cids or cid in seen_cids:
                        continue
                    mw = float(mw_raw)
                    if not (MW_MIN <= mw <= MW_MAX):
                        continue
                    els = formula_elements(formula)
                    if not els or not els.issubset(ELEMENTS):
                        continue
                    collected.append({
                        "cid": cid, "mw": mw, "formula": formula,
                        "smiles": str(smiles),
                        "homo": float(homo), "lumo": float(lumo), "gap": float(gap),
                    })
                    seen_cids.add(cid)
                    if len(collected) >= n_want:
                        break
            return collected
        except Exception:
            time.sleep(2 ** attempt)
    return collected


def load_training_cids():
    cids = set()
    for p in RAW_DIR.glob("phase*.csv"):
        try:
            df = pd.read_csv(p, usecols=["cid"])
            cids.update(df["cid"].tolist())
        except Exception:
            pass
    print(f"  Training CIDs to exclude: {len(cids)}")
    return cids


def main():
    ensure_dirs(OUT_DIR)
    np.random.seed(SEED)
    print(f"=== Fetch {N_TARGET} OOD molecules (MW 200-1000, CHONSFCl) ===\n")

    train_cids = load_training_cids()

    print("  Fetching file list from HuggingFace...")
    files = list_files()
    print(f"  Found {len(files)} files in b3lyp_pm6")

    np.random.shuffle(files)
    all_records = []
    seen_cids = set()
    for filename in files:
        if len(all_records) >= N_TARGET:
            break
        want = min(PER_FILE, N_TARGET - len(all_records))
        print(f"  Streaming {filename} (want {want})...", end=" ", flush=True)
        new = stream_ood(filename, want, train_cids, seen_cids)
        all_records.extend(new)
        print(f"kept {len(new)} (total: {len(all_records)})")

    if len(all_records) > N_TARGET:
        all_records = all_records[:N_TARGET]

    ood_df = pd.DataFrame(all_records)
    print(f"\n  Collected {len(ood_df)} OOD molecules")

    bins = [(200, 300), (300, 500), (500, 700), (700, 1000)]
    print("  MW distribution:")
    for lo, hi in bins:
        n = ((ood_df["mw"] >= lo) & (ood_df["mw"] < hi)).sum()
        print(f"    MW {lo}-{hi}: {n}")

    out_csv = OUT_DIR / "ood_molecules_1000.csv"
    ood_df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"\n  Saved to {out_csv}")


if __name__ == "__main__":
    main()
