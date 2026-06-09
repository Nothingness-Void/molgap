"""Phase 6: Fetch MW 500-1000 molecules from PubChemQC full b3lyp_pm6 dataset."""
from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import ijson
from tqdm import tqdm

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from molgap.utils import RAW_DIR, ensure_dirs

log = logging.getLogger("phase6")

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
MW_MIN, MW_MAX = 500, 1000
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


def fetch_data(max_records, chunk_bytes):
    tag = f"{max_records // 1000}k"
    out_path = RAW_DIR / f"phase6_chonsfcl_mw{MW_MIN}_{MW_MAX}_{tag}.csv"

    if out_path.exists():
        import pandas as pd
        df = pd.read_csv(out_path)
        if len(df) >= max_records * 0.9:
            print(f"Reusing existing {out_path} ({len(df)} rows)")
            return out_path
        print(f"Existing file has only {len(df)} rows, re-fetching")

    print(f"Fetching CHONSFCl MW {MW_MIN}-{MW_MAX}, target {max_records} records...")
    print(f"  chunk size: {chunk_bytes // 1_000_000}MB per file")
    ensure_dirs(out_path.parent)

    files = list_hf_files()
    print(f"Found {len(files)} files in {HF_SUBSET}")

    total_kept = 0
    total_scanned = 0

    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()

        pbar = tqdm(files, desc="Fetch phase6", unit="file")
        for fname in pbar:
            url = HF_RESOLVE.format(file=fname)
            try:
                buf, _ = http_get_range(url, 0, chunk_bytes - 1)
            except RuntimeError as e:
                log.warning(f"Skip {fname}: {e}")
                continue

            file_kept = 0
            try:
                for obj in ijson.items(io.BytesIO(buf), "item"):
                    total_scanned += 1
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
                    if homo is None or lumo is None or gap is None:
                        continue

                    smiles = obj.get("pubchem-isomeric-smiles")
                    if not smiles:
                        continue

                    writer.writerow({
                        "cid": obj.get("cid"), "mw": mw, "formula": formula,
                        "smiles": smiles, "homo": homo, "lumo": lumo, "gap": gap,
                    })
                    total_kept += 1
                    file_kept += 1

                    if total_kept >= max_records:
                        pbar.close()
                        print(f"\nReached {max_records} records")
                        print(f"Scanned {total_scanned} molecules total")
                        return out_path
            except Exception:
                pass

            pbar.set_postfix(kept=total_kept, file_hit=file_kept)

    print(f"\nFetched {total_kept} records (wanted {max_records})")
    print(f"Scanned {total_scanned} molecules total")
    return out_path


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")

    p = argparse.ArgumentParser(description="Phase 6: Fetch MW 500-1000 molecules")
    p.add_argument("--max-records", type=int, default=15000,
                   help="Target number of molecules (default: 15000)")
    p.add_argument("--chunk-bytes", type=int, default=20_000_000,
                   help="Bytes to download per file (default: 20MB)")
    args = p.parse_args()

    out_path = fetch_data(args.max_records, args.chunk_bytes)
    print(f"\nOutput: {out_path}")

    import pandas as pd
    df = pd.read_csv(out_path)
    print(f"\nSummary:")
    print(f"  Total molecules: {len(df)}")
    print(f"  MW range: {df['mw'].min():.1f} - {df['mw'].max():.1f}")
    print(f"  MW median: {df['mw'].median():.1f}")
    for q in [0.25, 0.5, 0.75, 0.9]:
        print(f"  MW {q*100:.0f}%: {df['mw'].quantile(q):.1f}")


if __name__ == "__main__":
    raise SystemExit(main())
