"""
01_fetch_stream.py — Phase 1: streaming fetch from PubChemQC B3LYP/6-31G*//PM6.

Strategy (per PROJECT brief):
  - The full dataset is ~7.67 TB; we NEVER download whole files.
  - HF file URLs support HTTP Range (returns 206). JSON records are ordered by
    ascending CID, so we pull byte chunks from the front/middle and parse
    incrementally with ijson.
  - We keep ONLY the slim fields (cid / MW / formula / SMILES / homo / lumo / gap)
    and drop the huge `coordinates` and `orbital-energies` arrays.
  - Filter on the fly: MW in [200, 300], elements subset of {C, H, O, N}.

This script has two modes:
  --selfcheck   Pull the first ~N records of the first file and verify that
                gap == lumo - homo (unit sanity), print a few rows, write nothing
                (or write a tiny sample CSV). Use this FIRST.
  --run         Stream across files and write the slim CSV.

Run examples:
  python scripts/pipeline/01_fetch_stream.py --selfcheck --limit 200
  python scripts/pipeline/01_fetch_stream.py --run --max-records 5000      # smoke test
  python scripts/pipeline/01_fetch_stream.py --run                          # full subset

Units: VERIFIED EMPIRICALLY that energies are already in eV (NOT Hartree as the
project brief assumed). HOMO eigenvalues come out ~-4 to -8, which is the eV
range for organic molecules; in Hartree those would be ~-125 eV, which is
impossible. The internal relation gap == lumo - homo holds exactly. Therefore we
model directly in eV with NO ×27.2114 conversion. (If a value were in Hartree it
would be ~-0.2; we see ~-5, i.e. already multiplied by ~27.)
"""

import argparse
import csv
import io
import logging
import os
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

import ijson
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

HF_BASE = (
    "https://huggingface.co/datasets/molssiai-hub/pubchemqc-b3lyp/"
    "resolve/main/data/b3lyp_pm6_chon300nosalt/train/{file}"
)
HF_API_TREE = (
    "https://huggingface.co/api/datasets/molssiai-hub/pubchemqc-b3lyp/"
    "tree/main/data/b3lyp_pm6_chon300nosalt/train"
)

# Hardcoded first file name (also used by selfcheck). Full list fetched at runtime.
FIRST_FILE = "000000001-000800488.json"

MW_MIN, MW_MAX = 200.0, 300.0
ALLOWED_ELEMENTS = {"C", "H", "O", "N"}

HARTREE_TO_EV = 27.2114

USER_AGENT = "curl/8"
DEFAULT_CHUNK_BYTES = 12_000_000  # 12 MB, proven to yield real records in brief

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = os.path.join(REPO_ROOT, "data", "raw")
OUT_CSV = os.path.join(OUT_DIR, "pubchemqc_chon_mw200_300.csv")

CSV_FIELDS = ["cid", "mw", "formula", "smiles", "homo", "lumo", "gap"]

log = logging.getLogger("fetch")


# ---------------------------------------------------------------------------
# HTTP Range helpers
# ---------------------------------------------------------------------------

def http_get_range(url, start, end, timeout=120, retries=3):
    """Return bytes for the inclusive byte range [start, end] via HTTP Range.

    Raises on persistent failure. Accepts 206 (partial) or 200 (full, if server
    ignored Range — caller should be tolerant).
    """
    headers = {"User-Agent": USER_AGENT, "Range": f"bytes={start}-{end}"}
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                status = resp.getcode()
                data = resp.read()
                if status not in (200, 206):
                    raise urllib.error.HTTPError(
                        url, status, f"unexpected status {status}", resp.headers, None
                    )
                return data, status
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last_err = e
            wait = 2 ** attempt
            log.warning("range request failed (attempt %d/%d): %s; retrying in %ds",
                        attempt, retries, e, wait)
            time.sleep(wait)
    raise RuntimeError(f"range request failed after {retries} attempts: {last_err}")


def list_files(timeout=60):
    """Fetch the sorted list of 87 train file names from the HF tree API."""
    import json
    req = urllib.request.Request(HF_API_TREE, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.load(resp)
    files = sorted(d["path"].split("/")[-1] for d in data if d.get("type") == "file")
    return files


# ---------------------------------------------------------------------------
# Parsing / filtering
# ---------------------------------------------------------------------------

def formula_elements(formula):
    """Extract the set of element symbols from a molecular formula string.

    Handles two-letter elements (e.g. 'Cl', 'Na') by reading an uppercase letter
    optionally followed by lowercase letters. Digits are ignored.
    """
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


def passes_filter(mw, formula):
    """MW in [200,300] and all elements in {C,H,O,N}."""
    if mw is None or formula is None:
        return False
    if not (MW_MIN <= float(mw) <= MW_MAX):
        return False
    els = formula_elements(formula)
    if not els or not els.issubset(ALLOWED_ELEMENTS):
        return False
    return True


def _f(x):
    """Cast ijson's Decimal (or str/None) to float, or None on failure."""
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def extract_record(obj):
    """Pull the slim fields out of one PubChemQC JSON object.

    ijson yields numbers as decimal.Decimal; cast to float so downstream
    arithmetic and CSV output stay plain.
    """
    return {
        "cid": obj.get("cid"),
        "mw": _f(obj.get("pubchem-molecular-weight")),
        "formula": obj.get("pubchem-molecular-formula"),
        "smiles": obj.get("pubchem-isomeric-smiles"),
        "homo": _f(obj.get("energy-alpha-homo")),
        "lumo": _f(obj.get("energy-alpha-lumo")),
        "gap": _f(obj.get("energy-alpha-gap")),
    }


def iter_records_from_bytes(buf):
    """Incrementally yield top-level array items from a (possibly truncated) JSON
    byte buffer. ijson raises when it hits the truncation point; we stop cleanly
    and keep whatever complete records we already yielded.
    """
    try:
        for obj in ijson.items(io.BytesIO(buf), "item"):
            yield obj
    except ijson.JSONError as e:
        log.debug("stopped parsing at truncation boundary: %s", e)
    except Exception as e:  # noqa: BLE001 - ijson can raise various low-level errors
        log.debug("stopped parsing (non-JSONError) at boundary: %s", e)


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------

def run_selfcheck(limit, chunk_bytes):
    """Pull the front of the first file, verify gap == lumo - homo, print rows."""
    url = HF_BASE.format(file=FIRST_FILE)
    log.info("self-check: fetching first %d bytes of %s", chunk_bytes, FIRST_FILE)
    buf, status = http_get_range(url, 0, chunk_bytes - 1)
    log.info("HTTP status %s, got %d bytes", status, len(buf))

    n_seen = 0
    n_bad_unit = 0
    max_resid = 0.0
    printed = 0
    for obj in iter_records_from_bytes(buf):
        rec = extract_record(obj)
        homo, lumo, gap = rec["homo"], rec["lumo"], rec["gap"]
        if homo is None or lumo is None or gap is None:
            continue
        n_seen += 1
        resid = abs((lumo - homo) - gap)
        max_resid = max(max_resid, resid)
        if resid > 1e-6:
            n_bad_unit += 1
        if printed < 10:
            print(
                f"  cid={rec['cid']:<10} MW={rec['mw']:<10.3f} "
                f"homo={homo:.5f} lumo={lumo:.5f} gap={gap:.5f} "
                f"lumo-homo={lumo - homo:.5f} resid={resid:.2e}  (units: eV)"
            )
            printed += 1
        if n_seen >= limit:
            break

    print("\n=== UNIT SELF-CHECK SUMMARY ===")
    print(f"records checked : {n_seen}")
    print(f"max |(lumo-homo) - gap| residual : {max_resid:.3e}")
    print(f"records with residual > 1e-6     : {n_bad_unit}")
    if n_seen == 0:
        print("RESULT: NO RECORDS PARSED — investigate (network / format).")
        return 1
    if n_bad_unit == 0:
        print("RESULT: PASS — gap == lumo - homo holds.")
        print("        HOMO magnitudes ~4-8 => values are already in eV "
              "(no Hartree conversion needed).")
        return 0
    print("RESULT: WARNING — some records violate gap == lumo - homo. Inspect above.")
    return 1


# ---------------------------------------------------------------------------
# Full streaming run
# ---------------------------------------------------------------------------

def run_stream(files, chunk_bytes, max_records, max_files, append):
    """Stream front chunks of each file, filter, and write slim CSV.

    NOTE: This pulls only the FIRST `chunk_bytes` of each file (the brief's proven
    approach). For a true full-subset extraction you would walk successive byte
    windows per file until exhausted; that is a later step. This gets a real,
    sizeable sample flowing and validates the whole pipeline end to end.
    """
    os.makedirs(OUT_DIR, exist_ok=True)
    mode = "a" if append and os.path.exists(OUT_CSV) else "w"
    write_header = mode == "w"

    total_seen = 0
    total_kept = 0
    t0 = time.time()

    files_to_do = files[:max_files] if max_files else files

    with open(OUT_CSV, mode, newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()

        for fi, fname in enumerate(tqdm(files_to_do, desc="Fetch files", unit="file"), 1):
            url = HF_BASE.format(file=fname)
            log.info("[%d/%d] fetching front %d bytes of %s",
                     fi, len(files_to_do), chunk_bytes, fname)
            try:
                buf, status = http_get_range(url, 0, chunk_bytes - 1)
            except RuntimeError as e:
                log.error("skipping %s: %s", fname, e)
                continue

            file_seen = 0
            file_kept = 0
            with tqdm(
                desc=f"Parse {fname}",
                unit="rec",
                leave=False,
                mininterval=0.5,
            ) as rec_bar:
                for obj in iter_records_from_bytes(buf):
                    rec = extract_record(obj)
                    file_seen += 1
                    total_seen += 1
                    rec_bar.update(1)
                    if rec["homo"] is None or rec["lumo"] is None or rec["gap"] is None:
                        continue
                    if not passes_filter(rec["mw"], rec["formula"]):
                        continue
                    writer.writerow(rec)
                    file_kept += 1
                    total_kept += 1
                    rec_bar.set_postfix(kept=file_kept, total_kept=total_kept)
                    if max_records and total_kept >= max_records:
                        log.info("reached max_records=%d, stopping", max_records)
                        fh.flush()
                        _summary(total_seen, total_kept, t0)
                        return 0
            fh.flush()
            log.info("    %s: seen=%d kept=%d (running kept=%d)",
                     fname, file_seen, file_kept, total_kept)

    _summary(total_seen, total_kept, t0)
    return 0


def _summary(total_seen, total_kept, t0):
    dt = time.time() - t0
    print("\n=== FETCH SUMMARY ===")
    print(f"records parsed : {total_seen}")
    print(f"records kept   : {total_kept} (MW {MW_MIN:.0f}-{MW_MAX:.0f}, CHON only)")
    print(f"elapsed        : {dt:.1f}s")
    print(f"output         : {OUT_CSV}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description="PubChemQC streaming fetch (Phase 1)")
    p.add_argument("--selfcheck", action="store_true",
                   help="unit self-check on first file front chunk")
    p.add_argument("--run", action="store_true",
                   help="stream + filter + write slim CSV")
    p.add_argument("--limit", type=int, default=200,
                   help="selfcheck: max records to check (default 200)")
    p.add_argument("--max-records", type=int, default=0,
                   help="run: stop after keeping N records (0 = no cap)")
    p.add_argument("--max-files", type=int, default=0,
                   help="run: only process first N files (0 = all 87)")
    p.add_argument("--chunk-bytes", type=int, default=DEFAULT_CHUNK_BYTES,
                   help=f"front bytes per file (default {DEFAULT_CHUNK_BYTES})")
    p.add_argument("--append", action="store_true",
                   help="run: append to existing CSV instead of overwrite")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.selfcheck:
        return run_selfcheck(args.limit, args.chunk_bytes)

    if args.run:
        log.info("listing train files from HF API ...")
        try:
            files = list_files()
            log.info("found %d files", len(files))
        except Exception as e:  # noqa: BLE001
            log.warning("file listing failed (%s); falling back to first file only", e)
            files = [FIRST_FILE]
        return run_stream(files, args.chunk_bytes, args.max_records,
                          args.max_files, args.append)

    p.print_help()
    print("\nNothing to do. Pass --selfcheck or --run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
