"""
Phase 8.2: targeted PubChemQC top-up fetch.

Uses results/phase8/sampling_spec.json to fill priority buckets with molecules
that are absent from the Phase 7 300k set. This writes a slim CSV only; graph
building is a later P8.3 step after we inspect availability and hold out hard
eval slices.

Smoke test:
  .venv\\Scripts\\python.exe scripts/phase8/archive/legacy/data_coverage/fetch_targeted_topup.py --max-kept 200 --max-files 3 --windows-per-file 1 --out-csv data/raw/phase8_targeted_topup_smoke.csv --overwrite

Rare-first scan:
  .venv\\Scripts\\python.exe scripts/phase8/archive/legacy/data_coverage/fetch_targeted_topup.py --include-buckets very_low_gap low_gap_aromatic_edge --max-kept 5000 --max-files 40 --windows-per-file 4 --out-csv data/raw/phase8_targeted_topup_rare_probe.csv --overwrite

Full run shape:
  .venv\\Scripts\\python.exe scripts/phase8/archive/legacy/data_coverage/fetch_targeted_topup.py --max-kept 200000 --windows-per-file 8 --chunk-bytes 30000000 --out-csv data/raw/phase8_targeted_topup_200k.csv --overwrite
  .venv\\Scripts\\python.exe scripts/phase8/archive/legacy/data_coverage/fetch_targeted_topup.py --max-kept 200000 --windows-per-file 8 --chunk-bytes 30000000 --out-csv data/raw/phase8_targeted_topup_200k.csv --resume
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import random
import time
import urllib.error
import urllib.request
from pathlib import Path

import ijson
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors
from tqdm import tqdm

from molgap.constants import RAW_DIR, RESULTS_DIR
from molgap.utils import canonicalize_smiles, ensure_dirs

RDLogger.DisableLog("rdApp.*")

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
ALLOWED_ELEMENTS = {"C", "H", "N", "O", "S", "F", "Cl"}
TRAIN_CSV = RAW_DIR / "phase7_chonsfcl_mw200_1000_300k.csv"
DESC_CACHE = RESULTS_DIR / "phase8" / "training_gap_descriptors.csv"
SPEC_JSON = RESULTS_DIR / "phase8" / "sampling_spec.json"
OUT_CSV = RAW_DIR / "phase8_targeted_topup_200k.csv"
EXTRA_BUCKET_QUOTAS = {
    # Probe-only buckets for the expansion500k residual tail. These are not part
    # of the original Phase 8 sampling spec, but keep the same CLI/fetch path.
    "low_gap_general": 12_000,       # 2.5 <= gap < 3.2, no aromatic-edge gate
    "very_large_tail": 8_000,        # MW >= 800 with low-gap or flexible signal
}

CSV_FIELDS = [
    "bucket", "cid", "mw", "formula", "smiles", "canonical_smiles",
    "homo", "lumo", "gap",
    "heavy_atoms", "ring_count", "aromatic_rings", "aromatic_atom_fraction",
    "rotatable_bonds", "conjugated_bonds", "has_s", "has_cl", "has_f",
]


def http_get_range(url: str, start: int, end: int, timeout: int = 120, retries: int = 3) -> bytes:
    headers = {"User-Agent": USER_AGENT, "Range": f"bytes={start}-{end}"}
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last_err = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"range request failed after {retries} attempts: {last_err}")


def list_hf_files() -> list[dict[str, int | str]]:
    req = urllib.request.Request(HF_API_TREE, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.load(resp)
    files = []
    for item in data:
        if item.get("type") != "file":
            continue
        path = item["path"]
        files.append({
            "name": path.split("/")[-1],
            "size": int(item.get("size") or item.get("lfs", {}).get("size") or 0),
        })
    return sorted(files, key=lambda x: str(x["name"]))


def formula_elements(formula: object) -> set[str]:
    if not isinstance(formula, str):
        return set()
    elements: set[str] = set()
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


def as_float(x):
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def iter_front_objects(buf: bytes):
    try:
        yield from ijson.items(io.BytesIO(buf), "item")
    except Exception:
        return


def iter_embedded_objects(buf: bytes):
    """Yield complete JSON objects from an arbitrary byte range.

    This is intentionally tolerant: ranges may start/end in the middle of a
    record. Incomplete objects are ignored.
    """
    text = buf.decode("utf-8", errors="ignore")
    depth = 0
    start = None
    in_str = False
    escape = False
    for i, ch in enumerate(text):
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth <= 0:
                continue
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    yield json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    pass
                start = None


def candidate_from_obj(obj: dict, active_buckets: set[str] | None = None) -> dict | None:
    cid = obj.get("cid")
    mw = as_float(obj.get("pubchem-molecular-weight"))
    homo = as_float(obj.get("energy-alpha-homo"))
    lumo = as_float(obj.get("energy-alpha-lumo"))
    gap = as_float(obj.get("energy-alpha-gap"))
    formula = obj.get("pubchem-molecular-formula")
    smiles = obj.get("pubchem-isomeric-smiles")
    if mw is None or homo is None or lumo is None or gap is None:
        return None
    if not (200 <= mw <= 1000) or gap <= 0 or not smiles:
        return None

    # Avoid expensive RDKit parsing when a bucket subset has a cheap label/size
    # precondition. This matters for rare low-gap scans, where >98% of otherwise
    # valid molecules can be skipped from labels alone.
    if active_buckets:
        if active_buckets <= {"very_low_gap"} and gap >= 2.5:
            return None
        low_gap_buckets = {"very_low_gap", "low_gap_aromatic_edge", "low_gap_general"}
        if active_buckets <= low_gap_buckets and gap >= 3.2:
            return None
        if active_buckets <= {"large_aromatic_edge", "very_large_general", "large_mw_500_700"} and mw < 500:
            return None

    elements = formula_elements(formula)
    if not elements or not elements.issubset(ALLOWED_ELEMENTS):
        return None

    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    can = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    atoms = list(mol.GetAtoms())
    bonds = list(mol.GetBonds())
    heavy = mol.GetNumHeavyAtoms()
    aromatic_atoms = sum(1 for atom in atoms if atom.GetIsAromatic())
    aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
    aromatic_fraction = aromatic_atoms / heavy if heavy else 0.0
    return {
        "cid": int(cid) if cid is not None else "",
        "mw": mw,
        "formula": formula,
        "smiles": smiles,
        "canonical_smiles": can,
        "homo": homo,
        "lumo": lumo,
        "gap": gap,
        "heavy_atoms": heavy,
        "ring_count": rdMolDescriptors.CalcNumRings(mol),
        "aromatic_rings": aromatic_rings,
        "aromatic_atom_fraction": aromatic_fraction,
        "rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "conjugated_bonds": sum(1 for bond in bonds if bond.GetIsConjugated()),
        "has_s": int("S" in elements),
        "has_cl": int("Cl" in elements),
        "has_f": int("F" in elements),
    }


def aromatic_edge(row: dict) -> bool:
    return row["aromatic_rings"] >= 5 or row["aromatic_atom_fraction"] >= 0.85


def bucket_matches(bucket_id: str, row: dict) -> bool:
    if bucket_id == "general_indomain":
        return True
    gap = row["gap"]
    mw = row["mw"]
    ar = row["aromatic_rings"]
    af = row["aromatic_atom_fraction"]
    rot = row["rotatable_bonds"]
    has_s_or_cl = bool(row["has_s"] or row["has_cl"])
    if bucket_id == "very_low_gap":
        return gap < 2.5
    if bucket_id == "low_gap_aromatic_edge":
        return 2.5 <= gap < 3.2 and aromatic_edge(row)
    if bucket_id == "low_gap_general":
        return 2.5 <= gap < 3.2
    if bucket_id == "large_aromatic_edge":
        return mw >= 500 and aromatic_edge(row)
    if bucket_id == "very_large_general":
        return mw >= 700
    if bucket_id == "s_or_cl_hard":
        return has_s_or_cl and (gap < 3.5 or ar >= 4 or af >= 0.70)
    if bucket_id == "aromatic_edge_general":
        return gap >= 3.2 and aromatic_edge(row)
    if bucket_id == "flexible_hard":
        return rot >= 8 and (gap < 3.5 or ar >= 4)
    if bucket_id == "large_mw_500_700":
        return 500 <= mw < 700
    if bucket_id == "very_large_tail":
        return mw >= 800 and (gap < 3.5 or rot >= 6)
    return False


def assign_bucket(row: dict, buckets: list[str], quotas: dict[str, int], counts: dict[str, int]) -> str | None:
    for bucket in buckets:
        if counts[bucket] >= quotas[bucket]:
            continue
        if bucket_matches(bucket, row):
            return bucket
    return None


def _load_ids_from_csv(path: Path) -> tuple[set[int], set[str]]:
    df = pd.read_csv(path, usecols=lambda c: c in {"cid", "smiles", "canonical_smiles"})
    cids = set(pd.to_numeric(df["cid"], errors="coerce").dropna().astype(int).tolist())
    if "canonical_smiles" in df.columns:
        smiles = set(df["canonical_smiles"].dropna().astype(str).tolist())
    elif path == TRAIN_CSV and DESC_CACHE.exists():
        desc = pd.read_csv(DESC_CACHE, usecols=["canonical_smiles"])
        smiles = set(desc["canonical_smiles"].dropna().astype(str).tolist())
    else:
        smiles = set()
        for smi in tqdm(df["smiles"].astype(str), desc="Canonicalize existing", unit="mol"):
            can = canonicalize_smiles(smi)
            if can:
                smiles.add(can)
    return cids, smiles


def load_existing_ids(train_csv: Path, extra_csvs: list[Path]) -> tuple[set[int], set[str]]:
    cids, smiles = _load_ids_from_csv(train_csv)
    for path in extra_csvs:
        if not path.exists():
            continue
        extra_cids, extra_smiles = _load_ids_from_csv(path)
        cids.update(extra_cids)
        smiles.update(extra_smiles)
    return cids, smiles


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", type=Path, default=SPEC_JSON)
    ap.add_argument("--train-csv", type=Path, default=TRAIN_CSV)
    ap.add_argument("--exclude-csv", type=Path, action="append", default=[],
                    help="Additional CSVs whose cid/canonical_smiles should be excluded")
    ap.add_argument("--out-csv", type=Path, default=OUT_CSV)
    ap.add_argument("--max-kept", type=int, default=200_000)
    ap.add_argument("--max-files", type=int, default=0)
    ap.add_argument("--windows-per-file", type=int, default=4)
    ap.add_argument("--chunk-bytes", type=int, default=30_000_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--include-buckets", nargs="*", default=None,
                    help="Restrict collection to these spec bucket IDs, preserving spec order")
    ap.add_argument("--general", action="store_true",
                    help="Collect any in-domain molecule instead of requiring a hard bucket match")
    ap.add_argument("--report-json", type=Path, default=None,
                    help="Optional path for scan/fill-rate summary")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--resume", action="store_true", help="append to an existing partial output CSV")
    args = ap.parse_args()

    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    raw_quotas = {b["id"]: int(b["quota"]) for b in spec["priority_buckets"]}
    buckets = [b["id"] for b in spec["priority_buckets"]]
    raw_quotas.update(EXTRA_BUCKET_QUOTAS)
    if args.general:
        buckets = ["general_indomain"]
        raw_quotas = {"general_indomain": args.max_kept}
    elif args.include_buckets:
        requested = set(args.include_buckets)
        known = set(buckets).union(EXTRA_BUCKET_QUOTAS)
        unknown = requested - known
        if unknown:
            raise ValueError(f"Unknown bucket(s): {sorted(unknown)}")
        buckets = [b for b in [*buckets, *EXTRA_BUCKET_QUOTAS] if b in requested]
        raw_quotas = {b: raw_quotas[b] for b in buckets}
    raw_total = sum(raw_quotas.values())
    if args.include_buckets or args.max_kept < raw_total:
        quotas = {
            b: max(1, int(round(raw_quotas[b] * args.max_kept / raw_total)))
            for b in buckets
        }
        # Keep the scaled quotas summing exactly to max_kept.
        drift = args.max_kept - sum(quotas.values())
        quotas[buckets[-1]] += drift
    else:
        quotas = raw_quotas

    if args.overwrite and args.resume:
        raise ValueError("Use only one of --overwrite or --resume")
    if args.out_csv.exists() and not (args.overwrite or args.resume):
        raise FileExistsError(f"{args.out_csv} exists; pass --overwrite or --resume")
    ensure_dirs(args.out_csv.parent)

    print("Loading Phase 7 exclusions...")
    existing_cids, existing_smiles = load_existing_ids(args.train_csv, args.exclude_csv)
    seen_cids = set(existing_cids)
    seen_smiles = set(existing_smiles)
    print(f"Excluding {len(existing_cids):,} CIDs and {len(existing_smiles):,} canonical SMILES")
    counts = {b: 0 for b in buckets}
    resumed_rows = 0
    if args.resume and args.out_csv.exists():
        prev = pd.read_csv(args.out_csv)
        resumed_rows = len(prev)
        for bucket, n in prev["bucket"].value_counts().items():
            if bucket in counts:
                counts[bucket] = int(n)
        if "cid" in prev:
            for cid in pd.to_numeric(prev["cid"], errors="coerce").dropna().astype(int):
                seen_cids.add(int(cid))
        if "canonical_smiles" in prev:
            seen_smiles.update(prev["canonical_smiles"].dropna().astype(str).tolist())
        print(f"Resuming {args.out_csv}: {resumed_rows:,} rows already written")

    files = list_hf_files()
    rng = random.Random(args.seed)
    rng.shuffle(files)
    if args.max_files:
        files = files[:args.max_files]
    print(f"Scanning {len(files)} HF files; windows/file={args.windows_per_file}, chunk={args.chunk_bytes:,} bytes")

    active_bucket_set = set(buckets)
    total_seen = 0
    total_parsed = 0
    total_kept = resumed_rows
    t0 = time.time()
    mode = "a" if args.resume and args.out_csv.exists() else "w"
    with args.out_csv.open(mode, newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        if mode == "w":
            writer.writeheader()
        pbar = tqdm(files, desc="HF files", unit="file")
        for file_info in pbar:
            if total_kept >= args.max_kept or all(counts[b] >= quotas[b] for b in buckets):
                break
            fname = str(file_info["name"])
            size = int(file_info["size"])
            max_start = max(0, size - args.chunk_bytes)
            starts = [0]
            for _ in range(max(0, args.windows_per_file - 1)):
                starts.append(rng.randint(0, max_start) if max_start else 0)
            url = HF_RESOLVE.format(file=fname)
            for start in starts:
                if total_kept >= args.max_kept or all(counts[b] >= quotas[b] for b in buckets):
                    break
                end = min(size - 1, start + args.chunk_bytes - 1) if size else start + args.chunk_bytes - 1
                try:
                    buf = http_get_range(url, start, end)
                except RuntimeError:
                    continue
                iterator = iter_front_objects(buf) if start == 0 else iter_embedded_objects(buf)
                for obj in iterator:
                    total_parsed += 1
                    row = candidate_from_obj(obj, active_bucket_set)
                    if row is None:
                        continue
                    total_seen += 1
                    cid = row["cid"]
                    can = row["canonical_smiles"]
                    if cid != "" and int(cid) in seen_cids:
                        continue
                    if can in seen_smiles:
                        continue
                    bucket = assign_bucket(row, buckets, quotas, counts)
                    if bucket is None:
                        continue
                    seen_smiles.add(can)
                    if cid != "":
                        seen_cids.add(int(cid))
                    row["bucket"] = bucket
                    writer.writerow({k: row.get(k, "") for k in CSV_FIELDS})
                    counts[bucket] += 1
                    total_kept += 1
                    if total_kept >= args.max_kept:
                        break
            pbar.set_postfix(kept=total_kept, parsed=total_parsed, **counts)
    elapsed_s = time.time() - t0
    print(f"\nSaved {total_kept:,} candidates to {args.out_csv}")
    print("Bucket counts:")
    for b in buckets:
        print(f"  {b:<26s} {counts[b]:>7,} / {quotas[b]:,}")
    print(f"Parsed objects: {total_parsed:,}; filter-passing candidates seen: {total_seen:,}")
    print(f"Elapsed: {elapsed_s:.1f}s; kept rate: {(total_kept - resumed_rows) / elapsed_s if elapsed_s else 0:.2f} rows/s")

    if args.report_json:
        ensure_dirs(args.report_json.parent)
        report = {
            "out_csv": str(args.out_csv),
            "include_buckets": buckets,
            "quotas": quotas,
            "counts": counts,
            "resumed_rows": resumed_rows,
            "new_rows": total_kept - resumed_rows,
            "total_rows": total_kept,
            "files_scanned": len(files),
            "windows_per_file": args.windows_per_file,
            "chunk_bytes": args.chunk_bytes,
            "parsed_objects": total_parsed,
            "filter_passing_candidates_seen": total_seen,
            "elapsed_s": elapsed_s,
            "kept_rows_per_s": (total_kept - resumed_rows) / elapsed_s if elapsed_s else 0.0,
            "parsed_objects_per_s": total_parsed / elapsed_s if elapsed_s else 0.0,
        }
        args.report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Saved report: {args.report_json}")


if __name__ == "__main__":
    main()
