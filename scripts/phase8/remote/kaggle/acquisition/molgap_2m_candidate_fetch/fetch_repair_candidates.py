"""Fetch a disjoint, coverage-targeted PubChemQC pool for 1M-v2 repair.

The rejected 1M continuation retained the validated expansion500K base but
appended a general in-domain half. This fetcher excludes every CID and
canonical SMILES in that complete rejected 1M table, then gathers a larger
candidate pool across predeclared sparse chemistry buckets. A later selector
uses only 500K rows from this pool; this command never builds graphs or trains.
"""
from __future__ import annotations

import argparse
import csv
import http.client
import io
import json
import os
import random
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import ijson
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors
from tqdm import tqdm

RAW_DIR = Path("/kaggle/working")
RESULTS_DIR = Path("/kaggle/working")


def canonicalize_smiles(smiles: object) -> str | None:
    mol = Chem.MolFromSmiles(str(smiles))
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True) if mol else None


def ensure_dirs(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

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
TRAIN_CSV = RAW_DIR / "phase8_expansion_1m.csv"
DESC_CACHE = RESULTS_DIR / "phase8" / "training_gap_descriptors.csv"
SPEC_JSON = RESULTS_DIR / "phase8" / "repair_1m_v2_sampling_spec.json"
OUT_CSV = RAW_DIR / "phase8_repair_v2_candidate_pool_600k.csv"
EXTRA_BUCKET_QUOTAS: dict[str, int] = {}

CSV_FIELDS = [
    "bucket", "cid", "mw", "formula", "smiles", "canonical_smiles",
    "homo", "lumo", "gap",
    "heavy_atoms", "ring_count", "aromatic_rings", "aromatic_atom_fraction",
    "rotatable_bonds", "conjugated_bonds", "fraction_csp3", "amide_bonds",
    "macrocycle", "bridgeheads", "has_s", "has_cl", "has_f",
    "n_n", "n_o", "n_s", "n_f", "n_cl", "heteroatom_fraction",
]

AMIDE = Chem.MolFromSmarts("[NX3][CX3](=[OX1])")


def http_get_range(url: str, start: int, end: int, timeout: int = 120, retries: int = 3) -> bytes:
    headers = {"User-Agent": USER_AGENT, "Range": f"bytes={start}-{end}"}
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.IncompleteRead) as e:
            last_err = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"range request failed after {retries} attempts: {last_err}")


def fetch_window(url: str, start: int, end: int) -> tuple[int, bytes | None]:
    """Fetch one range without letting a transient failure abort a whole scan."""
    try:
        return start, http_get_range(url, start, end)
    except RuntimeError:
        return start, None


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
        high_gap_buckets = {"high_gap_hetero", "high_gap_rigid"}
        if active_buckets <= high_gap_buckets and gap < 5.5:
            return None
        large_buckets = {
            "high_sp3_very_large", "macrocycle_very_large", "non_aromatic_very_large",
            "flexible_very_large", "multi_amide_very_large",
        }
        if active_buckets <= large_buckets and mw < 700:
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
    atomic_numbers = [atom.GetAtomicNum() for atom in atoms]
    n_n = atomic_numbers.count(7)
    n_o = atomic_numbers.count(8)
    n_f = atomic_numbers.count(9)
    n_s = atomic_numbers.count(16)
    n_cl = atomic_numbers.count(17)
    heteroatom_fraction = (n_n + n_o + n_f + n_s + n_cl) / heavy if heavy else 0.0
    fraction_csp3 = rdMolDescriptors.CalcFractionCSP3(mol)
    amide_bonds = len(mol.GetSubstructMatches(AMIDE))
    macrocycle = any(len(ring) >= 8 for ring in mol.GetRingInfo().AtomRings())
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
        "fraction_csp3": fraction_csp3,
        "amide_bonds": amide_bonds,
        "macrocycle": int(macrocycle),
        "bridgeheads": rdMolDescriptors.CalcNumBridgeheadAtoms(mol),
        "has_s": int("S" in elements),
        "has_cl": int("Cl" in elements),
        "has_f": int("F" in elements),
        "n_n": n_n,
        "n_o": n_o,
        "n_s": n_s,
        "n_f": n_f,
        "n_cl": n_cl,
        "heteroatom_fraction": heteroatom_fraction,
    }


def aromatic_edge(row: dict) -> bool:
    return row["aromatic_rings"] >= 5 or row["aromatic_atom_fraction"] >= 0.85


def bucket_matches(bucket_id: str, row: dict) -> bool:
    if bucket_id == "general_indomain":
        return True
    if bucket_id == "balanced_general":
        # Counterbalance the hard buckets with source-space support similar to
        # the validated expansion500K, not the high-Gap general top-up that
        # produced the rejected 1M continuation.
        return (
            3.2 <= row["gap"] < 5.5
            and 200 <= row["mw"] < 500
            and row["aromatic_rings"] <= 4
            and row["rotatable_bonds"] <= 7
        )
    gap = row["gap"]
    mw = row["mw"]
    ar = row["aromatic_rings"]
    af = row["aromatic_atom_fraction"]
    rot = row["rotatable_bonds"]
    has_s_or_cl = bool(row["has_s"] or row["has_cl"])
    sp3 = row["fraction_csp3"]
    amide = row["amide_bonds"]
    macrocycle = bool(row["macrocycle"])
    n_n = row["n_n"]
    n_o = row["n_o"]
    n_s = row["n_s"]
    n_f = row["n_f"]
    n_cl = row["n_cl"]
    heteroatom_fraction = row["heteroatom_fraction"]
    rings = row["ring_count"]
    bridgeheads = row["bridgeheads"]
    conjugated = row["conjugated_bonds"]
    hetero_nos = n_n + n_o + n_s
    hetero_acceptor = n_o + n_s + n_f + n_cl
    if bucket_id == "high_gap_hetero":
        return gap >= 5.5 and hetero_nos >= 2 and mw < 700
    if bucket_id == "high_gap_rigid":
        return gap >= 5.5 and rings >= 1 and rot <= 2
    if bucket_id == "small_hetero_dense":
        return 200 <= mw <= 300 and hetero_nos >= 3
    if bucket_id == "hetero_dense_midgap":
        return 3.2 <= gap <= 5.5 and heteroatom_fraction >= 0.25 and hetero_nos >= 3
    if bucket_id == "sulfur_rich":
        return n_s >= 2
    if bucket_id == "halogen_rich":
        return n_f + n_cl >= 3 and 2.5 < gap <= 5.5
    if bucket_id == "bridged_polycyclic":
        return bridgeheads >= 2 and rings >= 2
    if bucket_id == "fused_rigid":
        return rings >= 4 and rot <= 2 and 2 <= ar <= 4
    if bucket_id == "donor_acceptor_conjugated":
        return conjugated >= 8 and n_n >= 1 and hetero_acceptor >= 1 and gap <= 4.5
    if bucket_id == "conjugated_midgap":
        return conjugated >= 10 and 3.2 < gap <= 5.0 and ar <= 4
    if bucket_id == "high_sp3_very_large":
        return mw >= 700 and sp3 > 0.7
    if bucket_id == "macrocycle_very_large":
        return mw >= 700 and macrocycle
    if bucket_id == "non_aromatic_very_large":
        return mw >= 700 and ar == 0
    if bucket_id == "flexible_very_large":
        return mw >= 700 and rot >= 8
    if bucket_id == "low_mid_gap_flexible":
        return 2.5 < gap <= 4.0 and rot >= 8
    if bucket_id == "multi_amide_very_large":
        return mw >= 700 and amide >= 3
    if bucket_id == "high_sp3_non_aromatic":
        return sp3 > 0.7 and ar == 0
    if bucket_id == "residual_balanced":
        return (
            2.5 < gap <= 5.5 and 300 <= mw < 700 and rot <= 10
            and ar <= 4 and amide <= 3
        )
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


def write_json_atomic(path: Path, payload: dict) -> None:
    """Persist a status record without exposing a partially written JSON file."""
    ensure_dirs(path.parent)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def flush_checkpoint(fh, progress_json: Path | None, payload: dict) -> None:
    """Make CSV rows durable before advertising their count to an orchestrator."""
    fh.flush()
    os.fsync(fh.fileno())
    if progress_json:
        write_json_atomic(progress_json, payload)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", type=Path, default=SPEC_JSON)
    ap.add_argument("--train-csv", type=Path, default=TRAIN_CSV)
    ap.add_argument("--exclude-csv", type=Path, action="append", default=[],
                    help="Additional CSVs whose cid/canonical_smiles should be excluded")
    ap.add_argument("--out-csv", type=Path, default=OUT_CSV)
    ap.add_argument("--max-kept", type=int, default=600_000)
    ap.add_argument("--max-files", type=int, default=0)
    ap.add_argument("--file-shard-index", type=int, default=0,
                    help="Zero-based stable source-file shard index")
    ap.add_argument("--file-shard-count", type=int, default=1,
                    help="Number of disjoint stable source-file shards")
    ap.add_argument("--windows-per-file", type=int, default=4)
    ap.add_argument("--chunk-bytes", type=int, default=30_000_000)
    ap.add_argument("--download-workers", type=int, default=1,
                    help="concurrent HTTP range downloads per source file; 1 preserves serial behavior")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--include-buckets", nargs="*", default=None,
                    help="Restrict collection to these spec bucket IDs, preserving spec order")
    ap.add_argument("--general", action="store_true",
                    help="Collect any in-domain molecule instead of requiring a hard bucket match")
    ap.add_argument("--report-json", type=Path, default=None,
                    help="Optional path for scan/fill-rate summary")
    ap.add_argument("--progress-json", type=Path, default=None,
                    help="Atomically refreshed durable progress record for remote orchestration")
    ap.add_argument("--checkpoint-every", type=int, default=500,
                    help="Flush CSV and refresh --progress-json after this many newly collected rows")
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
    if args.download_workers < 1:
        raise ValueError("--download-workers must be at least 1")
    if args.file_shard_count < 1:
        raise ValueError("--file-shard-count must be at least 1")
    if not 0 <= args.file_shard_index < args.file_shard_count:
        raise ValueError("--file-shard-index must be in [0, --file-shard-count)")
    if args.checkpoint_every < 1:
        raise ValueError("--checkpoint-every must be at least 1")
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
    files = [
        info for position, info in enumerate(files)
        if position % args.file_shard_count == args.file_shard_index
    ]
    rng = random.Random(args.seed)
    rng.shuffle(files)
    if args.max_files:
        files = files[:args.max_files]
    print(
        f"Scanning {len(files)} HF files from shard "
        f"{args.file_shard_index}/{args.file_shard_count}; "
        f"windows/file={args.windows_per_file}, chunk={args.chunk_bytes:,} bytes"
    )

    active_bucket_set = set(buckets)
    total_seen = 0
    total_parsed = 0
    total_kept = resumed_rows
    files_scanned = 0
    t0 = time.time()

    def progress_payload(state: str) -> dict:
        elapsed_s = time.time() - t0
        return {
            "state": state,
            "out_csv": str(args.out_csv),
            "include_buckets": buckets,
            "quotas": quotas,
            "counts": counts,
            "resumed_rows": resumed_rows,
            "new_rows": total_kept - resumed_rows,
            "total_rows": total_kept,
            "files_scanned": files_scanned,
            "source_file_shard": {
                "index": args.file_shard_index,
                "count": args.file_shard_count,
            },
            "parsed_objects": total_parsed,
            "filter_passing_candidates_seen": total_seen,
            "elapsed_s": elapsed_s,
            "kept_rows_per_s": (total_kept - resumed_rows) / elapsed_s if elapsed_s else 0.0,
            "checkpoint_every": args.checkpoint_every,
        }

    mode = "a" if args.resume and args.out_csv.exists() else "w"
    active_fh = None
    try:
        with args.out_csv.open(mode, newline="", encoding="utf-8") as fh:
            active_fh = fh
            writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
            if mode == "w":
                writer.writeheader()
            flush_checkpoint(fh, args.progress_json, progress_payload("running"))
            pbar = tqdm(files, desc="HF files", unit="file")
            for file_info in pbar:
                if total_kept >= args.max_kept or all(counts[b] >= quotas[b] for b in buckets):
                    break
                files_scanned += 1
                fname = str(file_info["name"])
                size = int(file_info["size"])
                max_start = max(0, size - args.chunk_bytes)
                starts = [0]
                for _ in range(max(0, args.windows_per_file - 1)):
                    starts.append(rng.randint(0, max_start) if max_start else 0)
                url = HF_RESOLVE.format(file=fname)
                windows = [
                    (start, min(size - 1, start + args.chunk_bytes - 1) if size else start + args.chunk_bytes - 1)
                    for start in starts
                ]
                if args.download_workers == 1:
                    fetched = [fetch_window(url, start, end) for start, end in windows]
                else:
                    fetched_map: dict[int, bytes | None] = {}
                    with ThreadPoolExecutor(max_workers=min(args.download_workers, len(windows))) as pool:
                        futures = [pool.submit(fetch_window, url, start, end) for start, end in windows]
                        for future in as_completed(futures):
                            start, buf = future.result()
                            fetched_map[start] = buf
                    fetched = [(start, fetched_map.get(start)) for start, _ in windows]
                for start, buf in fetched:
                    if total_kept >= args.max_kept or all(counts[b] >= quotas[b] for b in buckets):
                        break
                    if buf is None:
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
                        if (total_kept - resumed_rows) % args.checkpoint_every == 0:
                            flush_checkpoint(fh, args.progress_json, progress_payload("running"))
                        if total_kept >= args.max_kept:
                            break
                pbar.set_postfix(kept=total_kept, parsed=total_parsed, **counts)
            flush_checkpoint(fh, args.progress_json, progress_payload("complete"))
    except KeyboardInterrupt:
        if active_fh and not active_fh.closed:
            flush_checkpoint(active_fh, args.progress_json, progress_payload("interrupted"))
        if args.progress_json:
            write_json_atomic(args.progress_json, progress_payload("interrupted"))
        raise
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
            "files_scanned": files_scanned,
            "source_file_shard": {
                "index": args.file_shard_index,
                "count": args.file_shard_count,
            },
            "windows_per_file": args.windows_per_file,
            "chunk_bytes": args.chunk_bytes,
            "download_workers": args.download_workers,
            "parsed_objects": total_parsed,
            "filter_passing_candidates_seen": total_seen,
            "elapsed_s": elapsed_s,
            "kept_rows_per_s": (total_kept - resumed_rows) / elapsed_s if elapsed_s else 0.0,
            "parsed_objects_per_s": total_parsed / elapsed_s if elapsed_s else 0.0,
        }
        write_json_atomic(args.report_json, report)
        print(f"Saved report: {args.report_json}")


if __name__ == "__main__":
    main()
