"""Stream deduplicated PubChemQC candidates into resumable Parquet parts."""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path

import pandas as pd

from molgap.constants import RAW_DIR, RESULTS_DIR
from molgap.pubchemqc import (
    HF_CONFIG, HF_RESOLVE, PubChemQCFilter, iter_json_objects, list_hf_files,
    pubchemqc_record, read_http_range, sha256_file,
)
from molgap.utils import ensure_dirs


RESULTS = RESULTS_DIR / "phase8" / "archive" / "archive-r02-pubchemqc-router"
DEFAULT_OUT = RAW_DIR / "archive-r02-router-candidates"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--manifest-out", type=Path, default=RESULTS / "candidate_pool_manifest.json")
    parser.add_argument("--max-kept", type=int, default=1_000_000)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--windows-per-file", type=int, default=8)
    parser.add_argument("--chunk-bytes", type=int, default=30_000_000)
    parser.add_argument("--part-size", type=int, default=25_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_column(path: Path, column: str) -> set:
    return set(pd.read_parquet(path, columns=[column])[column].dropna())


def main() -> None:
    args = parse_args()
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")
    manifest_path = RESULTS / "experiment_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("run 01_build_exclusion_set.py first")
    ensure_dirs(args.out_dir)
    parts = sorted(args.out_dir.glob("part-*.parquet"))
    if parts and not (args.resume or args.overwrite):
        raise FileExistsError(f"candidate parts already exist in {args.out_dir}")
    if args.overwrite:
        for path in parts + [args.out_dir / "scan_state.json"]:
            path.unlink(missing_ok=True)
        parts = []

    excluded_cids = load_column(RESULTS / "excluded_cids.parquet", "cid")
    excluded_smiles = load_column(RESULTS / "excluded_smiles.parquet", "canonical_smiles")
    excluded_inchikeys = load_column(RESULTS / "excluded_inchikeys.parquet", "inchikey")
    seen_cids, seen_smiles, seen_inchikeys = set(), set(), set()
    kept = 0
    for part in parts:
        frame = pd.read_parquet(part, columns=["cid", "canonical_smiles", "inchikey"])
        seen_cids.update(frame["cid"])
        seen_smiles.update(frame["canonical_smiles"])
        seen_inchikeys.update(frame["inchikey"])
        kept += len(frame)

    state_path = args.out_dir / "scan_state.json"
    state = json.loads(state_path.read_text()) if args.resume and state_path.exists() else {}
    completed = set(state.get("completed_windows", []))
    rejected = Counter(state.get("rejected", {}))
    scanned = int(state.get("scanned", 0))
    downloaded = int(state.get("downloaded_bytes", 0))
    rng = random.Random(args.seed)
    files = list_hf_files(HF_CONFIG)
    rng.shuffle(files)
    if args.max_files is not None:
        files = files[:args.max_files]
    tasks = []
    for file in files:
        size = int(file["size"])
        if size <= args.chunk_bytes:
            starts = [0]
        else:
            starts = [0]
            starts.extend(rng.randrange(1, size - args.chunk_bytes) for _ in range(max(0, args.windows_per_file - 1)))
        for start in sorted(set(starts)):
            end = min(size - 1, start + args.chunk_bytes - 1)
            tasks.append((file["name"], start, end))

    buffer = []
    started = time.perf_counter()

    def flush() -> None:
        nonlocal buffer, parts
        if not buffer:
            return
        path = args.out_dir / f"part-{len(parts):05d}.parquet"
        pd.DataFrame(buffer).to_parquet(path, index=False)
        parts.append(path)
        buffer = []

    for file_name, start, end in tasks:
        key = f"{file_name}:{start}:{end}"
        if key in completed or kept >= args.max_kept:
            continue
        url = HF_RESOLVE.format(config=HF_CONFIG, file=file_name)
        raw = read_http_range(url, start, end)
        downloaded += len(raw)
        for obj in iter_json_objects(raw, starts_at_zero=start == 0):
            scanned += 1
            row, reason = pubchemqc_record(obj, PubChemQCFilter())
            if row is None:
                rejected[reason] += 1
                continue
            cid, smiles, inchikey = row["cid"], row["canonical_smiles"], row["inchikey"]
            if cid in excluded_cids or smiles in excluded_smiles or inchikey in excluded_inchikeys:
                rejected["excluded"] += 1
                continue
            if cid in seen_cids or smiles in seen_smiles or inchikey in seen_inchikeys:
                rejected["duplicate"] += 1
                continue
            seen_cids.add(cid); seen_smiles.add(smiles); seen_inchikeys.add(inchikey)
            buffer.append(row)
            kept += 1
            if len(buffer) >= args.part_size:
                flush()
            if kept >= args.max_kept:
                break
        completed.add(key)
        state = {
            "config": HF_CONFIG,
            "file_order": "seeded_shuffle",
            "seed": args.seed,
            "max_kept": args.max_kept,
            "kept": kept,
            "scanned": scanned,
            "downloaded_bytes": downloaded,
            "completed_windows": sorted(completed),
            "rejected": dict(rejected),
            "elapsed_seconds_this_run": time.perf_counter() - started,
        }
        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        print(f"kept={kept:,} scanned={scanned:,} windows={len(completed):,}", flush=True)
    flush()
    state["parts"] = [path.name for path in parts]
    state["complete"] = kept >= args.max_kept
    state["elapsed_seconds_this_run"] = time.perf_counter() - started
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    concise_manifest = {
        "config": HF_CONFIG,
        "file_order": "seeded_shuffle",
        "seed": args.seed,
        "candidate_n": kept,
        "unique_cids": len(seen_cids),
        "unique_canonical_smiles": len(seen_smiles),
        "unique_inchikeys": len(seen_inchikeys),
        "scanned_raw_records": scanned,
        "downloaded_bytes": downloaded,
        "completed_windows": len(completed),
        "rejected": dict(rejected),
        "parts": {path.name: sha256_file(path) for path in parts},
        "experiment_manifest_sha256": sha256_file(manifest_path),
    }
    ensure_dirs(args.manifest_out.parent)
    args.manifest_out.write_text(json.dumps(concise_manifest, indent=2), encoding="utf-8")
    print(json.dumps(state, indent=2))


if __name__ == "__main__":
    main()
