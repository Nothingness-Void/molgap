"""Select the auditable 500K repair-v2 top-up without mutating expansion500K.

The repair-v2 collector produced several ordered candidate sources.  This script
forms their strict CID/canonical-SMILES union, excludes both the frozen 500K
base and the rejected 1M continuation, then selects exactly 500K rows using
the sampling-spec proportions and Bemis-Murcko scaffold novelty against the
base.  It deliberately writes only the *new* 500K rows.  A later, separate
assembler is responsible for concatenating the untouched base as rows 0-499999.

The scaffold cache is chunked and atomically written so a stopped local/cloud
run resumes from completed chunks instead of recomputing all RDKit scaffolds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterable

import pandas as pd

from molgap.constants import RAW_DIR, RESULTS_DIR
from molgap.utils import ensure_dirs, scaffold_split_key


BASE_CSV = RAW_DIR / "phase8_expansion_500k.csv"
REJECTED_CSV = RAW_DIR / "phase8_expansion_1m.csv"
SPEC_JSON = RESULTS_DIR / "phase8" / "repair_1m_v2_sampling_spec.json"
OUT_CSV = RAW_DIR / "phase8_repair_v2_selected_500k.csv"
OUT_AUDIT_CSV = RESULTS_DIR / "phase8" / "repair_v2_selected_500k_audit.csv"
OUT_REPORT_JSON = RESULTS_DIR / "phase8" / "repair_v2_selection_500k_report.json"
OUT_REPORT_MD = RESULTS_DIR / "phase8" / "repair_v2_selection_500k_report.md"
CACHE_DIR = RESULTS_DIR / "phase8" / "repair_v2_selection_scaffold_cache"

TRAIN_COLUMNS = ["cid", "mw", "formula", "smiles", "homo", "lumo", "gap", "canonical_smiles"]
SOURCE_NAMES = (
    "rare",
    "aromatic_large",
    "topology_elements",
    "balanced",
)
DEFAULT_CANDIDATE_CSVS = [
    RESULTS_DIR / "phase8" / "remote" / "kaggle" / "repair_v2_full_600k_v3" / "reconciled_20260719" / f"phase8_repair_v2_{name}.csv"
    for name in SOURCE_NAMES
] + [
    RAW_DIR / "phase8_repair_v2_checkpoint_dataset" / f"phase8_repair_v2_round{round_index:02d}_{name}.csv"
    for round_index in (1, 2)
    for name in SOURCE_NAMES
] + [
    RESULTS_DIR / "phase8" / "remote" / "kaggle" / "repair_v2_round_03" / "phase8_repair_v2_round_03" / f"phase8_repair_v2_round03_{name}.csv"
    for name in SOURCE_NAMES
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-csv", type=Path, default=BASE_CSV)
    parser.add_argument("--rejected-csv", type=Path, default=REJECTED_CSV)
    parser.add_argument("--sampling-spec", type=Path, default=SPEC_JSON)
    parser.add_argument("--candidate-csv", type=Path, action="append", default=None,
                        help="Ordered source CSV. If omitted, use the accepted full/R1/R2/R3 sources.")
    parser.add_argument("--out-csv", type=Path, default=OUT_CSV)
    parser.add_argument("--out-audit-csv", type=Path, default=OUT_AUDIT_CSV)
    parser.add_argument("--report-json", type=Path, default=OUT_REPORT_JSON)
    parser.add_argument("--report-md", type=Path, default=OUT_REPORT_MD)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--workers", type=int, default=max(1, min(12, (os.cpu_count() or 4) - 1)))
    parser.add_argument("--scaffold-chunk-size", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def atomic_write_json(path: Path, value: dict) -> None:
    ensure_dirs(path.parent)
    temp = path.with_name(f".{path.name}.tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temp, path)


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    ensure_dirs(path.parent)
    temp = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temp, index=False, encoding="utf-8")
    os.replace(temp, path)


def string_key(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def load_key_sets(path: Path) -> tuple[set[str], set[str]]:
    cids: set[str] = set()
    smiles: set[str] = set()
    for chunk in pd.read_csv(
        path,
        usecols=lambda name: name in {"cid", "canonical_smiles"},
        dtype={"cid": "string", "canonical_smiles": "string"},
        chunksize=100_000,
    ):
        cids.update(value for value in (string_key(item) for item in chunk.get("cid", [])) if value is not None)
        smiles.update(value for value in (string_key(item) for item in chunk.get("canonical_smiles", [])) if value is not None)
    return cids, smiles


def strict_union(paths: Iterable[Path]) -> tuple[pd.DataFrame, list[dict]]:
    """Return source-ordered strict union, tracking every duplicate rejection."""
    seen_cids: set[str] = set()
    seen_smiles: set[str] = set()
    accepted: list[pd.DataFrame] = []
    source_report: list[dict] = []
    for source_index, path in enumerate(paths):
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path, dtype={"cid": "string", "canonical_smiles": "string"})
        required = {"cid", "smiles", "canonical_smiles", "homo", "lumo", "gap", "bucket"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")
        keep: list[bool] = []
        duplicate_rows = 0
        for cid_value, smiles_value in frame[["cid", "canonical_smiles"]].itertuples(index=False):
            cid, smiles = string_key(cid_value), string_key(smiles_value)
            duplicate = (cid is not None and cid in seen_cids) or (smiles is not None and smiles in seen_smiles)
            keep.append(not duplicate)
            if duplicate:
                duplicate_rows += 1
                continue
            if cid is not None:
                seen_cids.add(cid)
            if smiles is not None:
                seen_smiles.add(smiles)
        kept = frame.loc[keep].copy()
        kept["source_index"] = source_index
        kept["source_file"] = path.name
        accepted.append(kept)
        source_report.append({
            "csv": str(path),
            "rows": int(len(frame)),
            "new_unique_rows": int(len(kept)),
            "overlap_rows": int(duplicate_rows),
        })
        print(f"union {source_index + 1}: {path.name}: {len(kept):,}/{len(frame):,} unique", flush=True)
    return pd.concat(accepted, ignore_index=True, sort=False), source_report


def filter_against_existing(
    candidates: pd.DataFrame,
    base_cids: set[str],
    base_smiles: set[str],
    rejected_cids: set[str],
    rejected_smiles: set[str],
) -> tuple[pd.DataFrame, dict]:
    cid_keys = candidates["cid"].map(string_key)
    smiles_keys = candidates["canonical_smiles"].map(string_key)
    invalid = candidates[["smiles", "canonical_smiles", "homo", "lumo", "gap"]].isna().any(axis=1)
    gap = pd.to_numeric(candidates["gap"], errors="coerce")
    base_overlap = cid_keys.isin(base_cids) | smiles_keys.isin(base_smiles)
    rejected_overlap = cid_keys.isin(rejected_cids) | smiles_keys.isin(rejected_smiles)
    valid = ~(invalid | (gap <= 0) | base_overlap | rejected_overlap)
    report = {
        "input_rows": int(len(candidates)),
        "excluded_invalid_or_nonpositive_gap": int((invalid | (gap <= 0)).sum()),
        "excluded_base_overlap": int(base_overlap.sum()),
        "excluded_rejected_1m_overlap": int(rejected_overlap.sum()),
        "usable_rows": int(valid.sum()),
    }
    return candidates.loc[valid].copy().reset_index(drop=True), report


def cache_signature(frame: pd.DataFrame) -> dict:
    values = frame[["cid", "canonical_smiles"]]
    sample = pd.concat([values.head(3), values.tail(3)], ignore_index=True).to_csv(index=False)
    return {
        "rows": int(len(frame)),
        "sample_sha256": hashlib.sha256(sample.encode("utf-8")).hexdigest(),
    }


def scaffold_cache(frame: pd.DataFrame, *, label: str, cache_dir: Path, workers: int, chunk_size: int) -> pd.Series:
    """Compute or resume scaffold keys with independently atomic chunk files."""
    label_dir = cache_dir / label
    ensure_dirs(label_dir)
    signature = cache_signature(frame)
    meta_path = label_dir / "meta.json"
    if meta_path.exists():
        existing = json.loads(meta_path.read_text(encoding="utf-8"))
        if existing.get("signature") != signature:
            raise RuntimeError(f"Scaffold cache signature mismatch at {label_dir}; choose a new --cache-dir")
    else:
        atomic_write_json(meta_path, {"signature": signature, "chunk_size": chunk_size})

    parts: list[pd.Series] = []
    total_parts = math.ceil(len(frame) / chunk_size)
    for part_index in range(total_parts):
        start = part_index * chunk_size
        end = min(len(frame), start + chunk_size)
        part_path = label_dir / f"part_{part_index:05d}.csv"
        expected = frame.iloc[start:end][["cid", "canonical_smiles"]].reset_index(drop=True)
        if part_path.exists():
            cached = pd.read_csv(part_path, dtype={"cid": "string", "canonical_smiles": "string", "scaffold": "string"})
            if len(cached) == len(expected) and cached[["cid", "canonical_smiles"]].equals(expected.astype("string")):
                parts.append(cached["scaffold"].astype(str))
                print(f"scaffold {label}: reuse {part_index + 1}/{total_parts}", flush=True)
                continue
            raise RuntimeError(f"Invalid scaffold cache chunk: {part_path}")

        smiles = expected["canonical_smiles"].astype(str).tolist()
        print(f"scaffold {label}: compute {part_index + 1}/{total_parts} ({start:,}-{end:,})", flush=True)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            scaffolds = list(pool.map(scaffold_split_key, smiles, chunksize=500))
        completed = expected.copy()
        completed["scaffold"] = scaffolds
        atomic_write_csv(part_path, completed)
        parts.append(completed["scaffold"].astype(str))
        atomic_write_json(label_dir / "progress.json", {
            "label": label,
            "completed_parts": part_index + 1,
            "total_parts": total_parts,
            "completed_rows": end,
        })
    return pd.concat(parts, ignore_index=True)


def proportional_quotas(spec: dict, selection_target: int) -> dict[str, int]:
    buckets = spec["priority_buckets"]
    source_total = sum(int(item["quota"]) for item in buckets)
    raw = [(item["id"], int(item["quota"]) * selection_target / source_total) for item in buckets]
    quotas = {bucket: math.floor(value) for bucket, value in raw}
    remainder = selection_target - sum(quotas.values())
    for bucket, _ in sorted(raw, key=lambda item: (-(item[1] % 1), item[0]))[:remainder]:
        quotas[bucket] += 1
    return quotas


def stable_rank(value: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


def rank_candidates(frame: pd.DataFrame, base_scaffold_frequency: Counter[str], seed: int) -> pd.DataFrame:
    ranked = frame.copy()
    ranked["base_scaffold_frequency"] = ranked["scaffold"].map(base_scaffold_frequency).fillna(0).astype(int)
    ranked["is_new_scaffold"] = ranked["base_scaffold_frequency"].eq(0)
    ranked["candidate_scaffold_frequency"] = ranked["scaffold"].map(ranked["scaffold"].value_counts()).astype(int)
    ranked["stable_rank"] = [stable_rank(f"{cid}|{smiles}", seed) for cid, smiles in ranked[["cid", "canonical_smiles"]].itertuples(index=False)]
    return ranked.sort_values(
        ["is_new_scaffold", "base_scaffold_frequency", "candidate_scaffold_frequency", "stable_rank"],
        ascending=[False, True, True, True],
        kind="mergesort",
    )


def select_rows(ranked: pd.DataFrame, quotas: dict[str, int], target: int) -> tuple[pd.DataFrame, dict]:
    selected_indices: list[int] = []
    selected_set: set[int] = set()
    bucket_report: dict[str, dict] = {}
    for bucket, quota in quotas.items():
        bucket_rows = ranked.loc[ranked["bucket"] == bucket]
        take = bucket_rows.head(quota)
        selected_indices.extend(int(index) for index in take.index)
        selected_set.update(int(index) for index in take.index)
        bucket_report[bucket] = {
            "requested": int(quota),
            "available": int(len(bucket_rows)),
            "selected_by_quota": int(len(take)),
            "shortfall": int(max(0, quota - len(take))),
        }

    if len(selected_indices) > target:
        raise RuntimeError("Bucket quotas exceed target")
    remaining = ranked.loc[~ranked.index.isin(selected_set)]
    top_up = remaining.head(target - len(selected_indices))
    selected_indices.extend(int(index) for index in top_up.index)
    if len(selected_indices) != target:
        raise RuntimeError(f"Candidate supply insufficient: selected {len(selected_indices):,}/{target:,}")
    selected = ranked.loc[selected_indices].copy()
    for bucket, count in top_up["bucket"].value_counts().items():
        bucket_report.setdefault(str(bucket), {"requested": 0, "available": 0, "selected_by_quota": 0, "shortfall": 0})
        bucket_report[str(bucket)]["selected_as_top_up"] = int(count)
    for details in bucket_report.values():
        details.setdefault("selected_as_top_up", 0)
    return selected, bucket_report


def write_markdown(report: dict, path: Path) -> None:
    lines = [
        "# Phase 8 Repair-v2 500K Selection",
        "",
        "This artifact is the new 500K top-up only. It does not alter or concatenate the frozen expansion500K base.",
        "",
        f"- strict candidate union: {report['strict_union_rows']:,}",
        f"- usable after exclusion: {report['filtering']['usable_rows']:,}",
        f"- selected top-up: {report['selected_rows']:,}",
        f"- base-overlap rows in selection: {report['post_selection_checks']['base_overlap_rows']}",
        f"- rejected-1M overlap rows in selection: {report['post_selection_checks']['rejected_overlap_rows']}",
        f"- selected unseen base scaffolds: {report['selected_scaffold_novelty']['unseen_base_scaffold_rows']:,}",
        "",
        "## Method",
        "",
        "- Strict CID/canonical-SMILES union in documented source order.",
        "- Explicit exclusion against frozen expansion500K and rejected expansion1M.",
        "- Bucket quotas scaled from the 600K collection specification to the 500K selection target.",
        "- Within each bucket: unseen Bemis-Murcko scaffold, lower base-scaffold frequency, lower candidate-scaffold frequency, then stable SHA-256 tie-break.",
        "- This is a scaffold-novelty selection. It does not claim an exhaustive all-pairs Morgan-fingerprint nearest-neighbour calculation.",
        "",
        "## Bucket Audit",
        "",
        "| bucket | requested | available | quota selected | top-up selected | shortfall |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for bucket, details in report["bucket_selection"].items():
        lines.append(
            f"| `{bucket}` | {details['requested']:,} | {details['available']:,} | "
            f"{details['selected_by_quota']:,} | {details['selected_as_top_up']:,} | {details['shortfall']:,} |"
        )
    ensure_dirs(path.parent)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    candidate_paths = args.candidate_csv or DEFAULT_CANDIDATE_CSVS
    spec = json.loads(args.sampling_spec.read_text(encoding="utf-8"))
    target = int(spec["selection_target"])
    print(f"Loading frozen base: {args.base_csv}", flush=True)
    base = pd.read_csv(args.base_csv, dtype={"cid": "string", "canonical_smiles": "string"})
    base_cids, base_smiles = load_key_sets(args.base_csv)
    rejected_cids, rejected_smiles = load_key_sets(args.rejected_csv)
    print(f"Strict-union candidate sources: {len(candidate_paths)}", flush=True)
    candidates, source_report = strict_union(candidate_paths)
    strict_union_rows = len(candidates)
    candidates, filtering = filter_against_existing(candidates, base_cids, base_smiles, rejected_cids, rejected_smiles)
    if len(candidates) < target:
        raise RuntimeError(f"Only {len(candidates):,} usable candidates for target {target:,}")

    base_scaffolds = scaffold_cache(base, label="base_500k", cache_dir=args.cache_dir,
                                    workers=args.workers, chunk_size=args.scaffold_chunk_size)
    candidates["scaffold"] = scaffold_cache(candidates, label="candidate_union", cache_dir=args.cache_dir,
                                               workers=args.workers, chunk_size=args.scaffold_chunk_size)
    base_scaffold_frequency = Counter(base_scaffolds.astype(str))
    ranked = rank_candidates(candidates, base_scaffold_frequency, args.seed)
    quotas = proportional_quotas(spec, target)
    selected, bucket_selection = select_rows(ranked, quotas, target)

    selected_cids = set(selected["cid"].map(string_key).dropna())
    selected_smiles = set(selected["canonical_smiles"].map(string_key).dropna())
    post_checks = {
        "base_overlap_rows": int(bool(selected_cids & base_cids) or bool(selected_smiles & base_smiles)),
        "rejected_overlap_rows": int(bool(selected_cids & rejected_cids) or bool(selected_smiles & rejected_smiles)),
        "duplicate_cid_rows": int(len(selected) - selected["cid"].nunique()),
        "duplicate_canonical_smiles_rows": int(len(selected) - selected["canonical_smiles"].nunique()),
    }
    if any(post_checks.values()):
        raise RuntimeError(f"Selection validation failed: {post_checks}")

    audit_columns = [
        "cid", "canonical_smiles", "bucket", "source_index", "source_file", "scaffold",
        "is_new_scaffold", "base_scaffold_frequency", "candidate_scaffold_frequency", "stable_rank",
    ]
    atomic_write_csv(args.out_csv, selected[TRAIN_COLUMNS])
    atomic_write_csv(args.out_audit_csv, selected[audit_columns])
    selected_bucket_counts = {str(key): int(value) for key, value in selected["bucket"].value_counts().items()}
    novelty = {
        "unseen_base_scaffold_rows": int(selected["is_new_scaffold"].sum()),
        "seen_base_scaffold_rows": int((~selected["is_new_scaffold"]).sum()),
        "unique_selected_scaffolds": int(selected["scaffold"].nunique()),
        "unique_base_scaffolds": int(len(base_scaffold_frequency)),
        "method": "Bemis-Murcko scaffold novelty and frequency ranking; no exhaustive all-pairs Morgan calculation",
    }
    report = {
        "dataset": spec.get("dataset"),
        "base_csv": str(args.base_csv),
        "rejected_csv": str(args.rejected_csv),
        "candidate_csvs": [str(path) for path in candidate_paths],
        "strict_union_rows": int(strict_union_rows),
        "strict_union_sources": source_report,
        "filtering": filtering,
        "selection_target": target,
        "quota_method": "proportional scaling of candidate-pool quotas from 600K to 500K, then deterministic top-up on shortages",
        "scaled_quotas": quotas,
        "bucket_selection": bucket_selection,
        "selected_bucket_counts": selected_bucket_counts,
        "selected_rows": int(len(selected)),
        "selected_scaffold_novelty": novelty,
        "post_selection_checks": post_checks,
        "outputs": {"selected_csv": str(args.out_csv), "audit_csv": str(args.out_audit_csv)},
        "seed": args.seed,
    }
    atomic_write_json(args.report_json, report)
    write_markdown(report, args.report_md)
    print(json.dumps({
        "strict_union_rows": strict_union_rows,
        "usable_rows": filtering["usable_rows"],
        "selected_rows": len(selected),
        "unseen_base_scaffold_rows": novelty["unseen_base_scaffold_rows"],
        "report": str(args.report_json),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
