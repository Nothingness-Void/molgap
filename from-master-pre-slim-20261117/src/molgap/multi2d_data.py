"""Leakage-safe data assembly for pure-2D coverage experts."""

from __future__ import annotations

import glob
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .router_sampling import compute_scaffold_keys, select_descriptor_diverse
from .utils import canonicalize_smiles


TRAIN_COLUMNS = (
    "cid",
    "mw",
    "formula",
    "smiles",
    "homo",
    "lumo",
    "gap",
    "canonical_smiles",
)
REQUIRED_COLUMNS = ("cid", "smiles", "canonical_smiles", "homo", "lumo", "gap")
SEALED_FEATURES = (
    "mw",
    "heavy_atoms",
    "ring_count",
    "aromatic_rings",
    "aromatic_atom_fraction",
    "rotatable_bonds",
    "conjugated_bonds",
    "has_s",
    "has_cl",
    "has_f",
)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_json(value: Mapping[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_labels(frame: pd.DataFrame, label: str) -> float:
    missing = [column for column in REQUIRED_COLUMNS if column not in frame]
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")
    if frame.loc[:, REQUIRED_COLUMNS].isna().any().any():
        raise ValueError(f"{label} contains missing required values")
    mismatch = float((frame.gap - (frame.lumo - frame.homo)).abs().max())
    if mismatch > 1e-8:
        raise ValueError(f"{label} violates Gap=LUMO-HOMO: max={mismatch}")
    return mismatch


def parse_family_values(values: Iterable[str], *, value_type=int) -> dict[str, object]:
    parsed: dict[str, object] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected FAMILY=VALUE, got {value!r}")
        family, raw = value.split("=", 1)
        if not family or family in parsed:
            raise ValueError(f"Invalid or repeated family in {value!r}")
        parsed[family] = value_type(raw)
    return parsed


def parse_source_patterns(values: Iterable[str]) -> dict[str, list[str]]:
    parsed: dict[str, list[str]] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected FAMILY=GLOB, got {value!r}")
        family, pattern = value.split("=", 1)
        if not family or not pattern:
            raise ValueError(f"Invalid source pattern {value!r}")
        parsed.setdefault(family, []).append(pattern)
    return parsed


def load_candidate_sources(
    source_patterns: Mapping[str, Sequence[str]],
) -> tuple[pd.DataFrame, list[dict]]:
    frames: list[pd.DataFrame] = []
    files: list[dict] = []
    for family, patterns in source_patterns.items():
        paths = [
            Path(value)
            for pattern in patterns
            for value in sorted(glob.glob(pattern))
        ]
        paths = list(dict.fromkeys(paths))
        if not paths:
            raise FileNotFoundError(f"No files match source {family}={patterns}")
        for path in paths:
            frame = pd.read_csv(path)
            validate_labels(frame, str(path))
            frame["source_family"] = family
            frame["source_file"] = path.name
            frames.append(frame)
            files.append({"family": family, "path": str(path), "rows": len(frame)})
    candidates = pd.concat(frames, ignore_index=True)
    return candidates, files


def load_scaffold_exclusions(cache_dirs: Sequence[Path]) -> set[str]:
    scaffolds: set[str] = set()
    for cache_dir in cache_dirs:
        paths = sorted(cache_dir.glob("part_*.csv"))
        if not paths:
            raise FileNotFoundError(f"No scaffold-cache parts in {cache_dir}")
        for path in paths:
            part = pd.read_csv(path, usecols=["scaffold"], dtype="string")
            scaffolds.update(part.scaffold.dropna().astype(str))
    return scaffolds


def cached_scaffolds(
    frame: pd.DataFrame,
    cache_dir: Path,
    *,
    workers: int,
    chunk_size: int,
) -> pd.Series:
    """Compute scaffold keys in durable chunks and validate reused chunks."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[pd.Series] = []
    total_parts = (len(frame) + chunk_size - 1) // chunk_size
    for part_index, start in enumerate(range(0, len(frame), chunk_size)):
        stop = min(start + chunk_size, len(frame))
        expected = frame.iloc[start:stop][["cid", "canonical_smiles"]].reset_index(drop=True)
        path = cache_dir / f"part_{part_index:05d}.csv"
        if path.exists():
            part = pd.read_csv(path, dtype="string")
            if len(part) != len(expected) or not part[["cid", "canonical_smiles"]].equals(
                expected.astype("string")
            ):
                raise RuntimeError(f"Invalid scaffold cache part: {path}")
        else:
            part = expected.copy()
            part["scaffold"] = compute_scaffold_keys(
                expected.canonical_smiles.astype(str).tolist(), workers=workers
            )
            atomic_csv(part, path)
        outputs.append(part.scaffold.astype(str))
        atomic_json(
            {
                "completed_parts": part_index + 1,
                "total_parts": total_parts,
                "completed_rows": stop,
            },
            cache_dir / "progress.json",
        )
        print(
            f"scaffolds {part_index + 1}/{total_parts}: {stop:,}/{len(frame):,}",
            flush=True,
        )
    return pd.concat(outputs, ignore_index=True)


def _stable_rank(frame: pd.DataFrame, namespace: str) -> pd.Series:
    values = (
        frame["source_family"].astype(str)
        + "|"
        + frame["cid"].astype(str)
        + "|"
        + frame["canonical_smiles"].astype(str)
    )
    return values.map(
        lambda value: hashlib.sha256(f"{namespace}|{value}".encode()).hexdigest()
    )


def _select_sealed(
    candidates: pd.DataFrame,
    quotas: Mapping[str, int],
    *,
    excluded_scaffolds: set[str],
    seed: int,
    clusters: int,
) -> pd.DataFrame:
    novel = candidates.loc[~candidates.scaffold.isin(excluded_scaffolds)].copy()
    novel["_rank"] = _stable_rank(novel, f"sealed-{seed}")
    novel = novel.sort_values("_rank").drop_duplicates("scaffold").reset_index(drop=True)
    selected: list[pd.DataFrame] = []
    used_scaffolds: set[str] = set()
    for family, quota in quotas.items():
        family_rows = novel.loc[
            (novel.source_family == family) & ~novel.scaffold.isin(used_scaffolds)
        ].copy()
        if len(family_rows) < quota:
            raise RuntimeError(
                f"Only {len(family_rows):,} novel {family} scaffolds for sealed quota {quota:,}"
            )
        indices, probabilities = select_descriptor_diverse(
            family_rows,
            family_rows.index.to_numpy(dtype=np.int64),
            features=SEALED_FEATURES,
            n_select=quota,
            n_clusters=min(clusters, quota),
            seed=seed + len(selected),
        )
        part = family_rows.loc[indices].copy()
        part["selection_probability"] = [probabilities[int(index)] for index in indices]
        selected.append(part)
        used_scaffolds.update(part.scaffold.astype(str))
    sealed = pd.concat(selected, ignore_index=True)
    if len(sealed) != sum(quotas.values()) or sealed.scaffold.nunique() != len(sealed):
        raise RuntimeError("Sealed selection did not preserve one unique scaffold per row")
    return sealed.drop(columns=["_rank"], errors="ignore")


def assemble_coverage_topup(
    *,
    base_csv: Path,
    source_patterns: Mapping[str, Sequence[str]],
    train_quotas: Mapping[str, int],
    sealed_quotas: Mapping[str, int],
    exclusion_cache_dirs: Sequence[Path],
    existing_sealed_csvs: Sequence[Path],
    scaffold_cache_dir: Path,
    topup_out: Path,
    sealed_out: Path,
    audit_out: Path,
    report_out: Path,
    workers: int,
    chunk_size: int,
    seed: int,
    clusters: int,
) -> dict:
    candidates, source_files = load_candidate_sources(source_patterns)
    raw_rows = len(candidates)
    duplicate_cid = int(candidates.cid.astype(str).duplicated().sum())
    duplicate_smiles = int(candidates.canonical_smiles.astype(str).duplicated().sum())
    if duplicate_cid or duplicate_smiles:
        raise RuntimeError(
            f"Accepted sources are not mutually unique: cid={duplicate_cid}, smiles={duplicate_smiles}"
        )

    base = pd.read_csv(base_csv, usecols=["cid", "canonical_smiles"], dtype="string")
    base_cids = set(base.cid.dropna())
    base_smiles = set(base.canonical_smiles.dropna())
    overlap_cid = int(candidates.cid.astype(str).isin(base_cids).sum())
    overlap_smiles = int(candidates.canonical_smiles.astype(str).isin(base_smiles).sum())
    if overlap_cid or overlap_smiles:
        raise RuntimeError(
            f"Accepted sources overlap the 1.5M base: cid={overlap_cid}, smiles={overlap_smiles}"
        )

    candidates["scaffold"] = cached_scaffolds(
        candidates,
        scaffold_cache_dir,
        workers=workers,
        chunk_size=chunk_size,
    )
    training_scaffolds = load_scaffold_exclusions(exclusion_cache_dirs)
    prior_sealed_scaffolds: set[str] = set()
    for path in existing_sealed_csvs:
        frame = pd.read_csv(path)
        if "scaffold" not in frame:
            frame["scaffold"] = compute_scaffold_keys(
                frame.canonical_smiles.astype(str).tolist(), workers=workers
            )
        prior_sealed_scaffolds.update(frame.scaffold.dropna().astype(str))

    sealed = _select_sealed(
        candidates.loc[~candidates.scaffold.isin(prior_sealed_scaffolds)],
        sealed_quotas,
        excluded_scaffolds=training_scaffolds,
        seed=seed,
        clusters=clusters,
    )
    forbidden_scaffolds = prior_sealed_scaffolds | set(sealed.scaffold.astype(str))
    eligible = candidates.loc[~candidates.scaffold.isin(forbidden_scaffolds)].copy()
    eligible["_rank"] = _stable_rank(eligible, f"train-{seed}")
    selected_parts: list[pd.DataFrame] = []
    for family, quota in train_quotas.items():
        part = (
            eligible.loc[eligible.source_family == family]
            .sort_values("_rank")
            .head(quota)
        )
        if len(part) != quota:
            raise RuntimeError(f"Only {len(part):,} eligible {family} rows for quota {quota:,}")
        selected_parts.append(part)
    topup = pd.concat(selected_parts, ignore_index=True)
    if topup.cid.astype(str).duplicated().any() or topup.canonical_smiles.astype(str).duplicated().any():
        raise RuntimeError("Selected top-up contains duplicate identities")
    if set(topup.scaffold.astype(str)) & forbidden_scaffolds:
        raise RuntimeError("Selected top-up overlaps a sealed scaffold")

    sealed = sealed.copy()
    sealed["sealed_role"] = "multi2d_2m_future_acceptance"
    topup_train = topup.loc[:, TRAIN_COLUMNS]
    atomic_csv(topup_train, topup_out)
    atomic_csv(sealed.drop(columns=["_rank"], errors="ignore"), sealed_out)
    audit = (
        topup.groupby(["source_family", "source_file", "bucket"], dropna=False)
        .size()
        .rename("selected_rows")
        .reset_index()
    )
    atomic_csv(audit, audit_out)
    report = {
        "dataset": "phase8_multi2d_coverage_expert_2m",
        "base_csv": str(base_csv),
        "base_rows": len(base),
        "candidate_raw_rows": raw_rows,
        "candidate_files": source_files,
        "train_quotas": {key: int(value) for key, value in train_quotas.items()},
        "sealed_quotas": {key: int(value) for key, value in sealed_quotas.items()},
        "selected_topup_rows": len(topup),
        "future_sealed_rows": len(sealed),
        "future_sealed_unique_scaffolds": int(sealed.scaffold.nunique()),
        "prior_sealed_scaffolds_excluded": len(prior_sealed_scaffolds),
        "training_scaffold_exclusions": len(training_scaffolds),
        "full_training_rows_after_append": len(base) + len(topup),
        "topup_sha256": sha256_file(topup_out),
        "future_sealed_sha256": sha256_file(sealed_out),
        "gap_identity_max_abs_eV": validate_labels(topup, "selected top-up"),
        "seed": seed,
        "scaffold_disjoint_future_sealed": True,
        "future_sealed_training_use_forbidden": True,
    }
    atomic_json(report, report_out)
    return report


def _identity_sets(paths: Sequence[Path]) -> tuple[set[str], set[str]]:
    cids: set[str] = set()
    smiles: set[str] = set()
    for path in paths:
        columns = pd.read_csv(path, nrows=0).columns
        usecols = [column for column in ("cid", "canonical_smiles") if column in columns]
        if not usecols:
            continue
        for chunk in pd.read_csv(path, usecols=usecols, dtype="string", chunksize=250_000):
            if "cid" in chunk:
                cids.update(chunk.cid.dropna().astype(str))
            if "canonical_smiles" in chunk:
                smiles.update(chunk.canonical_smiles.dropna().astype(str))
    return cids, smiles


def _evaluation_exclusions(
    paths: Sequence[Path], *, workers: int
) -> tuple[set[str], set[str], set[str], list[dict]]:
    cids: set[str] = set()
    smiles: set[str] = set()
    scaffolds: set[str] = set()
    records: list[dict] = []
    for path in paths:
        frame = pd.read_csv(path)
        if "cid" in frame:
            cids.update(frame.cid.dropna().astype(str))
        smiles_column = "canonical_smiles" if "canonical_smiles" in frame else "smiles"
        raw_smiles = frame[smiles_column].dropna().astype(str)
        canonical = (
            raw_smiles
            if smiles_column == "canonical_smiles"
            else raw_smiles.map(canonicalize_smiles).dropna().astype(str)
        )
        smiles.update(canonical)
        if "scaffold" in frame:
            path_scaffolds = set(frame.scaffold.dropna().astype(str))
        else:
            path_scaffolds = set(compute_scaffold_keys(raw_smiles.tolist(), workers=workers))
        scaffolds.update(path_scaffolds)
        records.append(
            {
                "path": str(path),
                "rows": len(frame),
                "unique_scaffolds": len(path_scaffolds),
            }
        )
    return cids, smiles, scaffolds, records


def select_bucketed_rescue_topup(
    *,
    assembly_report: Path,
    source_families: Sequence[str],
    quotas: Mapping[str, int],
    used_csvs: Sequence[Path],
    evaluation_csvs: Sequence[Path],
    scaffold_cache_dir: Path,
    topup_out: Path,
    audit_out: Path,
    report_out: Path,
    workers: int,
    seed: int,
    clusters: int,
) -> dict:
    """Select a descriptor-diverse hard top-up without evaluation leakage."""
    assembly = json.loads(assembly_report.read_text(encoding="utf-8"))
    requested_families = set(source_families)
    sources = [
        record for record in assembly["candidate_files"]
        if record["family"] in requested_families
    ]
    if not sources:
        raise ValueError(f"No {sorted(requested_families)!r} sources in {assembly_report}")

    frames: list[pd.DataFrame] = []
    for record in sources:
        path = Path(record["path"])
        frame = pd.read_csv(path)
        validate_labels(frame, str(path))
        frame["source_family"] = record["family"]
        frame["source_file"] = path.name
        frames.append(frame)
    candidates = pd.concat(frames, ignore_index=True)
    candidates["cid"] = candidates.cid.astype("string")
    candidates["canonical_smiles"] = candidates.canonical_smiles.astype("string")
    if candidates.cid.duplicated().any() or candidates.canonical_smiles.duplicated().any():
        raise RuntimeError(f"{sorted(requested_families)} sources contain duplicate identities")

    cache_paths = sorted(scaffold_cache_dir.glob("part_*.csv"))
    if not cache_paths:
        raise FileNotFoundError(f"No scaffold cache parts in {scaffold_cache_dir}")
    scaffold_cache = pd.concat(
        [pd.read_csv(path, dtype="string") for path in cache_paths],
        ignore_index=True,
    )
    scaffold_cache = scaffold_cache[["cid", "canonical_smiles", "scaffold"]]
    candidates = candidates.merge(
        scaffold_cache,
        on=["cid", "canonical_smiles"],
        how="left",
        validate="one_to_one",
    )
    if candidates.scaffold.isna().any():
        raise RuntimeError(
            f"Scaffold cache misses {int(candidates.scaffold.isna().sum()):,} candidates"
        )

    used_cids, used_smiles = _identity_sets(used_csvs)
    eval_cids, eval_smiles, eval_scaffolds, eval_records = _evaluation_exclusions(
        evaluation_csvs, workers=workers
    )
    exact_excluded = (
        candidates.cid.isin(used_cids | eval_cids)
        | candidates.canonical_smiles.isin(used_smiles | eval_smiles)
    )
    scaffold_excluded = candidates.scaffold.isin(eval_scaffolds)
    eligible = candidates.loc[~exact_excluded & ~scaffold_excluded].copy()
    eligible["_rank"] = _stable_rank(eligible, f"hard-rescue-{seed}")
    eligible = eligible.sort_values("_rank").reset_index(drop=True)

    selected_parts: list[pd.DataFrame] = []
    available_by_bucket: dict[str, int] = {}
    available_scaffolds_by_bucket: dict[str, int] = {}
    for bucket, quota in quotas.items():
        pool = eligible.loc[eligible.bucket == bucket].copy()
        available_by_bucket[bucket] = len(pool)
        available_scaffolds_by_bucket[bucket] = int(pool.scaffold.nunique())
        if len(pool) < quota:
            raise RuntimeError(
                f"Only {len(pool):,} eligible rows for {bucket}; need {quota:,}"
            )
        indices, probabilities = select_descriptor_diverse(
            pool,
            pool.index.to_numpy(dtype=np.int64),
            features=SEALED_FEATURES,
            n_select=quota,
            n_clusters=min(clusters, quota),
            seed=seed + len(selected_parts),
        )
        part = pool.loc[indices].copy()
        part["selection_probability"] = [probabilities[int(index)] for index in indices]
        selected_parts.append(part)

    selected = pd.concat(selected_parts, ignore_index=True)
    expected_rows = sum(int(value) for value in quotas.values())
    if len(selected) != expected_rows:
        raise RuntimeError("Hard top-up selection returned an unexpected row count")
    if selected.cid.duplicated().any() or selected.canonical_smiles.duplicated().any():
        raise RuntimeError("Hard top-up selection contains duplicate identities")
    if set(selected.scaffold.astype(str)) & eval_scaffolds:
        raise RuntimeError("Hard top-up overlaps an evaluation scaffold")

    atomic_csv(selected.loc[:, TRAIN_COLUMNS], topup_out)
    audit = (
        selected.groupby(["bucket", "source_file"], dropna=False)
        .size()
        .rename("selected_rows")
        .reset_index()
    )
    atomic_csv(audit, audit_out)
    report = {
        "dataset": "phase8_multi2d_2m_hard20k_rescue",
        "assembly_report": str(assembly_report),
        "source_families": sorted(requested_families),
        "candidate_rows": len(candidates),
        "used_csvs": [str(path) for path in used_csvs],
        "evaluation_exclusions": eval_records,
        "used_identity_cids": len(used_cids),
        "used_identity_canonical_smiles": len(used_smiles),
        "evaluation_scaffolds": len(eval_scaffolds),
        "candidate_exact_excluded_rows": int(exact_excluded.sum()),
        "candidate_eval_scaffold_excluded_rows": int(scaffold_excluded.sum()),
        "eligible_rows": len(eligible),
        "eligible_unique_scaffolds": int(eligible.scaffold.nunique()),
        "available_by_bucket": available_by_bucket,
        "available_scaffolds_by_bucket": available_scaffolds_by_bucket,
        "quotas": {key: int(value) for key, value in quotas.items()},
        "selected_rows": len(selected),
        "selected_unique_scaffolds": int(selected.scaffold.nunique()),
        "topup_sha256": sha256_file(topup_out),
        "gap_identity_max_abs_eV": validate_labels(selected, "hard rescue top-up"),
        "scaffold_disjoint_evaluation": True,
        "future_sealed_training_use_forbidden": True,
        "seed": seed,
    }
    atomic_json(report, report_out)
    return report
