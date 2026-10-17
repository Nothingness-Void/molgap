"""Durable row-level auditing and deterministic repair of scaled B3LYP corpora."""

from __future__ import annotations

import hashlib
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold


TARGET_COLUMNS = ("homo", "lumo", "gap")
TRAIN_COLUMNS = (
    "cid", "mw", "formula", "smiles", "homo", "lumo", "gap", "canonical_smiles"
)
IDENTITY_COLUMNS = ("cid", "canonical_smiles")
NOBLE_GASES = frozenset({2, 10, 18, 36, 54, 86})
COMMON_ORGANIC_ELEMENTS = frozenset({1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53})
HARD_GROUPS = frozenset(
    {"flexible_lowmid", "macro_amide", "sp3_nonaromatic", "very_large"}
)
COMPLEMENTARY_GROUPS = frozenset(
    {"high_gap", "hetero_dense", "bridged_rigid", "conjugated_da"}
)
BROAD_GROUPS = frozenset(
    {"aromatic_large", "balanced", "rare", "topology_elements"}
)


@dataclass(frozen=True)
class SourceSpec:
    source_id: str
    path: str
    source_family: str
    source_group: str
    start_row: int = 0
    stop_row: int | None = None
    immutable: bool = False
    current_2m: bool = False


def atomic_json(value: object, path: Path) -> None:
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


def stable_hash(*parts: object, seed: int) -> str:
    value = "|".join(str(part) for part in parts)
    return hashlib.sha256(f"{seed}|{value}".encode()).hexdigest()


def infer_source_group(path: Path) -> tuple[str, str]:
    name = path.stem.lower()
    if "general_overnight" in name:
        return "general", "general"
    match = re.search(r"round\d+_(.+)$", name)
    group = match.group(1) if match else "unknown"
    if group in HARD_GROUPS:
        return "hard", group
    if group in COMPLEMENTARY_GROUPS:
        return "complementary", group
    if group in BROAD_GROUPS:
        return "broad", group
    return "other", group


def discover_sources(root: Path) -> list[SourceSpec]:
    """Return the exact current-2M segments plus accepted acquisition CSVs."""
    core = root / "data" / "raw" / "phase8_repair_v3_1p5m.csv"
    topup = root / "data" / "raw" / "phase8_multi2d_2m_topup_500k.csv"
    if not core.exists() or not topup.exists():
        raise FileNotFoundError("The exact-2M source CSVs are not both available")
    sources = [
        SourceSpec(
            "core_targeted500k", str(core), "core", "targeted",
            0, 500_000, True, True,
        ),
        SourceSpec(
            "core_original_general500k", str(core), "core", "original_general",
            500_000, 1_000_000, False, True,
        ),
        SourceSpec(
            "core_repair500k", str(core), "core", "repair_v2",
            1_000_000, 1_500_000, False, True,
        ),
        SourceSpec(
            "core_exact2m_topup500k", str(topup), "core", "exact2m_topup",
            0, 500_000, False, True,
        ),
    ]
    completed = root / "results" / "kaggle" / "acquisition" / "completed"
    accepted_paths = {
        path.resolve()
        for path in completed.rglob("*.csv")
        if (
            "accepted" in str(path.parent).lower()
            and "sealed" not in path.name.lower()
            and "pcqm" not in str(path).lower()
        )
    }
    for path in sorted(accepted_paths):
        family, group = infer_source_group(path)
        relative = path.relative_to(root).as_posix()
        sources.append(
            SourceSpec(
                source_id=f"accepted_{hashlib.sha1(relative.encode()).hexdigest()[:12]}",
                path=str(path),
                source_family=family,
                source_group=group,
            )
        )
    return sources


def _descriptor_row(smiles: str) -> dict[str, object]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {
            "valid_smiles": False,
            "scaffold": "",
            "scaffold_ok": False,
            "heavy_atoms": -1,
            "ring_count": -1,
            "aromatic_rings": -1,
            "rotatable_bonds": -1,
            "hetero_atoms": -1,
            "formal_charge": 0,
            "fragments": -1,
            "radical_electrons": -1,
            "aromatic_atom_fraction": np.nan,
            "fraction_csp3": np.nan,
            "has_noble_gas": False,
            "has_uncommon_element": False,
            "element_set": "",
        }
    atoms = list(mol.GetAtoms())
    atomic_numbers = {atom.GetAtomicNum() for atom in atoms}
    heavy_atoms = mol.GetNumHeavyAtoms()
    scaffold_ok = True
    try:
        scaffold_mol = MurckoScaffold.GetScaffoldForMol(mol)
        scaffold_target = scaffold_mol if scaffold_mol.GetNumAtoms() else mol
        scaffold = Chem.MolToSmiles(
            scaffold_target, canonical=True, isomericSmiles=False
        )
    except RuntimeError:
        # Some PubChem structures retain inconsistent double-bond stereo flags.
        # Scaffold identity does not need stereo, so clear it deterministically.
        scaffold_ok = False
        try:
            fallback = Chem.Mol(mol)
            Chem.RemoveStereochemistry(fallback)
            for bond in fallback.GetBonds():
                bond.SetStereo(Chem.BondStereo.STEREONONE)
            scaffold_mol = MurckoScaffold.GetScaffoldForMol(fallback)
            scaffold_target = scaffold_mol if scaffold_mol.GetNumAtoms() else fallback
            scaffold = Chem.MolToSmiles(
                scaffold_target, canonical=True, isomericSmiles=False
            )
            scaffold_ok = True
        except RuntimeError:
            scaffold = ""
    return {
        "valid_smiles": True,
        "scaffold": scaffold,
        "scaffold_ok": scaffold_ok,
        "heavy_atoms": heavy_atoms,
        "ring_count": Lipinski.RingCount(mol),
        "aromatic_rings": Lipinski.NumAromaticRings(mol),
        "rotatable_bonds": Lipinski.NumRotatableBonds(mol),
        "hetero_atoms": Lipinski.NumHeteroatoms(mol),
        "formal_charge": sum(atom.GetFormalCharge() for atom in atoms),
        "fragments": len(Chem.GetMolFrags(mol)),
        "radical_electrons": sum(atom.GetNumRadicalElectrons() for atom in atoms),
        "aromatic_atom_fraction": (
            sum(atom.GetIsAromatic() for atom in atoms) / heavy_atoms
            if heavy_atoms
            else 0.0
        ),
        "fraction_csp3": rdMolDescriptors.CalcFractionCSP3(mol),
        "has_noble_gas": bool(atomic_numbers & NOBLE_GASES),
        "has_uncommon_element": bool(atomic_numbers - COMMON_ORGANIC_ELEMENTS),
        "element_set": ",".join(str(value) for value in sorted(atomic_numbers)),
    }


def _describe(smiles: Sequence[str], workers: int) -> pd.DataFrame:
    if workers <= 1:
        rows = map(_descriptor_row, smiles)
        return pd.DataFrame(rows)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = pool.map(_descriptor_row, smiles, chunksize=256)
        return pd.DataFrame(rows)


def _joint_bucket(frame: pd.DataFrame) -> pd.Series:
    gap = pd.cut(
        frame["gap"], [-np.inf, 2, 3, 4, 6, 8, np.inf],
        labels=["gap_lt2", "gap_2_3", "gap_3_4", "gap_4_6", "gap_6_8", "gap_ge8"],
        right=False,
    ).astype("string")
    mw = pd.cut(
        frame["mw"], [-np.inf, 350, 500, 700, np.inf],
        labels=["mw_lt350", "mw_350_500", "mw_500_700", "mw_ge700"],
        right=False,
    ).astype("string")
    aromatic = pd.cut(
        frame["aromatic_rings"], [-np.inf, 2, 4, np.inf],
        labels=["arom_lt2", "arom_2_3", "arom_ge4"],
        right=False,
    ).astype("string")
    flexible = pd.cut(
        frame["rotatable_bonds"], [-np.inf, 5, 10, np.inf],
        labels=["rot_lt5", "rot_5_9", "rot_ge10"],
        right=False,
    ).astype("string")
    return gap + "|" + mw + "|" + aromatic + "|" + flexible


def enrich_chunk(frame: pd.DataFrame, spec: SourceSpec, row_offset: int, workers: int) -> pd.DataFrame:
    required = {"cid", "canonical_smiles", "homo", "lumo", "gap"}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"{spec.path} is missing {sorted(missing)}")
    output = frame.copy()
    output["cid"] = output["cid"].astype("string")
    output["canonical_smiles"] = output["canonical_smiles"].astype("string")
    output["source_id"] = spec.source_id
    output["source_family"] = spec.source_family
    output["source_group"] = spec.source_group
    output["source_path"] = spec.path
    output["source_row"] = np.arange(row_offset, row_offset + len(output), dtype=np.int64)
    output["immutable"] = spec.immutable
    output["current_2m"] = spec.current_2m
    descriptors = _describe(output["canonical_smiles"].fillna("").astype(str).tolist(), workers)
    for column in descriptors:
        output[column] = descriptors[column].to_numpy()
    finite = np.isfinite(output.loc[:, TARGET_COLUMNS].to_numpy(np.float64)).all(axis=1)
    output["finite_targets"] = finite
    output["gap_identity_error"] = (
        output["gap"] - (output["lumo"] - output["homo"])
    ).abs()
    output["label_extreme"] = (
        (output["gap"] <= 0)
        | (output["gap"] > 12)
        | (output["homo"] < -20)
        | (output["homo"] > 5)
        | (output["lumo"] < -15)
        | (output["lumo"] > 10)
    )
    output["quality_ok"] = (
        output["valid_smiles"]
        & output["finite_targets"]
        & (output["gap_identity_error"] <= 1e-8)
        & ~output["label_extreme"]
        & output["scaffold_ok"]
        & (output["radical_electrons"] == 0)
        & (output["fragments"] == 1)
        & ~output["has_noble_gas"]
    )
    output["joint_bucket"] = _joint_bucket(output)
    return output


def _iter_source_chunks(spec: SourceSpec, chunk_size: int) -> Iterable[tuple[int, pd.DataFrame]]:
    usecols = lambda column: column in {
        "cid", "mw", "formula", "smiles", "canonical_smiles", *TARGET_COLUMNS
    }
    start = spec.start_row
    stop = spec.stop_row
    row_position = 0
    for chunk in pd.read_csv(spec.path, usecols=usecols, chunksize=chunk_size):
        chunk_start = row_position
        chunk_stop = row_position + len(chunk)
        row_position = chunk_stop
        if chunk_stop <= start:
            continue
        if stop is not None and chunk_start >= stop:
            break
        left = max(start, chunk_start) - chunk_start
        right = len(chunk) if stop is None else min(stop, chunk_stop) - chunk_start
        selected = chunk.iloc[left:right].reset_index(drop=True)
        if len(selected):
            yield chunk_start + left, selected


def build_ledger(
    sources: Sequence[SourceSpec],
    output_dir: Path,
    *,
    chunk_size: int = 25_000,
    workers: int = 8,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    inventory = []
    total_rows = 0
    for spec in sources:
        source_path = Path(spec.path)
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        inventory.append(
            {
                **asdict(spec),
                "size_bytes": source_path.stat().st_size,
                "sha256": sha256_file(source_path),
            }
        )
        part_index = 0
        for row_offset, chunk in _iter_source_chunks(spec, chunk_size):
            part_path = output_dir / f"{spec.source_id}_part_{part_index:05d}.parquet"
            rebuild = not part_path.exists()
            if part_path.exists():
                try:
                    cached = pd.read_parquet(
                        part_path,
                        columns=[
                            "source_id", "source_row", "cid", "canonical_smiles",
                            "scaffold_ok",
                        ],
                    )
                    expected_rows = np.arange(
                        row_offset, row_offset + len(chunk), dtype=np.int64
                    )
                    if (
                        len(cached) != len(chunk)
                        or cached["source_id"].ne(spec.source_id).any()
                        or not np.array_equal(
                            cached["source_row"].to_numpy(), expected_rows
                        )
                    ):
                        raise RuntimeError(f"Invalid cached ledger part: {part_path}")
                except Exception:
                    rebuild = True
            if rebuild:
                enriched = enrich_chunk(chunk, spec, row_offset, workers)
                temporary = part_path.with_name(f".{part_path.name}.tmp")
                enriched.to_parquet(temporary, index=False)
                os.replace(temporary, part_path)
            total_rows += len(chunk)
            part_index += 1
            atomic_json(
                {
                    "source_id": spec.source_id,
                    "completed_parts": part_index,
                    "completed_rows": row_offset + len(chunk) - spec.start_row,
                },
                output_dir / f"{spec.source_id}_progress.json",
            )
            print(
                f"ledger {spec.source_id}: part={part_index} total_rows={total_rows:,}",
                flush=True,
            )
    report = {
        "state": "complete",
        "sources": inventory,
        "source_count": len(sources),
        "ledger_rows_including_overlaps": total_rows,
        "chunk_size": chunk_size,
        "workers": workers,
    }
    atomic_json(report, output_dir.parent / "source_inventory.json")
    return report


def _category_masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    donor_acceptor = (
        frame["source_group"].isin(["conjugated_da", "hetero_dense", "topology_elements", "rare"])
        | (frame["hetero_atoms"] >= 5)
        | frame["has_uncommon_element"]
    )
    return {
        "low_gap_aromatic": (
            frame["gap"].between(2, 4, inclusive="left")
            & ((frame["aromatic_rings"] >= 4) | (frame["aromatic_atom_fraction"] >= 0.5))
        ),
        "large_flexible": (
            (frame["mw"] > 700)
            | (frame["rotatable_bonds"] >= 10)
            | frame["heavy_atoms"].between(35, 50, inclusive="both")
        ),
        "donor_acceptor_elements": donor_acceptor,
        "broad_control": (
            (frame["source_family"].isin(["general", "broad"]))
            | (frame["gap"] >= 6)
        ),
    }


def _rank_pool(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    ranked = frame.copy()
    scaffold_frequency = ranked["scaffold"].value_counts()
    bucket_frequency = ranked["joint_bucket"].value_counts()
    ranked["_scaffold_frequency"] = ranked["scaffold"].map(scaffold_frequency)
    ranked["_bucket_frequency"] = ranked["joint_bucket"].map(bucket_frequency)
    ranked["_stable_rank"] = [
        stable_hash(cid, smiles, source_id, source_row, seed=seed)
        for cid, smiles, source_id, source_row in ranked[
            ["cid", "canonical_smiles", "source_id", "source_row"]
        ].itertuples(index=False)
    ]
    return ranked.sort_values(
        ["_current_priority", "_scaffold_frequency", "_stable_rank"],
        ascending=[True, True, True],
    )


def _bucket_distance(left: str, right: str) -> int:
    orders = (
        ("gap_lt2", "gap_2_3", "gap_3_4", "gap_4_6", "gap_6_8", "gap_ge8"),
        ("mw_lt350", "mw_350_500", "mw_500_700", "mw_ge700"),
        ("arom_lt2", "arom_2_3", "arom_ge4"),
        ("rot_lt5", "rot_5_9", "rot_ge10"),
    )
    weights = (3, 2, 1, 1)
    left_parts = left.split("|")
    right_parts = right.split("|")
    if len(left_parts) != 4 or len(right_parts) != 4:
        return 10_000
    return sum(
        weight * abs(order.index(a) - order.index(b))
        for order, weight, a, b in zip(orders, weights, left_parts, right_parts)
    )


def select_repaired_manifest(
    ledger_dir: Path,
    output_path: Path,
    *,
    target_rows: int = 2_000_000,
    seed: int = 20260723,
) -> tuple[dict[str, object], pd.DataFrame]:
    paths = sorted(ledger_dir.glob("*_part_*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No ledger parts in {ledger_dir}")
    columns = [
        "cid", "canonical_smiles", "source_id", "source_family", "source_group",
        "source_path", "source_row", "immutable", "current_2m", "quality_ok",
        "gap", "mw", "heavy_atoms", "aromatic_rings", "rotatable_bonds",
        "hetero_atoms", "aromatic_atom_fraction", "has_uncommon_element",
        "scaffold", "joint_bucket",
    ]
    frame = pd.concat((pd.read_parquet(path, columns=columns) for path in paths), ignore_index=True)
    # The targeted 500K is the frozen retention contract. Preserve every row
    # while retaining quality flags in the ledger for explicit downstream audit.
    immutable = frame.loc[frame["immutable"]].copy()
    if len(immutable) != 500_000:
        raise RuntimeError(f"Expected 500,000 immutable rows, found {len(immutable):,}")
    if immutable["cid"].duplicated().any() or immutable["canonical_smiles"].duplicated().any():
        raise RuntimeError("Immutable targeted 500K contains duplicate identities")
    base_cids = set(immutable["cid"])
    base_smiles = set(immutable["canonical_smiles"])
    pool = frame.loc[
        frame["quality_ok"]
        & ~frame["immutable"]
        & ~frame["cid"].isin(base_cids)
        & ~frame["canonical_smiles"].isin(base_smiles)
    ].copy()
    pool["_current_priority"] = (~pool["current_2m"]).astype(np.int8)
    pool["_identity_rank"] = [
        stable_hash(cid, smiles, seed=seed)
        for cid, smiles in pool[["cid", "canonical_smiles"]].itertuples(index=False)
    ]
    pool = pool.sort_values(
        ["_current_priority", "_identity_rank"], ascending=[True, True]
    )
    before_dedup = len(pool)
    pool = pool.drop_duplicates("cid", keep="first")
    pool = pool.drop_duplicates("canonical_smiles", keep="first").reset_index(drop=True)
    ranked = _rank_pool(pool, seed)
    base_bucket_counts = immutable["joint_bucket"].value_counts()
    mutable_budget = target_rows - len(immutable)
    if mutable_budget != len(immutable) * 3:
        raise ValueError("Retention matching currently requires a 500K:1.5M split")
    desired = (base_bucket_counts * 3).astype(int)
    available = ranked["joint_bucket"].value_counts()
    bucket_quota = {
        bucket: min(int(required), int(available.get(bucket, 0)))
        for bucket, required in desired.items()
    }
    shortage = {
        bucket: int(required) - bucket_quota[bucket]
        for bucket, required in desired.items()
        if int(required) > bucket_quota[bucket]
    }
    surplus = {
        bucket: int(count) - bucket_quota.get(bucket, 0)
        for bucket, count in available.items()
        if int(count) > bucket_quota.get(bucket, 0)
    }
    allocations: dict[str, list[tuple[str, int, int]]] = {
        bucket: [(bucket, count, 0)]
        for bucket, count in bucket_quota.items()
        if count
    }
    redistributed_rows = 0
    weighted_distance = 0
    for target_bucket, missing in sorted(
        shortage.items(), key=lambda item: (-item[1], item[0])
    ):
        remaining = missing
        candidates = sorted(
            (
                (_bucket_distance(source_bucket, target_bucket), source_bucket)
                for source_bucket, count in surplus.items()
                if count > 0
            ),
            key=lambda item: (item[0], item[1]),
        )
        for distance, source_bucket in candidates:
            take = min(remaining, surplus[source_bucket])
            if not take:
                continue
            allocations.setdefault(source_bucket, []).append(
                (target_bucket, take, distance)
            )
            surplus[source_bucket] -= take
            remaining -= take
            redistributed_rows += take
            weighted_distance += take * distance
            if remaining == 0:
                break
        if remaining:
            raise RuntimeError(
                f"Cannot fill {remaining:,} rows near joint bucket {target_bucket}"
            )
    selected_parts = []
    for source_bucket, bucket_allocations in sorted(allocations.items()):
        candidates = ranked.loc[ranked["joint_bucket"] == source_bucket]
        offset = 0
        for target_bucket, count, distance in bucket_allocations:
            part = candidates.iloc[offset:offset + count].copy()
            if len(part) != count:
                raise RuntimeError(f"Insufficient rows in {source_bucket}")
            part["selection_category"] = (
                "joint_bucket_exact" if distance == 0 else "nearest_bucket_fill"
            )
            part["target_joint_bucket"] = target_bucket
            part["bucket_distance"] = distance
            selected_parts.append(part)
            offset += count
    mutable = pd.concat(selected_parts, ignore_index=True)
    if len(mutable) != mutable_budget:
        raise RuntimeError(
            f"Selected {len(mutable):,} mutable rows, expected {mutable_budget:,}"
        )
    immutable["selection_category"] = "immutable_targeted"
    immutable["target_joint_bucket"] = immutable["joint_bucket"]
    immutable["bucket_distance"] = 0
    selected = pd.concat([immutable, mutable], ignore_index=True)
    if len(selected) != target_rows:
        raise RuntimeError(f"Selected {len(selected):,} rows, expected {target_rows:,}")
    if selected["cid"].duplicated().any() or selected["canonical_smiles"].duplicated().any():
        raise RuntimeError("Selected manifest contains duplicate identities")
    selected["manifest_row"] = np.arange(len(selected), dtype=np.int64)
    keep = [
        "manifest_row", "cid", "canonical_smiles", "source_id", "source_family",
        "source_group", "source_path", "source_row", "selection_category",
        "immutable", "current_2m", "quality_ok",
        "target_joint_bucket", "bucket_distance",
        "gap", "mw", "heavy_atoms", "aromatic_rings", "rotatable_bonds",
        "hetero_atoms", "scaffold", "joint_bucket",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    selected.loc[:, keep].to_parquet(temporary, index=False)
    os.replace(temporary, output_path)
    report = {
        "state": "manifest_ready",
        "target_rows": target_rows,
        "immutable_rows": len(immutable),
        "immutable_quality_warning_rows": int((~immutable["quality_ok"]).sum()),
        "mutable_rows": len(mutable),
        "ledger_rows_including_overlaps": len(frame),
        "eligible_pool_rows_before_dedup": before_dedup,
        "eligible_pool_rows_after_dedup": len(pool),
        "selection_strategy": "targeted500k_joint_distribution_match",
        "joint_bucket_target_shortfall_rows": sum(shortage.values()),
        "nearest_bucket_fill_rows": redistributed_rows,
        "nearest_bucket_weighted_distance": weighted_distance,
        "exact_joint_bucket_rows": int(
            (mutable["selection_category"] == "joint_bucket_exact").sum()
        ),
        "seed": seed,
        "manifest": str(output_path),
        "manifest_sha256": sha256_file(output_path),
    }
    atomic_json(report, output_path.parent / "selection_report.json")
    return report, selected


def summarize_distribution(frame: pd.DataFrame, label: str) -> dict[str, object]:
    return {
        "dataset": label,
        "rows": len(frame),
        "unique_cid": int(frame["cid"].nunique()),
        "unique_smiles": int(frame["canonical_smiles"].nunique()),
        "unique_scaffold": int(frame["scaffold"].nunique()),
        "mean_gap": float(frame["gap"].mean()),
        "gap_lt3_fraction": float((frame["gap"] < 3).mean()),
        "gap_2_4_fraction": float(frame["gap"].between(2, 4, inclusive="left").mean()),
        "gap_ge6_fraction": float((frame["gap"] >= 6).mean()),
        "mean_mw": float(frame["mw"].mean()),
        "mw_gt700_fraction": float((frame["mw"] > 700).mean()),
        "aromatic_ge4_fraction": float((frame["aromatic_rings"] >= 4).mean()),
        "rotatable_ge10_fraction": float((frame["rotatable_bonds"] >= 10).mean()),
        "heavy_35_50_fraction": float(
            frame["heavy_atoms"].between(35, 50, inclusive="both").mean()
        ),
    }


def compare_current_and_repaired(
    ledger_dir: Path, selected: pd.DataFrame, output_path: Path
) -> pd.DataFrame:
    columns = [
        "cid", "canonical_smiles", "immutable", "current_2m", "quality_ok", "gap", "mw",
        "heavy_atoms", "aromatic_rings", "rotatable_bonds", "scaffold",
    ]
    current = pd.concat(
        (
            pd.read_parquet(path, columns=columns)
            for path in sorted(ledger_dir.glob("core_*_part_*.parquet"))
        ),
        ignore_index=True,
    )
    targeted = current.loc[current["immutable"]]
    current = current.loc[current["current_2m"]]
    summary = pd.DataFrame(
        [
            summarize_distribution(targeted, "targeted500k_reference"),
            summarize_distribution(current, "current_exact2m_quality_rows"),
            summarize_distribution(selected, "repaired2m_manifest"),
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    summary.to_csv(temporary, index=False)
    os.replace(temporary, output_path)
    return summary


def validate_manifest_references(
    ledger_dir: Path,
    selected: pd.DataFrame,
    output_path: Path,
    *,
    sample_rows: int = 2_000,
    seed: int = 20260723,
) -> dict[str, object]:
    if selected["cid"].duplicated().any() or selected["canonical_smiles"].duplicated().any():
        raise RuntimeError("Manifest identity uniqueness failed")
    if selected["source_path"].str.contains("sealed", case=False, na=False).any():
        raise RuntimeError("A sealed path entered the training manifest")
    sample = selected.sample(min(sample_rows, len(selected)), random_state=seed)[
        ["source_id", "source_row", "cid", "canonical_smiles"]
    ].copy()
    verified = 0
    for source_id, expected in sample.groupby("source_id"):
        parts = sorted(ledger_dir.glob(f"{source_id}_part_*.parquet"))
        actual = pd.concat(
            (
                pd.read_parquet(
                    part,
                    columns=["source_id", "source_row", "cid", "canonical_smiles"],
                )
                for part in parts
            ),
            ignore_index=True,
        )
        merged = expected.merge(
            actual,
            on=["source_id", "source_row", "cid", "canonical_smiles"],
            how="left",
            indicator=True,
        )
        if merged["_merge"].ne("both").any():
            raise RuntimeError(f"Manifest source reference mismatch: {source_id}")
        verified += len(merged)
    report = {
        "state": "accepted",
        "manifest_rows": len(selected),
        "unique_cid": int(selected["cid"].nunique()),
        "unique_canonical_smiles": int(selected["canonical_smiles"].nunique()),
        "immutable_rows": int(selected["immutable"].sum()),
        "quality_warning_rows": int((~selected["quality_ok"]).sum()),
        "sealed_source_rows": int(
            selected["source_path"].str.contains("sealed", case=False, na=False).sum()
        ),
        "sampled_source_references_verified": verified,
        "seed": seed,
    }
    atomic_json(report, output_path)
    return report


def materialize_manifest(
    selected: pd.DataFrame,
    output_csv: Path,
    report_path: Path,
) -> dict[str, object]:
    """Materialize an accepted manifest without mutating any source table."""
    pieces = []
    for source_path, rows in selected.groupby("source_path", sort=False):
        source = pd.read_csv(
            source_path,
            usecols=lambda column: column in TRAIN_COLUMNS,
            dtype={"cid": "string", "canonical_smiles": "string"},
        )
        positions = rows["source_row"].to_numpy(np.int64)
        if positions.max(initial=-1) >= len(source):
            raise RuntimeError(f"Source row out of range: {source_path}")
        part = source.iloc[positions].copy()
        part["manifest_row"] = rows["manifest_row"].to_numpy(np.int64)
        expected = rows[["cid", "canonical_smiles"]].reset_index(drop=True)
        actual = part[["cid", "canonical_smiles"]].reset_index(drop=True)
        if not actual.equals(expected.astype(actual.dtypes.to_dict())):
            raise RuntimeError(f"Materialization identity mismatch: {source_path}")
        pieces.append(part)
    materialized = (
        pd.concat(pieces, ignore_index=True)
        .sort_values("manifest_row")
        .reset_index(drop=True)
    )
    if not np.array_equal(
        materialized["manifest_row"].to_numpy(),
        np.arange(len(materialized), dtype=np.int64),
    ):
        raise RuntimeError("Materialized rows do not follow manifest order")
    materialized = materialized.loc[:, TRAIN_COLUMNS]
    if len(materialized) != 2_000_000:
        raise RuntimeError(f"Materialized {len(materialized):,} rows")
    if materialized["cid"].duplicated().any() or materialized["canonical_smiles"].duplicated().any():
        raise RuntimeError("Materialized identities are not unique")
    gap_error = float(
        (materialized["gap"] - (materialized["lumo"] - materialized["homo"]))
        .abs()
        .max()
    )
    if gap_error > 1e-8:
        raise RuntimeError(f"Materialized Gap identity failure: {gap_error}")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_csv.with_name(f".{output_csv.name}.tmp")
    materialized.to_csv(temporary, index=False)
    os.replace(temporary, output_csv)
    report = {
        "state": "accepted",
        "output_csv": str(output_csv),
        "rows": len(materialized),
        "unique_cid": int(materialized["cid"].nunique()),
        "unique_canonical_smiles": int(materialized["canonical_smiles"].nunique()),
        "gap_identity_max_abs_eV": gap_error,
        "sha256": sha256_file(output_csv),
        "size_bytes": output_csv.stat().st_size,
    }
    atomic_json(report, report_path)
    return report


def write_manifest_audit(
    ledger_dir: Path,
    selected: pd.DataFrame,
    output_dir: Path,
) -> dict[str, object]:
    """Persist provenance, quality, and joint-bucket evidence for review."""
    output_dir.mkdir(parents=True, exist_ok=True)
    source_mix = (
        selected.groupby(
            [
                "selection_category", "current_2m", "source_family", "source_group"
            ],
            dropna=False,
        )
        .size()
        .rename("rows")
        .reset_index()
    )
    source_mix.to_csv(output_dir / "selected_source_mix.csv", index=False)

    ledger_columns = [
        "source_family", "source_group", "current_2m", "quality_ok",
        "valid_smiles", "finite_targets", "label_extreme", "scaffold_ok",
        "radical_electrons", "fragments", "has_noble_gas",
    ]
    ledger = pd.concat(
        (
            pd.read_parquet(path, columns=ledger_columns)
            for path in sorted(ledger_dir.glob("*_part_*.parquet"))
        ),
        ignore_index=True,
    )
    quality = (
        ledger.assign(
            invalid_smiles=~ledger["valid_smiles"],
            nonfinite_targets=~ledger["finite_targets"],
            scaffold_failure=~ledger["scaffold_ok"],
            radical=(ledger["radical_electrons"] > 0),
            disconnected=(ledger["fragments"] > 1),
        )
        .groupby(["current_2m", "source_family", "source_group"], dropna=False)
        .agg(
            rows=("quality_ok", "size"),
            quality_ok=("quality_ok", "sum"),
            invalid_smiles=("invalid_smiles", "sum"),
            nonfinite_targets=("nonfinite_targets", "sum"),
            label_extreme=("label_extreme", "sum"),
            scaffold_failure=("scaffold_failure", "sum"),
            radical=("radical", "sum"),
            disconnected=("disconnected", "sum"),
            noble_gas=("has_noble_gas", "sum"),
        )
        .reset_index()
    )
    quality.to_csv(output_dir / "quality_by_source.csv", index=False)

    current_bucket = pd.concat(
        (
            pd.read_parquet(path, columns=["joint_bucket", "current_2m"])
            for path in sorted(ledger_dir.glob("core_*_part_*.parquet"))
        ),
        ignore_index=True,
    )
    current_counts = (
        current_bucket.loc[current_bucket["current_2m"], "joint_bucket"]
        .value_counts(dropna=False)
        .rename("current_rows")
    )
    repaired_counts = (
        selected["joint_bucket"].value_counts(dropna=False).rename("repaired_rows")
    )
    bucket_delta = pd.concat([current_counts, repaired_counts], axis=1).fillna(0)
    bucket_delta["current_fraction"] = bucket_delta["current_rows"] / len(current_bucket)
    bucket_delta["repaired_fraction"] = bucket_delta["repaired_rows"] / len(selected)
    bucket_delta["fraction_delta"] = (
        bucket_delta["repaired_fraction"] - bucket_delta["current_fraction"]
    )
    bucket_delta = bucket_delta.reset_index().rename(columns={"index": "joint_bucket"})
    bucket_delta.sort_values("fraction_delta", ascending=False).to_csv(
        output_dir / "joint_bucket_delta.csv", index=False
    )

    report = {
        "state": "accepted",
        "current_rows_retained": int(selected["current_2m"].sum()),
        "current_rows_replaced": int(2_000_000 - selected["current_2m"].sum()),
        "accepted_candidate_rows_added": int((~selected["current_2m"]).sum()),
        "selected_unique_scaffolds": int(selected["scaffold"].nunique()),
        "selected_quality_warning_rows": int((~selected["quality_ok"]).sum()),
        "selected_categories": {
            str(key): int(value)
            for key, value in selected["selection_category"].value_counts().items()
        },
    }
    atomic_json(report, output_dir / "manifest_audit.json")
    return report
