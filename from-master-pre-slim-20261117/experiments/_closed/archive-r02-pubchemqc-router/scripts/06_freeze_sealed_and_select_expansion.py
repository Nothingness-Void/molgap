"""Freeze archive-r02 sealed sets, then select 30k additional development rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from molgap.constants import RAW_DIR, RESULTS_DIR
from molgap.pubchemqc import sha256_file
from molgap.router_sampling import compute_scaffold_keys, select_descriptor_diverse
from molgap.utils import ensure_dirs


RESULTS = RESULTS_DIR / "phase8" / "archive" / "archive-r02-pubchemqc-router"
POOL_DIR = RAW_DIR / "archive-r02-router-candidates"
FEATURES = [
    "mw", "heavy_atoms", "aromatic_rings", "rotatable_bonds", "tpsa", "logp",
    "formal_charge", "has_s", "has_p", "has_cl", "has_f",
]
LABELS = ["homo", "lumo", "gap"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-dir", type=Path, default=POOL_DIR)
    parser.add_argument("--probe", type=Path, default=RESULTS / "oracle_probe_20k.parquet")
    parser.add_argument("--sealed-random-n", type=int, default=20_000)
    parser.add_argument("--sealed-hard-n", type=int, default=10_000)
    parser.add_argument("--dev-random-n", type=int, default=15_000)
    parser.add_argument("--dev-diverse-n", type=int, default=15_000)
    parser.add_argument("--clusters", type=int, default=500)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=RESULTS)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def hard_selection(frame: pd.DataFrame, indices: np.ndarray, n: int, seed: int):
    work = frame.loc[indices].copy()
    rng = np.random.default_rng(seed)
    work["tie"] = rng.random(len(work)) * 1e-9
    work["score_large_aromatic"] = (
        work.aromatic_rings.rank(pct=True) + work.heavy_atoms.rank(pct=True) + work.tie
    )
    work["score_flexible"] = work.rotatable_bonds.rank(pct=True) + work.tie
    work["score_high_mw"] = work.mw.rank(pct=True) + work.tie
    work["score_property_extreme"] = np.maximum(
        work.tpsa.rank(pct=True), work.logp.abs().rank(pct=True)
    ) + work.tie
    work["score_rare_elements"] = (
        2 * work.has_s + 2 * work.has_cl + work.has_f + work.tie
    )
    buckets = [
        ("large_aromatic", "score_large_aromatic"),
        ("flexible", "score_flexible"),
        ("high_mw", "score_high_mw"),
        ("property_extreme", "score_property_extreme"),
        ("rare_elements", "score_rare_elements"),
    ]
    base, extra = divmod(n, len(buckets))
    chosen: list[int] = []
    source: dict[int, str] = {}
    for rank, (name, score) in enumerate(buckets):
        quota = base + int(rank < extra)
        available = work.drop(index=chosen, errors="ignore")
        rows = available.nlargest(quota, score).index.to_numpy(dtype=np.int64)
        chosen.extend(rows.tolist())
        source.update({int(index): name for index in rows})
    return np.asarray(chosen, dtype=np.int64), source


def write_sealed(frame: pd.DataFrame, prefix: Path) -> dict:
    input_columns = [column for column in frame.columns if column not in LABELS]
    inputs = prefix.with_name(prefix.name + "_inputs.parquet")
    labels = prefix.with_name(prefix.name + "_labels.parquet")
    frame[input_columns].to_parquet(inputs, index=False)
    frame[["sealed_idx", "cid", *LABELS]].to_parquet(labels, index=False)
    return {inputs.name: sha256_file(inputs), labels.name: sha256_file(labels)}


def main() -> None:
    args = parse_args()
    outputs = [
        args.out_dir / "sealed_random_inputs.parquet",
        args.out_dir / "sealed_random_labels.parquet",
        args.out_dir / "sealed_hard_inputs.parquet",
        args.out_dir / "sealed_hard_labels.parquet",
        args.out_dir / "development_expansion_30k.parquet",
        args.out_dir / "sealed_split_manifest.json",
    ]
    if any(path.exists() for path in outputs) and not args.overwrite:
        raise FileExistsError("sealed split is frozen; pass --overwrite only intentionally")
    ensure_dirs(args.out_dir)
    parts = sorted(args.pool_dir.glob("part-*.parquet"))
    pool = pd.concat([pd.read_parquet(path) for path in parts], ignore_index=True)
    pool = pool.drop_duplicates(["cid", "canonical_smiles", "inchikey"]).reset_index(drop=True)
    probe = pd.read_parquet(args.probe)
    probe_smiles = set(probe.canonical_smiles)

    print(f"Computing scaffolds for {len(pool):,} candidates", flush=True)
    pool["scaffold"] = compute_scaffold_keys(pool.canonical_smiles.tolist(), args.workers)
    probe_scaffolds = set(pool.loc[pool.canonical_smiles.isin(probe_smiles), "scaffold"])
    non_probe = ~pool.canonical_smiles.isin(probe_smiles)
    eligible = pool.index[non_probe & ~pool.scaffold.isin(probe_scaffolds)].to_numpy()
    rng = np.random.default_rng(args.seed)

    random_idx = rng.choice(eligible, size=args.sealed_random_n, replace=False)
    random_scaffolds = set(pool.loc[random_idx, "scaffold"])
    hard_candidates = np.asarray(
        [index for index in eligible if pool.at[index, "scaffold"] not in random_scaffolds],
        dtype=np.int64,
    )
    hard_idx, hard_sources = hard_selection(
        pool, hard_candidates, args.sealed_hard_n, args.seed + 1
    )
    hard_scaffolds = set(pool.loc[hard_idx, "scaffold"])
    reserved_scaffolds = random_scaffolds | hard_scaffolds
    dev_candidates = pool.index[
        non_probe & ~pool.scaffold.isin(reserved_scaffolds)
    ].to_numpy(dtype=np.int64)

    dev_random_idx = rng.choice(dev_candidates, size=args.dev_random_n, replace=False)
    dev_random_set = set(dev_random_idx.tolist())
    diverse_candidates = np.asarray(
        [index for index in dev_candidates if index not in dev_random_set], dtype=np.int64
    )
    dev_diverse_idx, probabilities = select_descriptor_diverse(
        pool, diverse_candidates, features=FEATURES, n_select=args.dev_diverse_n,
        n_clusters=args.clusters, seed=args.seed + 2,
    )

    sealed_random = pool.loc[random_idx].copy().reset_index(drop=True)
    sealed_random.insert(0, "sealed_idx", np.arange(len(sealed_random), dtype=np.int64))
    sealed_random["sealed_source"] = "random"
    sealed_hard = pool.loc[hard_idx].copy().reset_index(drop=True)
    sealed_hard.insert(
        0, "sealed_idx", np.arange(len(sealed_random), len(sealed_random) + len(sealed_hard))
    )
    sealed_hard["sealed_source"] = [hard_sources[int(index)] for index in hard_idx]
    hashes = {}
    hashes.update(write_sealed(sealed_random, args.out_dir / "sealed_random"))
    hashes.update(write_sealed(sealed_hard, args.out_dir / "sealed_hard"))

    dev_random = pool.loc[dev_random_idx].copy()
    dev_random["sampling_source"] = "expansion_random"
    dev_random["sampling_probability"] = args.dev_random_n / len(dev_candidates)
    dev_diverse = pool.loc[dev_diverse_idx].copy()
    dev_diverse["sampling_source"] = "expansion_descriptor_diverse"
    dev_diverse["sampling_probability"] = [probabilities[int(i)] for i in dev_diverse_idx]
    development = pd.concat([dev_random, dev_diverse], ignore_index=True)
    development["sampling_weight"] = 1.0 / development.sampling_probability
    development.insert(0, "probe_idx", np.arange(20_000, 20_000 + len(development)))
    development.to_parquet(outputs[4], index=False)
    hashes[outputs[4].name] = sha256_file(outputs[4])

    manifest = {
        "pool_n": len(pool),
        "prior_probe_n": len(probe),
        "scaffold_rule": "Bemis-Murcko; acyclic rows use ACYCLIC::<canonical_smiles>",
        "selection_uses_b3lyp_labels": False,
        "counts": {
            "probe_scaffolds": len(probe_scaffolds),
            "eligible_after_probe_scaffold_exclusion": len(eligible),
            "sealed_random": len(sealed_random),
            "sealed_hard": len(sealed_hard),
            "reserved_sealed_scaffolds": len(reserved_scaffolds),
            "development_candidates_after_sealed_exclusion": len(dev_candidates),
            "development_expansion": len(development),
        },
        "hard_sources": sealed_hard.sealed_source.value_counts().to_dict(),
        "seed": args.seed,
        "artifacts_sha256": hashes,
        "sealed_policy": "Do not run or inspect sealed metrics until Router policy is locked.",
    }
    outputs[5].write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
