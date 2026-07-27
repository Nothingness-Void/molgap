"""Select a label-blind 20k Oracle probe from the independent candidate pool."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from molgap.constants import RAW_DIR, RESULTS_DIR
from molgap.pubchemqc import sha256_file
from molgap.router_sampling import select_descriptor_diverse
from molgap.utils import ensure_dirs


INPUT_DIR = RAW_DIR / "archive-r02-router-candidates"
OUT_DIR = RESULTS_DIR / "phase8" / "archive" / "archive-r02-pubchemqc-router"
FEATURES = [
    "mw", "heavy_atoms", "aromatic_rings", "rotatable_bonds", "tpsa", "logp",
    "formal_charge", "has_s", "has_p", "has_cl", "has_f",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    parser.add_argument("--out", type=Path, default=OUT_DIR / "oracle_probe_20k.parquet")
    parser.add_argument("--manifest-out", type=Path, default=OUT_DIR / "oracle_probe_sampling_manifest.json")
    parser.add_argument("--random-n", type=int, default=10_000)
    parser.add_argument("--diverse-n", type=int, default=10_000)
    parser.add_argument("--clusters", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (args.out.exists() or args.manifest_out.exists()) and not args.overwrite:
        raise FileExistsError("Oracle probe already exists; pass --overwrite intentionally")
    parts = sorted(args.input_dir.glob("part-*.parquet"))
    if not parts:
        raise FileNotFoundError(args.input_dir)
    pool = pd.concat([pd.read_parquet(path) for path in parts], ignore_index=True)
    pool = pool.drop_duplicates(["cid", "canonical_smiles", "inchikey"]).reset_index(drop=True)
    total = args.random_n + args.diverse_n
    if len(pool) < total:
        raise ValueError(f"Need {total:,} candidates, found {len(pool):,}")

    rng = np.random.default_rng(args.seed)
    random_idx = rng.choice(len(pool), size=args.random_n, replace=False)
    random_set = set(random_idx.tolist())
    remaining_idx = np.asarray([i for i in range(len(pool)) if i not in random_set])
    selected, probabilities = select_descriptor_diverse(
        pool, remaining_idx, features=FEATURES, n_select=args.diverse_n,
        n_clusters=args.clusters, seed=args.seed,
    )

    random_frame = pool.iloc[random_idx].copy()
    random_frame["sampling_source"] = "random"
    random_frame["sampling_probability"] = args.random_n / len(pool)
    diverse_frame = pool.iloc[selected].copy()
    diverse_frame["sampling_source"] = "descriptor_diverse"
    diverse_frame["sampling_probability"] = [probabilities[int(index)] for index in selected]
    probe = pd.concat([random_frame, diverse_frame], ignore_index=True)
    probe["sampling_weight"] = 1.0 / probe["sampling_probability"]
    probe.insert(0, "probe_idx", np.arange(len(probe), dtype=np.int64))
    ensure_dirs(args.out.parent)
    probe.to_parquet(args.out, index=False)

    manifest = {
        "candidate_pool_n": int(len(pool)),
        "probe_n": int(len(probe)),
        "random_n": int((probe.sampling_source == "random").sum()),
        "descriptor_diverse_n": int((probe.sampling_source == "descriptor_diverse").sum()),
        "selection_uses_b3lyp_labels": False,
        "features": FEATURES,
        "clusters": args.clusters,
        "seed": args.seed,
        "candidate_cid_range": [int(pool.cid.min()), int(pool.cid.max())],
        "probe_cid_range": [int(probe.cid.min()), int(probe.cid.max())],
        "output_sha256": sha256_file(args.out),
    }
    args.manifest_out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
