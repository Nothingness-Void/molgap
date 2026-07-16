"""Build archive-r02 identity exclusions and freeze the experiment manifest."""

from __future__ import annotations

import argparse
import json
import subprocess
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import rdkit

from molgap.constants import MODELS_DIR, RAW_DIR, REPO_ROOT, RESULTS_DIR
from molgap.pubchemqc import (
    HF_CONFIG, PubChemQCFilter, fetch_dataset_metadata, molecule_identity, sha256_file,
)
from molgap.utils import ensure_dirs


OUT_DIR = RESULTS_DIR / "phase8" / "archive" / "archive-r02-pubchemqc-router"
SOURCES = [
    RAW_DIR / "phase7_chonsfcl_mw200_1000_300k.csv",
    RAW_DIR / "phase8_replacement_300k.csv",
    RAW_DIR / "phase8_expansion_500k.csv",
    RESULTS_DIR / "phase8" / "gps_arch_dualgps_common_eval_predictions.csv",
    RESULTS_DIR / "phase7" / "ood_1000" / "ood_molecules_1000.csv",
    RESULTS_DIR / "phase8" / "gps_arch_dualgps_pcqm_proxy_predictions.csv",
]
CHECKPOINTS = [
    MODELS_DIR / "phase8_gps_expansion_500k.pt",
    MODELS_DIR / "phase8_schnet_expansion_500k.pt",
    MODELS_DIR / "phase8_hybrid_fusion_expansion_500k.pt",
    MODELS_DIR / "phase8_gps_expansion_500k_depth9.pt",
    MODELS_DIR / "phase8_hybrid_fusion_expansion_500k_dualgps.pt",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def git_value(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def main() -> None:
    args = parse_args()
    ensure_dirs(args.out_dir)
    manifest_path = args.out_dir / "experiment_manifest.json"
    outputs = [
        args.out_dir / "excluded_cids.parquet",
        args.out_dir / "excluded_inchikeys.parquet",
        args.out_dir / "excluded_smiles.parquet",
        args.out_dir / "exclusion_manifest.json",
        manifest_path,
    ]
    if any(path.exists() for path in outputs) and not args.overwrite:
        raise FileExistsError("archive-r02 exclusions are frozen; pass --overwrite only to rebuild intentionally")

    source_meta = []
    raw_smiles: set[str] = set()
    cid_values: set[int] = set()
    invalid = 0
    for source in SOURCES:
        if not source.exists():
            raise FileNotFoundError(source)
        frame = pd.read_csv(source, usecols=lambda column: column in {"cid", "smiles", "canonical_smiles"})
        smiles_column = "canonical_smiles" if "canonical_smiles" in frame else "smiles"
        for row in frame.itertuples(index=False):
            cid = getattr(row, "cid", None)
            if pd.notna(cid):
                try:
                    cid_values.add(int(cid))
                except (TypeError, ValueError):
                    pass
            value = getattr(row, smiles_column)
            if isinstance(value, str) and value:
                raw_smiles.add(value)
            else:
                invalid += 1
        source_meta.append({
            "path": str(source.relative_to(REPO_ROOT)),
            "rows": int(len(frame)),
            "sha256": sha256_file(source),
        })

    # The three training CSVs overlap heavily. Deduplicate their already-canonical
    # strings before the expensive RDKit/InChI pass, then parallelize unique rows.
    identity_rows: dict[str, dict] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for identity in pool.map(molecule_identity, sorted(raw_smiles), chunksize=500):
            if identity is None:
                invalid += 1
                continue
            canonical, inchikey = identity
            identity_rows.setdefault(canonical, {"canonical_smiles": canonical, "inchikey": inchikey})
    identities = pd.DataFrame(identity_rows.values()).sort_values("canonical_smiles")
    cids = pd.DataFrame({"cid": sorted(cid_values)})
    inchikeys = pd.DataFrame({"inchikey": sorted(set(identities["inchikey"]) - {""})})
    smiles = identities[["canonical_smiles"]]
    cids.to_parquet(outputs[0], index=False)
    inchikeys.to_parquet(outputs[1], index=False)
    smiles.to_parquet(outputs[2], index=False)

    output_hashes = {path.name: sha256_file(path) for path in outputs[:3]}
    exclusion_manifest = {
        "sources": source_meta,
        "counts": {
            "cids": len(cids), "inchikeys": len(inchikeys),
            "canonical_smiles": len(smiles), "invalid_smiles_rows": invalid,
        },
        "output_sha256": output_hashes,
    }
    outputs[3].write_text(json.dumps(exclusion_manifest, indent=2), encoding="utf-8")

    checkpoints = {
        str(path.relative_to(REPO_ROOT)): sha256_file(path) for path in CHECKPOINTS
    }
    dataset = fetch_dataset_metadata()
    dataset["config"] = HF_CONFIG
    dataset["smiles_field"] = "pubchem-isomeric-smiles"
    dataset["label_fields"] = [
        f"energy-{spin}-{target}"
        for spin in ("alpha", "beta") for target in ("homo", "lumo", "gap")
    ]
    manifest = {
        "experiment": "archive-r02 PubChemQC Learned Router",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_head_before_archive_r02": git_value("rev-parse", "HEAD"),
        "git_worktree_dirty": bool(git_value("status", "--porcelain")),
        "checkpoints_sha256": checkpoints,
        "dataset": dataset,
        "filter": PubChemQCFilter().to_dict(),
        "identity": {
            "canonicalization": "RDKit canonical isomeric SMILES; no salt stripping",
            "rdkit_version": rdkit.__version__,
            "dedup_keys": ["cid", "canonical_smiles", "inchikey"],
        },
        "geometry": {
            "inference": "ETKDGv3 + MMFF, max 200 iterations",
            "pubchem_pm6_coordinates_used": False,
            "random_seed": 42,
        },
        "legacy_filter_note": (
            "P7/P8 CSV fetches enforced CHONSFCl, MW 200-1000, positive alpha Gap, "
            "but did not retain state/charge/multiplicity. archive-r02 adds strict S0/neutral/"
            "singlet and alpha-beta consistency checks for Router-label quality."
        ),
        "exclusion_manifest_sha256": sha256_file(outputs[3]),
        "exclusion_outputs_sha256": output_hashes,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(exclusion_manifest["counts"], indent=2))
    print(f"wrote frozen manifest: {manifest_path}")


if __name__ == "__main__":
    main()
