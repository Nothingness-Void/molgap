"""Build immutable scaffold-disjoint OOF plans without submitting training."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def assign_scaffold_folds(scaffolds: pd.Series, n_folds: int) -> np.ndarray:
    """Greedily balance whole scaffold groups across deterministic folds."""
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    if scaffolds.isna().any():
        raise ValueError("Scaffold keys must be complete")
    counts = scaffolds.value_counts(sort=False)
    ordered = sorted(
        counts.items(),
        key=lambda item: (
            -int(item[1]),
            hashlib.sha256(str(item[0]).encode()).hexdigest(),
        ),
    )
    fold_rows = [0] * n_folds
    scaffold_to_fold: dict[str, int] = {}
    for scaffold, count in ordered:
        fold = min(range(n_folds), key=lambda value: (fold_rows[value], value))
        scaffold_to_fold[str(scaffold)] = fold
        fold_rows[fold] += int(count)
    return scaffolds.astype(str).map(scaffold_to_fold).to_numpy(dtype=np.int16)


def build_oof_plan(
    source_csv: Path,
    repaired_manifest: Path,
    out_dir: Path,
    *,
    n_folds: int = 5,
    seed: int = 42,
) -> dict:
    manifest = pd.read_parquet(
        repaired_manifest,
        columns=["manifest_row", "cid", "canonical_smiles", "scaffold"],
    ).sort_values("manifest_row")
    labels = pd.read_csv(
        source_csv,
        usecols=["cid", "canonical_smiles", "homo", "lumo", "gap"],
    )
    if len(labels) != len(manifest):
        raise ValueError("Source CSV and repaired manifest row counts differ")
    if not np.array_equal(
        labels.cid.astype(str).to_numpy(),
        manifest.cid.astype(str).to_numpy(),
    ):
        raise ValueError("Source CSV and repaired manifest CID order differs")
    if not np.array_equal(
        labels.canonical_smiles.astype(str).to_numpy(),
        manifest.canonical_smiles.astype(str).to_numpy(),
    ):
        raise ValueError("Source CSV and repaired manifest SMILES order differs")
    target_values = labels[["homo", "lumo", "gap"]].to_numpy(dtype=np.float64)
    if not np.isfinite(target_values).all():
        raise ValueError("OOF source labels are not finite")

    folds = assign_scaffold_folds(manifest.scaffold, n_folds)
    fold_frame = pd.DataFrame(
        {
            "source_idx": manifest.manifest_row.to_numpy(dtype=np.int64),
            "fold": folds,
            "cid": labels.cid.astype(str),
            "canonical_smiles": labels.canonical_smiles.astype(str),
            "homo": target_values[:, 0],
            "lumo": target_values[:, 1],
            "gap": target_values[:, 2],
        }
    )
    fold_path = out_dir / "folds.parquet"
    _atomic_parquet(fold_frame, fold_path)

    fold_reports = []
    scaffold_sets = []
    for fold in range(n_folds):
        mask = folds == fold
        fold_scaffolds = set(manifest.loc[mask, "scaffold"].astype(str))
        scaffold_sets.append(fold_scaffolds)
        fold_reports.append(
            {
                "fold": fold,
                "rows": int(mask.sum()),
                "unique_cids": int(fold_frame.loc[mask, "cid"].nunique()),
                "unique_smiles": int(
                    fold_frame.loc[mask, "canonical_smiles"].nunique()
                ),
                "unique_scaffolds": len(fold_scaffolds),
                "target_mean_eV": dict(
                    zip(
                        ("homo", "lumo", "gap"),
                        target_values[mask].mean(axis=0).tolist(),
                        strict=True,
                    )
                ),
                "target_std_eV": dict(
                    zip(
                        ("homo", "lumo", "gap"),
                        target_values[mask].std(axis=0).tolist(),
                        strict=True,
                    )
                ),
            }
        )
    overlap = {
        f"{left}_{right}": len(scaffold_sets[left] & scaffold_sets[right])
        for left in range(n_folds)
        for right in range(left + 1, n_folds)
    }
    if any(overlap.values()):
        raise RuntimeError(f"Scaffold leakage detected: {overlap}")

    prediction_contract = {
        "format": "parquet",
        "required_columns": [
            "source_idx",
            "fold",
            "cid",
            "canonical_smiles",
            "homo",
            "lumo",
            "gap",
            "gps7_homo",
            "gps7_lumo",
            "gps7_gap",
            "gps9_homo",
            "gps9_lumo",
            "gps9_gap",
        ],
        "identity_key": ["source_idx", "cid", "canonical_smiles"],
        "one_row_per_source": True,
        "gain_label_formula": {
            "per_target": "abs(gps7_target - target) - abs(gps9_target - target)",
            "average": "mean(gain_homo, gain_lumo, gain_gap)",
            "interpretation": "positive means GPS9 improves over GPS7",
        },
        "training_rule": "predictions for each row must come only from its held-out fold",
    }
    _atomic_json(out_dir / "prediction_contract.json", prediction_contract)
    scnet_plan = {
        "status": "prepared_not_submitted",
        "array": {"folds": list(range(n_folds)), "models": ["gps7", "gps9"]},
        "jobs": [
            {
                "fold": fold,
                "model": model,
                "train_folds": [value for value in range(n_folds) if value != fold],
                "holdout_fold": fold,
                "split_seed": seed,
                "checkpoint_policy": "atomic best and last plus resumable epoch state",
            }
            for fold in range(n_folds)
            for model in ("gps7", "gps9")
        ],
        "submitted": False,
        "sealed_20k_used": False,
    }
    _atomic_json(out_dir / "scnet_jobs.json", scnet_plan)
    report = {
        "experiment": "repaired_2m_gps7_gps9_oof_plan",
        "status": "folds_frozen_jobs_prepared_not_submitted",
        "seed": seed,
        "n_folds": n_folds,
        "rows": len(fold_frame),
        "source_csv": source_csv.as_posix(),
        "source_csv_sha256": sha256_file(source_csv),
        "source_manifest": repaired_manifest.as_posix(),
        "source_manifest_sha256": sha256_file(repaired_manifest),
        "folds_path": fold_path.as_posix(),
        "folds_sha256": sha256_file(fold_path),
        "folds": fold_reports,
        "scaffold_overlap": overlap,
        "prediction_contract": "prediction_contract.json",
        "scnet_jobs": "scnet_jobs.json",
        "router_training_authorized": False,
        "sealed_20k_used": False,
        "production_registry_changed": False,
    }
    _atomic_json(out_dir / "manifest.json", report)
    return report
