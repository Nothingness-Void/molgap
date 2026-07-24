"""Validate completed distillation and PCQM residual-scan artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_distillation(project: Path, run_name: str) -> dict:
    root = project / "results/phase8/distilled_2m_scnet" / run_name
    metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    embedding_manifest = json.loads(
        (root / "student_embeddings/manifest.json").read_text(encoding="utf-8")
    )
    parts = embedding_manifest["parts"]
    checks = {
        "model": sha256_file(project / metrics["model"]["path"])
        == metrics["model"]["sha256"],
        "test_predictions": sha256_file(project / metrics["test_predictions"]["path"])
        == metrics["test_predictions"]["sha256"],
        "fusion_prefix": sha256_file(root / "student_1m_embeddings_fp16.pt")
        == metrics["fusion_prefix"]["sha256"],
        "embedding_parts": len(parts) == 40
        and sum(record["rows"] for record in parts) == 2_000_000,
        "embedding_part_hashes": all(
            sha256_file(project / record["path"]) == record["sha256"]
            for record in parts
        ),
        "finite_metrics": all(
            math.isfinite(metrics["test_metrics"][target]["mae_eV"])
            for target in ("homo", "lumo", "gap", "average")
        ),
    }
    return {
        "checks": checks,
        "all_pass": all(checks.values()),
        "best_epoch": metrics["training"]["best_epoch"],
        "test": metrics["test_metrics"],
        "teacher_test": metrics["teacher_test_metrics"],
    }


def validate_pcqm(root: Path) -> dict:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    reports = sorted((root / "parts").glob("part-*.json"))
    parts = sorted((root / "parts").glob("part-*.parquet"))
    indices = []
    rows = invalid = 0
    hashes_match = finite = True
    for report_path, part_path in zip(reports, parts, strict=True):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        rows += report["rows"]
        invalid += report["invalid_rows"]
        hashes_match &= sha256_file(part_path) == report["sha256"]
        frame = pd.read_parquet(
            part_path,
            columns=["idx", "homolumogap", "teacher_gap", "teacher_abs_error"],
        )
        indices.append(frame["idx"].to_numpy(np.int64))
        finite &= bool(
            np.isfinite(
                frame[
                    ["homolumogap", "teacher_gap", "teacher_abs_error"]
                ].to_numpy()
            ).all()
        )
    source_idx = np.concatenate(indices)
    hard_path = root / "pcqm4mv2_train_hard_pool.parquet"
    hard = pd.read_parquet(hard_path)
    checks = {
        "part_counts": len(parts) == 136 and len(reports) == 136,
        "accounted_rows": rows + invalid == 3_378_606,
        "part_hashes": hashes_match,
        "idx_range": int(source_idx.min()) >= 0
        and int(source_idx.max()) < 3_378_606,
        "idx_unique": len(np.unique(source_idx)) == len(source_idx),
        "finite": finite,
        "hard_rows": len(hard) == 200_000,
        "hard_unique_canonical": bool(hard["canonical_smiles"].notna().all())
        and hard["canonical_smiles"].nunique() == len(hard),
        "hard_idx_train_only": int(hard["idx"].min()) >= 0
        and int(hard["idx"].max()) < 3_378_606,
        "hard_hash": sha256_file(hard_path) == manifest["hard_pool_sha256"],
    }
    return {
        "checks": checks,
        "all_pass": all(checks.values()),
        "rows": rows,
        "invalid": invalid,
        "hard_error": {
            "min": float(hard.teacher_abs_error.min()),
            "median": float(hard.teacher_abs_error.median()),
            "max": float(hard.teacher_abs_error.max()),
        },
        "hard_disagreement": {
            "median": float(hard.expert_gap_disagreement.median()),
            "max": float(hard.expert_gap_disagreement.max()),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--pcqm-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = {
        "distillation": {
            run_name: validate_distillation(args.project, run_name)
            for run_name in ("student_gps7", "student_gps7_w30")
        },
        "pcqm": validate_pcqm(args.pcqm_root),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name(f".{args.out.name}.tmp")
    temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
    temporary.replace(args.out)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
