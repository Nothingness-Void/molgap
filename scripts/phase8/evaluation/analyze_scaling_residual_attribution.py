"""Attribute Phase 8 scaling regressions with paired molecular residuals."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from molgap.residual_attribution import TARGETS, analyze_comparison, molecular_descriptors


ROOT = Path("results/phase8/scaling_residual_attribution")


def atomic_json(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def atomic_csv(value: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    value.to_csv(temporary, index=False)
    os.replace(temporary, path)


def main() -> None:
    comparisons = {
        "pure2d_exact2m": {
            "path": Path("results/phase8/multi2d_2m_dev_eval/common_predictions.csv"),
            "baseline": "anchor",
            "candidates": [
                "repair",
                "coverage2m",
                "incumbent",
                "anchor_coverage",
                "tri_expert",
            ],
        },
        "pure2d_add_coverage_to_incumbent": {
            "path": Path("results/phase8/multi2d_2m_dev_eval/common_predictions.csv"),
            "baseline": "incumbent",
            "candidates": ["tri_expert"],
        },
        "one_million_fusion": {
            "path": Path("results/phase8/expansion_1m/common_eval_kaggle_predictions.csv"),
            "baseline": "routed_v4",
            "candidates": ["candidate_1m"],
        },
        "distilled_exact2m": {
            "path": Path("results/phase8/distilled_2m_external_eval/common_predictions.csv"),
            "baseline": "teacher",
            "candidates": ["student_w30"],
        },
    }
    summary = {}
    all_strata = []
    worst_rows = []
    for name, config in comparisons.items():
        frame = pd.read_csv(config["path"])
        report, strata = analyze_comparison(
            frame,
            baseline=config["baseline"],
            candidates=config["candidates"],
        )
        report["source"] = str(config["path"])
        summary[name] = report
        strata.insert(0, "comparison", name)
        all_strata.append(strata)
        descriptors = molecular_descriptors(frame.smiles)
        for candidate in config["candidates"]:
            baseline_error = sum(
                (frame[f"{config['baseline']}_{target}"] - frame[target]).abs()
                for target in TARGETS
            ) / len(TARGETS)
            candidate_error = sum(
                (frame[f"{candidate}_{target}"] - frame[target]).abs()
                for target in TARGETS
            ) / len(TARGETS)
            disagreement = sum(
                (frame[f"{candidate}_{target}"] - frame[f"{config['baseline']}_{target}"]).abs()
                for target in TARGETS
            ) / len(TARGETS)
            rows = frame.loc[
                :,
                [column for column in ("eval_set", "cid", "smiles", *TARGETS) if column in frame],
            ].copy()
            rows = pd.concat([rows.reset_index(drop=True), descriptors], axis=1)
            rows.insert(0, "candidate", candidate)
            rows.insert(0, "baseline", config["baseline"])
            rows.insert(0, "comparison", name)
            rows["average_error_delta_eV"] = candidate_error - baseline_error
            rows["prediction_disagreement_eV"] = disagreement
            worst_rows.append(rows.nlargest(100, "average_error_delta_eV"))
    atomic_json(summary, ROOT / "attribution.json")
    atomic_csv(pd.concat(all_strata, ignore_index=True), ROOT / "strata.csv")
    atomic_csv(pd.concat(worst_rows, ignore_index=True), ROOT / "worst_rows.csv")


if __name__ == "__main__":
    main()
