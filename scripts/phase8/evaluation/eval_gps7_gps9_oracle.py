"""Evaluate the repaired-2M GPS7/GPS9 hard-expert Oracle ceiling."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from molgap.hierarchical_oracle import hierarchical_oracle_analysis

TARGETS = ("homo", "lumo", "gap")
KEYS = ("eval_set", "cid", "smiles")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _delta(report: dict, method: str, target: str | None = None) -> float:
    methods = report["methods"]
    if target is None:
        return float(
            methods[method]["average_mae_eV"]
            - methods["base"]["average_mae_eV"]
        )
    return float(
        methods[method]["targets"][target]["mae_eV"]
        - methods["base"]["targets"][target]["mae_eV"]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gps7", type=Path, required=True)
    parser.add_argument("--gps9", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    gps7 = pd.read_csv(args.gps7)
    gps9 = pd.read_csv(args.gps9)
    merged = gps7.merge(
        gps9,
        on=list(KEYS),
        how="inner",
        validate="one_to_one",
        suffixes=("_gps7_source", "_gps9_source"),
    )
    if len(merged) != len(gps7) or len(merged) != len(gps9):
        raise RuntimeError(
            f"Prediction alignment failed: gps7={len(gps7)} "
            f"gps9={len(gps9)} aligned={len(merged)}"
        )
    for target in TARGETS:
        if (
            merged[f"{target}_gps7_source"]
            - merged[f"{target}_gps9_source"]
        ).abs().max() > 1e-9:
            raise RuntimeError(f"{target} labels differ after alignment")

    reports = {}
    frames = [("common", merged)]
    frames.extend(merged.groupby("eval_set", sort=True))
    for scope, frame in frames:
        truth = frame[
            [f"{target}_gps7_source" for target in TARGETS]
        ].to_numpy()
        base = frame[
            [f"repaired_2m_d_gps7_seed42_{target}" for target in TARGETS]
        ].to_numpy()
        expert = frame[
            [f"repaired_2m_d_gps9_seed42_{target}" for target in TARGETS]
        ].to_numpy()
        report, _ = hierarchical_oracle_analysis(
            truth,
            base,
            expert,
            target_names=TARGETS,
            base_encoder_passes=1.0,
            expert_encoder_passes=1.0,
        )
        reports[str(scope)] = report

    gate = {
        "required_p8_hard_improvement_eV": 0.001,
        "maximum_common_regression_eV": 0.0005,
        "maximum_ood_regression_eV": 0.0005,
        "p8_hard_10pct_improvement_eV": -_delta(
            reports["p8_targeted_hard"], "switch_10pct"
        ),
        "common_10pct_regression_eV": _delta(
            reports["common"], "switch_10pct"
        ),
        "ood_10pct_regression_eV": _delta(
            reports["ood1000"], "switch_10pct"
        ),
    }
    gate["passed"] = bool(
        gate["p8_hard_10pct_improvement_eV"]
        >= gate["required_p8_hard_improvement_eV"]
        and gate["common_10pct_regression_eV"]
        <= gate["maximum_common_regression_eV"]
        and gate["ood_10pct_regression_eV"]
        <= gate["maximum_ood_regression_eV"]
    )
    gate["decision"] = (
        "prepare_scaffold_disjoint_oof_gain_labels"
        if gate["passed"]
        else "close_gps7_gps9_hard_router"
    )

    _atomic_json(
        {
            "experiment": "repaired_2m_gps7_gps9_oracle",
            "inputs": {
                "gps7": str(args.gps7),
                "gps7_sha256": _sha256(args.gps7),
                "gps9": str(args.gps9),
                "gps9_sha256": _sha256(args.gps9),
            },
            "alignment": {
                "gps7_rows": len(gps7),
                "gps9_rows": len(gps9),
                "aligned_rows": len(merged),
            },
            "reports": reports,
            "gate": gate,
            "external_labels_training_eligible": False,
            "sealed_20k_used": False,
            "registry_changed": False,
        },
        args.out,
    )


if __name__ == "__main__":
    main()
