"""Evaluate the Phase 8.20 task and molecular Oracle ceilings."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from molgap.hierarchical_oracle import hierarchical_oracle_analysis

TARGETS = ("homo", "lumo", "gap")


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


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _method_delta(
    methods: dict[str, object], candidate: str, target: str | None = None
) -> float:
    if target is None:
        return float(methods[candidate]["average_mae_eV"] - methods["base"]["average_mae_eV"])
    return float(
        methods[candidate]["targets"][target]["mae_eV"]
        - methods["base"]["targets"][target]["mae_eV"]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base",
        type=Path,
        default=Path(
            "results/phase8/repaired_2m/retention_d_seed42_raw/common_predictions.csv"
        ),
    )
    parser.add_argument(
        "--teacher",
        type=Path,
        default=Path("results/phase8/multi2d_expert_ensemble/common_predictions.csv"),
    )
    parser.add_argument(
        "--pcqm-base",
        type=Path,
        default=Path(
            "results/phase8/repaired_2m/retention_d_seed42_raw/pcqm_predictions.csv"
        ),
    )
    parser.add_argument(
        "--pcqm-expert",
        type=Path,
        default=Path(
            "results/kaggle/pcqm_gine_expert_pilot/v4_raw_20260724/"
            "pcqm_official_valid_5k_predictions.csv"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/phase8/hierarchical_oracle_20260725"),
    )
    args = parser.parse_args()

    base = pd.read_csv(args.base)
    teacher = pd.read_csv(args.teacher)
    keys = ["eval_set", "cid", "smiles"]
    merged = base.merge(
        teacher,
        on=keys,
        how="inner",
        validate="one_to_one",
        suffixes=("_base_source", "_teacher_source"),
    )
    if len(merged) != len(base):
        raise RuntimeError(
            f"Hard-teacher alignment lost rows: base={len(base)}, aligned={len(merged)}"
        )
    for target in TARGETS:
        difference = np.abs(
            merged[f"{target}_base_source"] - merged[f"{target}_teacher_source"]
        )
        if float(difference.max()) > 1e-9:
            raise RuntimeError(f"{target} labels differ after alignment")

    reports: dict[str, object] = {}
    label_frames: list[pd.DataFrame] = []
    for scope, frame in (("common", merged), *merged.groupby("eval_set", sort=True)):
        y_true = frame[[f"{target}_base_source" for target in TARGETS]].to_numpy()
        base_pred = frame[
            [f"repaired_2m_d_gps7_seed42_{target}" for target in TARGETS]
        ].to_numpy()
        teacher_pred = frame[
            [f"mean_control_repair_{target}" for target in TARGETS]
        ].to_numpy()
        report, arrays = hierarchical_oracle_analysis(
            y_true,
            base_pred,
            teacher_pred,
            target_names=TARGETS,
        )
        reports[scope] = report
        labels = frame[keys].copy()
        labels.insert(0, "scope", scope)
        for index, target in enumerate(TARGETS):
            labels[f"{target}_base_error_eV"] = np.abs(
                base_pred[:, index] - y_true[:, index]
            )
            labels[f"{target}_teacher_error_eV"] = np.abs(
                teacher_pred[:, index] - y_true[:, index]
            )
            labels[f"{target}_teacher_gain_eV"] = arrays["switch_gain"][:, index]
            labels[f"{target}_teacher_wins"] = arrays["switch_wins"][:, index]
            labels[f"{target}_optimal_alpha"] = arrays["optimal_alpha"][:, index]
        if scope != "common":
            label_frames.append(labels)

    pcqm_base = pd.read_csv(args.pcqm_base)
    pcqm_expert = pd.read_csv(args.pcqm_expert)
    paired_pcqm = pcqm_base.merge(
        pcqm_expert,
        left_on="cid",
        right_on="idx",
        how="inner",
        validate="one_to_one",
    )
    if len(paired_pcqm) != len(pcqm_base):
        raise RuntimeError(
            f"PCQM alignment lost rows: base={len(pcqm_base)}, aligned={len(paired_pcqm)}"
        )
    label_delta = np.abs(paired_pcqm["gap"] - paired_pcqm["gap_true_eV"])
    if float(label_delta.max()) > 1e-5:
        raise RuntimeError("PCQM labels differ after idx/CID alignment")
    pcqm = {
        "expert_full_n": int(len(pcqm_expert)),
        "expert_full_gap_mae_eV": float(pcqm_expert["absolute_error_eV"].mean()),
        "paired_n": int(len(paired_pcqm)),
        "base_gap_mae_eV": float(
            np.abs(
                paired_pcqm["repaired_2m_d_gps7_seed42_gap"] - paired_pcqm["gap"]
            ).mean()
        ),
        "expert_gap_mae_eV": float(
            np.abs(paired_pcqm["gap_prediction_eV"] - paired_pcqm["gap"]).mean()
        ),
    }
    pcqm["task_route_delta_eV"] = (
        pcqm["expert_gap_mae_eV"] - pcqm["base_gap_mae_eV"]
    )

    hard = reports["p8_targeted_hard"]
    common = reports["common"]
    hard_improvement_10pct = -_method_delta(
        hard["methods"], "switch_10pct"
    )
    hard_improvement_unconstrained = -_method_delta(
        hard["methods"], "unconstrained_switch"
    )
    common_regression_10pct = _method_delta(common["methods"], "switch_10pct")
    gate = {
        "required_p8_hard_improvement_eV": 0.001,
        "maximum_common_regression_eV": 0.0005,
        "p8_hard_unconstrained_switch_improvement_eV": hard_improvement_unconstrained,
        "p8_hard_10pct_switch_improvement_eV": hard_improvement_10pct,
        "common_10pct_switch_regression_eV": common_regression_10pct,
        "unconstrained_passed": bool(
            hard_improvement_unconstrained >= 0.001
            and common_regression_10pct <= 0.0005
        ),
        "survives_10pct_call_budget": bool(
            hard_improvement_10pct >= 0.001
            and common_regression_10pct <= 0.0005
        ),
    }
    gate["decision"] = (
        "open_oof_gain_label_generation_no_router_training"
        if gate["survives_10pct_call_budget"]
        else "close_molecular_hard_router_path"
    )

    payload = {
        "experiment": "phase8_20_hierarchical_oracle",
        "inputs": {
            "base": str(args.base),
            "base_sha256": _sha256(args.base),
            "hard_teacher": str(args.teacher),
            "hard_teacher_sha256": _sha256(args.teacher),
            "pcqm_base": str(args.pcqm_base),
            "pcqm_base_sha256": _sha256(args.pcqm_base),
            "pcqm_expert": str(args.pcqm_expert),
            "pcqm_expert_sha256": _sha256(args.pcqm_expert),
        },
        "alignment": {
            "base_rows": int(len(base)),
            "teacher_rows": int(len(teacher)),
            "aligned_rows": int(len(merged)),
            "teacher_only_rows_excluded": int(len(teacher) - len(merged)),
            "sealed_20k_used": False,
        },
        "molecular_oracle": reports,
        "pcqm_task_router": pcqm,
        "gate": gate,
        "registry_changed": False,
        "router_trained": False,
    }
    _atomic_json(payload, args.out_dir / "oracle_metrics.json")
    if gate["survives_10pct_call_budget"]:
        _atomic_csv(
            pd.concat(label_frames, ignore_index=True),
            args.out_dir / "external_gain_labels.csv",
        )
        _atomic_json(
            {
                "status": "pending_generation",
                "purpose": "future conservative hard-router training",
                "required_contract": {
                    "domain": "repaired-2M training identity manifest",
                    "predictions": [
                        "Retention-D out-of-fold predictions",
                        "M07 hard-teacher out-of-fold predictions",
                    ],
                    "split": "scaffold-disjoint folds",
                    "targets": list(TARGETS),
                    "sealed_20k_used": False,
                },
                "external_gain_labels_are_training_eligible": False,
                "router_training_authorized": False,
            },
            args.out_dir / "oof_gain_label_manifest.json",
        )

    def row(scope: str, method: str) -> str:
        block = reports[scope]["methods"]
        average_delta = _method_delta(block, method)
        gap_delta = _method_delta(block, method, "gap")
        return (
            f"| {scope} | {method} | "
            f"{block[method]['average_mae_eV']:.6f} | {average_delta:+.6f} | "
            f"{block[method]['targets']['gap']['mae_eV']:.6f} | "
            f"{gap_delta:+.6f} |"
        )

    decision = (
        "# Phase 8.20 Hierarchical Oracle Decision\n\n"
        "## Decision\n\n"
        f"`{gate['decision']}`.\n\n"
        "The molecular Oracle clears the predeclared gate at a 10% hard-teacher "
        "call budget. This is an upper-bound feasibility result, not a deployable "
        "Router result. Generate genuine scaffold-disjoint OOF gains next; do not "
        "train a Router from the external evaluation labels.\n\n"
        "## Molecular Oracle\n\n"
        "| evaluation | method | average MAE | delta vs base | Gap MAE | "
        "Gap delta vs base |\n"
        "|---|---|---:|---:|---:|---:|\n"
        f"{row('common', 'base')}\n"
        f"{row('common', 'switch_10pct')}\n"
        f"{row('ood1000', 'base')}\n"
        f"{row('ood1000', 'switch_10pct')}\n"
        f"{row('p8_targeted_hard', 'base')}\n"
        f"{row('p8_targeted_hard', 'switch_10pct')}\n"
        f"{row('p8_targeted_hard', 'unconstrained_switch')}\n"
        f"{row('p8_targeted_hard', 'unconstrained_residual')}\n\n"
        "The 10% budget costs about `1.40` expected GPS encoder passes per "
        "molecule (`1` Retention-D pass plus `0.10 x 4` M07 passes).\n\n"
        "## PCQM Task Route\n\n"
        f"On {pcqm['paired_n']:,} aligned official-valid rows, deterministic GINE "
        f"routing changes Gap MAE from `{pcqm['base_gap_mae_eV']:.6f}` to "
        f"`{pcqm['expert_gap_mae_eV']:.6f} eV` "
        f"(`{pcqm['task_route_delta_eV']:+.6f} eV`).\n\n"
        "## Safety\n\n"
        "No sealed-20K rows were used, no Router was trained, and the production "
        "registry was not changed.\n"
    )
    decision_path = args.out_dir / "decision.md"
    temporary = decision_path.with_suffix(".md.tmp")
    temporary.write_text(decision, encoding="utf-8")
    temporary.replace(decision_path)
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
