"""Build the compact production-model comparison from accepted evidence."""

from __future__ import annotations

import csv
import json
from pathlib import Path


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_comparison_rows(repo: Path) -> list[dict]:
    repaired = repo / "experiments" / "repaired_2m_scaling" / "results"
    seed42 = _read(repaired / "retention_d_seed42_raw" / "common_metrics.json")
    seed42_pcqm = _read(repaired / "retention_d_seed42_raw" / "pcqm_metrics.json")
    oracle = _read(
        repaired / "gps7_gps9_oracle" / "oracle_metrics.json"
    )
    gps9_pcqm = _read(
        repaired / "gps9_seed42_raw" / "external" / "pcqm_metrics.json"
    )
    ensemble = _read(repaired / "retention_d_three_seed_equal_ensemble.json")
    route_b = _read(
        repo
        / "experiments"
        / "pubchemqc100k_architecture"
        / "results"
        / "route_b_fusion_summary.json"
    )
    route_b_residual = _read(
        repo
        / "experiments"
        / "pubchemqc100k_architecture"
        / "results"
        / "route_b_residual_scale_summary.json"
    )

    hierarchical = _read(
        repaired / "hierarchical_dual_schnet_external" / "metrics.json"
    )
    three_gps_pcqm = _read(
        repaired / "three_gps_router_fusion" / "pcqm_valid" / "metrics.json"
    )

    def common_model(model: str, cost: float, role: str) -> dict:
        scopes = seed42["scopes"]
        return {
            "model": model,
            "role": role,
            "common_eval_rows": seed42["n_valid"],
            "common_average_mae_eV": scopes["all"][model]["average"]["mae_eV"],
            "ood_average_mae_eV": scopes["ood1000"][model]["average"]["mae_eV"],
            "p8_hard_average_mae_eV": scopes["p8_targeted_hard"][model]["average"]["mae_eV"],
            "pcqm_gap_mae_eV": None,
            "pubchemqc100k_average_mae_eV": None,
            "pubchemqc100k_gap_mae_eV": None,
            "encoder_passes": cost,
        }

    def paired_2d_model(
        method: str, model: str, cost: float, role: str, pcqm_method: str
    ) -> dict:
        # The hierarchical external comparison dropped 4 ETKDG failures so every
        # method shared identical rows; its 1,973 rows are therefore not directly
        # comparable to the 1,977-row single-model runs above.
        scopes = hierarchical["metrics"]
        return {
            "model": model,
            "role": role,
            "common_eval_rows": hierarchical["aligned_rows"],
            "common_average_mae_eV": scopes["all"]["methods"][method]["average"]["mae_eV"],
            "ood_average_mae_eV": scopes["ood1000"]["methods"][method]["average"]["mae_eV"],
            "p8_hard_average_mae_eV": scopes["p8_targeted_hard"]["methods"][method][
                "average"
            ]["mae_eV"],
            "pcqm_gap_mae_eV": three_gps_pcqm["metrics"][pcqm_method]["gap_mae_eV"],
            "pubchemqc100k_average_mae_eV": None,
            "pubchemqc100k_gap_mae_eV": None,
            "encoder_passes": cost,
        }

    rows = [
        common_model("routed_v4_500k", 3.0, "previous production baseline"),
        common_model(
            "repaired_2m_d_gps7_seed42",
            1.0,
            "accepted general candidate",
        ),
    ]
    rows[0]["pcqm_gap_mae_eV"] = seed42_pcqm["routed_v4_500k_gap_mae_eV"]
    rows[1]["pcqm_gap_mae_eV"] = seed42_pcqm[
        "repaired_2m_d_gps7_seed42_gap_mae_eV"
    ]
    rows.append(
        {
            "model": "repaired_2m_d_gps9_seed42",
            "role": "hard expert candidate",
            "common_eval_rows": oracle["reports"]["common"]["n"],
            "common_average_mae_eV": oracle["reports"]["common"]["methods"]["expert"][
                "average_mae_eV"
            ],
            "ood_average_mae_eV": oracle["reports"]["ood1000"]["methods"]["expert"][
                "average_mae_eV"
            ],
            "p8_hard_average_mae_eV": oracle["reports"]["p8_targeted_hard"][
                "methods"
            ]["expert"]["average_mae_eV"],
            "pcqm_gap_mae_eV": gps9_pcqm[
                "repaired_2m_d_gps9_seed42_gap_mae_eV"
            ],
            "pubchemqc100k_average_mae_eV": None,
            "pubchemqc100k_gap_mae_eV": None,
            "encoder_passes": 1.0,
        }
    )
    rows.append(
        {
            "model": "retention_d_three_seed_equal_ensemble",
            "role": "accuracy mode candidate",
            "common_eval_rows": ensemble["common"]["common"]["n"],
            "common_average_mae_eV": ensemble["common"]["common"]["metrics"][
                "average"
            ]["mae_eV"],
            "ood_average_mae_eV": ensemble["common"]["ood1000"]["metrics"][
                "average"
            ]["mae_eV"],
            "p8_hard_average_mae_eV": ensemble["common"]["p8_targeted_hard"][
                "metrics"
            ]["average"]["mae_eV"],
            "pcqm_gap_mae_eV": ensemble["pcqm"]["gap"]["mae_eV"],
            "pubchemqc100k_average_mae_eV": None,
            "pubchemqc100k_gap_mae_eV": None,
            "encoder_passes": 3.0,
        }
    )
    for candidate in ("minimal", "cost", "precision"):
        metrics = route_b["two_schnet_pass_fusion"][candidate]
        rows.append(
            {
                "model": f"route_b_{candidate}",
                "role": (
                    "accepted scale-up candidate"
                    if candidate == "precision"
                    else "architecture control"
                ),
                "common_eval_rows": None,
                "common_average_mae_eV": None,
                "ood_average_mae_eV": None,
                "p8_hard_average_mae_eV": None,
                "pcqm_gap_mae_eV": None,
                "pubchemqc100k_average_mae_eV": metrics[
                    "test_average_mae_eV"
                ],
                "pubchemqc100k_gap_mae_eV": metrics["test_gap_mae_eV"],
                "encoder_passes": 3.0 if candidate == "minimal" else 4.0,
            }
        )
    selected = route_b_residual["scales"]["0.10"]
    rows.append(
        {
            "model": "route_b_precision_bounded_residual",
            "role": "accepted scale-up protocol",
            "common_eval_rows": None,
            "common_average_mae_eV": None,
            "ood_average_mae_eV": None,
            "p8_hard_average_mae_eV": None,
            "pcqm_gap_mae_eV": None,
            "pubchemqc100k_average_mae_eV": selected[
                "test_average_mae_eV"
            ],
            "pubchemqc100k_gap_mae_eV": selected["test_gap_mae_eV"],
            "encoder_passes": 4.0,
        }
    )
    rows.extend(
        (
            paired_2d_model(
                "repaired_2m_dense_2d",
                "repaired_2m_three_gps_dense_2d",
                3.0,
                "production",
                "dense_soft_gate",
            ),
            paired_2d_model(
                "repaired_2m_equal_2d",
                "repaired_2m_gps7_gps9_equal_2d",
                2.0,
                "production lower-cost preset",
                "equal_gps7_gps9",
            ),
        )
    )
    return rows


def write_comparison(rows: list[dict], csv_path: Path, markdown_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Phase 8 Model Comparison",
        "",
        "Generated from accepted result artifacts. Blank cells indicate that a model was not evaluated on that domain.",
        "",
        "Common/OOD/P8-hard columns are only comparable within the same `Rows` "
        "value: the paired repaired-2M comparison dropped the 4 rows where "
        "ETKDG failed so every method shared identical molecules, while the "
        "single-model runs kept all 1,977.",
        "",
        "| Model | Role | Rows | Common | OOD | P8-hard | PCQM Gap | PC100K avg | PC100K Gap | Encoder passes |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        values = [
            row["model"],
            row["role"],
            "" if row["common_eval_rows"] is None else f"{row['common_eval_rows']:,}",
            *[
                "" if row[key] is None else f"{row[key]:.6f}"
                for key in (
                    "common_average_mae_eV",
                    "ood_average_mae_eV",
                    "p8_hard_average_mae_eV",
                    "pcqm_gap_mae_eV",
                    "pubchemqc100k_average_mae_eV",
                    "pubchemqc100k_gap_mae_eV",
                )
            ],
            "" if row["encoder_passes"] is None else f"{row['encoder_passes']:.2f}",
        ]
        lines.append("| " + " | ".join(values) + " |")
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
