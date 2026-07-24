"""Build a unified evidence table for Phase 8 scale-up experiments."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance

from molgap.residual_attribution import analyze_comparison, molecular_descriptors


OUT = Path("results/phase8/scaleup_full_analysis")
TARGETS = ("homo", "lumo", "gap")


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


def add_model(
    base: pd.DataFrame,
    source_path: Path,
    source_name: str,
    output_name: str,
) -> None:
    source = pd.read_csv(source_path)
    columns = ["cid", *(f"{source_name}_{target}" for target in TARGETS)]
    source = source.loc[:, columns].drop_duplicates("cid")
    source = source.rename(
        columns={
            f"{source_name}_{target}": f"{output_name}_{target}"
            for target in TARGETS
        }
    )
    merged = base[["cid"]].merge(source, on="cid", how="left", validate="one_to_one")
    for target in TARGETS:
        base[f"{output_name}_{target}"] = merged[f"{output_name}_{target}"].to_numpy()


def unified_common_predictions() -> pd.DataFrame:
    one_million = pd.read_csv(
        "results/phase8/expansion_1m/common_eval_kaggle_predictions.csv"
    )
    base = one_million.loc[:, ["eval_set", "cid", "smiles", *TARGETS]].copy()
    for target in TARGETS:
        base[f"routed_v4_500k_{target}"] = one_million[f"routed_v4_{target}"]
        base[f"fusion_1m_{target}"] = one_million[f"candidate_1m_{target}"]
    model_sources = [
        (
            "results/phase8/repair_v2_2d_external_eval/repair_v2_2d_common_predictions.csv",
            "v2",
            "repair_v2_1m_2d",
        ),
        (
            "results/phase8/repair_v3_1p5m_external_eval/common_predictions.csv",
            "additive_1p5m",
            "additive_1p5m_2d",
        ),
        (
            "results/phase8/broad_residual98k_external_eval/common_predictions.csv",
            "broad_residual98k_uniform",
            "broad_1p098m_2d",
        ),
        (
            "results/phase8/multi2d_2m_dev_eval/common_predictions.csv",
            "coverage2m",
            "coverage_2m_2d",
        ),
        (
            "results/phase8/multi2d_2m_dev_eval/common_predictions.csv",
            "incumbent",
            "ensemble_two_1m_2d",
        ),
        (
            "results/phase8/multi2d_2m_dev_eval/common_predictions.csv",
            "tri_expert",
            "ensemble_three_2m_2d",
        ),
        (
            "results/phase8/distilled_2m_external_eval/common_predictions.csv",
            "student_w30",
            "distilled_2m_2d",
        ),
    ]
    for path, source_name, output_name in model_sources:
        add_model(base, Path(path), source_name, output_name)
    return base


def common_metrics(table: pd.DataFrame) -> pd.DataFrame:
    models = sorted(
        {
            column.rsplit("_", 1)[0]
            for column in table
            if any(column.endswith(f"_{target}") for target in TARGETS)
            and column not in TARGETS
        }
    )
    rows = []
    for scope in ("all", "ood1000", "p8_targeted_hard"):
        scoped = table if scope == "all" else table.loc[table.eval_set.eq(scope)]
        for model in models:
            prediction_columns = [f"{model}_{target}" for target in TARGETS]
            valid = scoped[prediction_columns].notna().all(axis=1)
            selected = scoped.loc[valid]
            if selected.empty:
                continue
            truth = selected.loc[:, TARGETS].to_numpy(np.float64)
            prediction = selected.loc[:, prediction_columns].to_numpy(np.float64)
            baseline = selected.loc[
                :, [f"routed_v4_500k_{target}" for target in TARGETS]
            ].to_numpy(np.float64)
            absolute = np.abs(prediction - truth)
            baseline_absolute = np.abs(baseline - truth)
            for target_index, target in enumerate((*TARGETS, "average")):
                if target == "average":
                    error = absolute.mean(axis=1)
                    baseline_error = baseline_absolute.mean(axis=1)
                else:
                    error = absolute[:, target_index]
                    baseline_error = baseline_absolute[:, target_index]
                delta = error - baseline_error
                standard_error = (
                    float(delta.std(ddof=1) / np.sqrt(len(delta)))
                    if len(delta) > 1
                    else float("nan")
                )
                rows.append(
                    {
                        "scope": scope,
                        "model": model,
                        "target": target,
                        "n": len(selected),
                        "mae_eV": float(error.mean()),
                        "delta_vs_routed_v4_500k_eV": float(delta.mean()),
                        "paired_normal_ci95_low_eV": float(
                            delta.mean() - 1.96 * standard_error
                        ),
                        "paired_normal_ci95_high_eV": float(
                            delta.mean() + 1.96 * standard_error
                        ),
                        "win_rate_vs_routed_v4_500k": float(
                            (error < baseline_error).mean()
                        ),
                        "residual_correlation_vs_routed_v4_500k": float(
                            np.corrcoef(error, baseline_error)[0, 1]
                        ),
                    }
                )
    return pd.DataFrame(rows)


def unified_pcqm_predictions() -> pd.DataFrame:
    base = pd.read_csv(
        "results/phase8/expansion_1m/pcqm4mv2_valid_5k_component_predictions.csv"
    ).loc[:, ["cid", "smiles", "gap", "routed_v4_gap", "candidate_1m_gap"]]
    base = base.rename(
        columns={
            "cid": "idx",
            "routed_v4_gap": "routed_v4_500k_gap",
            "candidate_1m_gap": "fusion_1m_gap",
        }
    )
    sources = [
        (
            "results/phase8/repair_v2_2d_external_eval/repair_v2_2d_pcqm_predictions.csv",
            "v2_gap",
            "repair_v2_1m_2d_gap",
        ),
        (
            "results/phase8/broad_residual98k_external_eval/pcqm_predictions.csv",
            "broad_residual98k_uniform_gap",
            "broad_1p098m_2d_gap",
        ),
        (
            "results/phase8/multi2d_final_eval/pcqm_predictions.csv",
            "mean_control_repair_gap",
            "ensemble_two_1m_2d_gap",
        ),
        (
            "results/phase8/distilled_2m_external_eval/pcqm_predictions.csv",
            "student_w30_gap",
            "distilled_2m_2d_gap",
        ),
    ]
    for path, source_column, output_column in sources:
        source = pd.read_csv(path).loc[:, ["idx", source_column]]
        source = source.drop_duplicates("idx").rename(
            columns={source_column: output_column}
        )
        base = base.merge(source, on="idx", how="left", validate="one_to_one")
    return base


def pcqm_metrics(table: pd.DataFrame) -> pd.DataFrame:
    truth = table.gap.to_numpy(np.float64)
    baseline_error = np.abs(table.routed_v4_500k_gap.to_numpy(np.float64) - truth)
    rows = []
    for column in sorted(
        column for column in table if column.endswith("_gap") and column != "gap"
    ):
        valid = table[column].notna().to_numpy()
        error = np.abs(table.loc[valid, column].to_numpy(np.float64) - truth[valid])
        baseline = baseline_error[valid]
        delta = error - baseline
        standard_error = float(delta.std(ddof=1) / np.sqrt(len(delta)))
        rows.append(
            {
                "model": column.removesuffix("_gap"),
                "n": int(valid.sum()),
                "gap_mae_eV": float(error.mean()),
                "delta_vs_routed_v4_500k_eV": float(delta.mean()),
                "paired_normal_ci95_low_eV": float(
                    delta.mean() - 1.96 * standard_error
                ),
                "paired_normal_ci95_high_eV": float(
                    delta.mean() + 1.96 * standard_error
                ),
                "win_rate_vs_routed_v4_500k": float((error < baseline).mean()),
            }
        )
    return pd.DataFrame(rows)


def distribution_row(name: str, table: pd.DataFrame, base: pd.DataFrame) -> dict:
    gap_bins = np.linspace(0, 12, 49)
    mw_bins = np.linspace(200, 1000, 33)
    reference_hist, _, _ = np.histogram2d(
        base.gap, base.mw, bins=(gap_bins, mw_bins)
    )
    candidate_hist, _, _ = np.histogram2d(
        table.gap, table.mw, bins=(gap_bins, mw_bins)
    )
    reference_hist = reference_hist.ravel() + 1e-12
    candidate_hist = candidate_hist.ravel() + 1e-12
    identity_error = np.abs(table.gap - (table.lumo - table.homo))
    return {
        "segment": name,
        "rows": len(table),
        "mw_mean": float(table.mw.mean()),
        "mw_p50": float(table.mw.median()),
        "gap_mean": float(table.gap.mean()),
        "gap_p50": float(table.gap.median()),
        "homo_std": float(table.homo.std()),
        "lumo_std": float(table.lumo.std()),
        "gap_std": float(table.gap.std()),
        "gap_lt_3_fraction": float(table.gap.lt(3).mean()),
        "gap_lt_4_fraction": float(table.gap.lt(4).mean()),
        "gap_gt_6_fraction": float(table.gap.gt(6).mean()),
        "mw_gt_500_fraction": float(table.mw.gt(500).mean()),
        "mw_gt_700_fraction": float(table.mw.gt(700).mean()),
        "mw_gt_800_fraction": float(table.mw.gt(800).mean()),
        "multi_fragment_fraction": float(
            table.smiles.astype(str).str.contains(".", regex=False).mean()
        ),
        "mw_wasserstein_vs_base500k": float(
            wasserstein_distance(base.mw, table.mw)
        ),
        "gap_wasserstein_vs_base500k": float(
            wasserstein_distance(base.gap, table.gap)
        ),
        "gap_mw_js_distance_vs_base500k": float(
            jensenshannon(reference_hist, candidate_hist)
        ),
        "gap_identity_max_abs_eV": float(identity_error.max()),
    }


def dataset_distributions() -> pd.DataFrame:
    columns = ["mw", "homo", "lumo", "gap", "smiles"]
    raw = Path("data/raw")
    base = pd.read_csv(raw / "phase8_expansion_500k.csv", usecols=columns)
    original_1m = pd.read_csv(raw / "phase8_expansion_1m.csv", usecols=columns)
    repair_1m = pd.read_csv(raw / "phase8_repair_v2_1m.csv", usecols=columns)
    additive_1p5m = pd.read_csv(
        raw / "phase8_repair_v3_1p5m.csv", usecols=columns
    )
    broad = pd.read_csv(
        raw / "phase8_expansion_1m_broad_residual.csv", usecols=columns
    )
    exact_2m_topup = pd.read_csv(
        raw / "phase8_multi2d_2m_topup_500k.csv", usecols=columns
    )
    segments = {
        "base500k": base,
        "original1m_general_topup500k": original_1m.iloc[500_000:],
        "repair_targeted_topup500k": repair_1m.iloc[500_000:],
        "broad_residual_topup97798": broad.iloc[1_000_000:],
        "exact2m_mixed_topup500k": exact_2m_topup,
        "combined_original1m": original_1m,
        "combined_repair1m": repair_1m,
        "combined_additive1p5m": additive_1p5m,
        "combined_exact2m": pd.concat(
            [additive_1p5m, exact_2m_topup], ignore_index=True
        ),
    }
    return pd.DataFrame(
        [distribution_row(name, table, base) for name, table in segments.items()]
    )


def sampled_chemical_distributions(sample_rows: int = 30_000) -> pd.DataFrame:
    raw = Path("data/raw")
    base = pd.read_csv(raw / "phase8_expansion_500k.csv", usecols=["smiles"])
    original = pd.read_csv(raw / "phase8_expansion_1m.csv", usecols=["smiles"])
    repair = pd.read_csv(raw / "phase8_repair_v2_1m.csv", usecols=["smiles"])
    broad = pd.read_csv(
        raw / "phase8_expansion_1m_broad_residual.csv", usecols=["smiles"]
    )
    exact2m = pd.read_csv(
        raw / "phase8_multi2d_2m_topup_500k.csv", usecols=["smiles"]
    )
    segments = {
        "base500k": base,
        "original1m_general_topup500k": original.iloc[500_000:],
        "repair_targeted_topup500k": repair.iloc[500_000:],
        "broad_residual_topup97798": broad.iloc[1_000_000:],
        "exact2m_mixed_topup500k": exact2m,
    }
    rows = []
    for rank, (name, table) in enumerate(segments.items()):
        sample = table.sample(
            n=min(sample_rows, len(table)), random_state=20260723 + rank
        )
        descriptors = molecular_descriptors(sample.smiles.tolist())
        valid = descriptors.heavy_atoms.notna()
        selected = descriptors.loc[valid]
        rows.append(
            {
                "segment": name,
                "sample_rows": len(sample),
                "valid_fraction": float(valid.mean()),
                "heavy_atoms_mean": float(selected.heavy_atoms.mean()),
                "aromatic_rings_mean": float(selected.aromatic_rings.mean()),
                "aromatic_rings_ge_4_fraction": float(
                    selected.aromatic_rings.ge(4).mean()
                ),
                "aromatic_atom_fraction_mean": float(
                    selected.aromatic_atom_fraction.mean()
                ),
                "rotatable_bonds_mean": float(selected.rotatable_bonds.mean()),
                "rotatable_bonds_ge_10_fraction": float(
                    selected.rotatable_bonds.ge(10).mean()
                ),
                "multi_fragment_fraction": float(selected.fragments.gt(1).mean()),
                "radical_fraction": float(selected.radical_electrons.gt(0).mean()),
                "has_s_fraction": float(selected.has_s.mean()),
                "has_f_fraction": float(selected.has_f.mean()),
                "has_cl_fraction": float(selected.has_cl.mean()),
            }
        )
    return pd.DataFrame(rows)


def training_dynamics() -> pd.DataFrame:
    specs = {
        "base500k_gps7": "results/phase8/gps_expansion_500k_metrics.json",
        "repair1m_gps7": "results/phase8/repair_v2_scnet/controlled_2d/gps7_metrics.json",
        "repair1m_gps9": "results/phase8/repair_v2_scnet/controlled_2d/gps9_metrics.json",
        "exact2m_gps7": "results/phase8/multi2d_2m_scnet/gps7/metrics.json",
        "exact2m_gps9": "results/phase8/multi2d_2m_scnet/gps9/metrics.json",
    }
    rows = []
    for name, path in specs.items():
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        best_epoch = int(value["best_epoch"])
        best_row = next(
            row for row in value["log"] if int(row["epoch"]) == best_epoch
        )
        rows.append(
            {
                "run": name,
                "rows": value["n_graphs"],
                "best_epoch": best_epoch,
                "best_train_mae_eV": best_row["train_loss"],
                "best_validation_mae_eV": value["best_val_mae"],
                "train_validation_gap_eV": value["best_val_mae"]
                - best_row["train_loss"],
                "learning_rate": value["params"]["lr"],
                "init_from": value.get("init_from"),
                "replay_sampling": json.dumps(value.get("replay_sampling")),
            }
        )
    return pd.DataFrame(rows)


def training_composition() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dataset": "base500k",
                "rows": 500_000,
                "base500k_fraction": 1.0,
                "general_topup_fraction": 0.0,
                "repair_topup_fraction": 0.0,
                "mixed2m_topup_fraction": 0.0,
                "actual_replay": False,
            },
            {
                "dataset": "original1m",
                "rows": 1_000_000,
                "base500k_fraction": 0.5,
                "general_topup_fraction": 0.5,
                "repair_topup_fraction": 0.0,
                "mixed2m_topup_fraction": 0.0,
                "actual_replay": False,
            },
            {
                "dataset": "repair1m",
                "rows": 1_000_000,
                "base500k_fraction": 0.5,
                "general_topup_fraction": 0.0,
                "repair_topup_fraction": 0.5,
                "mixed2m_topup_fraction": 0.0,
                "actual_replay": False,
            },
            {
                "dataset": "additive1p5m",
                "rows": 1_500_000,
                "base500k_fraction": 1 / 3,
                "general_topup_fraction": 1 / 3,
                "repair_topup_fraction": 1 / 3,
                "mixed2m_topup_fraction": 0.0,
                "actual_replay": False,
            },
            {
                "dataset": "exact2m",
                "rows": 2_000_000,
                "base500k_fraction": 0.25,
                "general_topup_fraction": 0.25,
                "repair_topup_fraction": 0.25,
                "mixed2m_topup_fraction": 0.25,
                "actual_replay": False,
            },
        ]
    )


def deployment_matrix(
    common_summary: pd.DataFrame, pcqm_summary: pd.DataFrame
) -> pd.DataFrame:
    metadata = {
        "fusion_1m": ("2D+3D fusion", 3.0),
        "repair_v2_1m_2d": ("pure 2D", 2.0),
        "additive_1p5m_2d": ("pure 2D", 2.0),
        "broad_1p098m_2d": ("pure 2D", 2.0),
        "coverage_2m_2d": ("pure 2D", 2.0),
        "ensemble_two_1m_2d": ("pure 2D ensemble", 4.0),
        "ensemble_three_2m_2d": ("pure 2D ensemble", 6.0),
        "distilled_2m_2d": ("pure 2D distilled", 1.0),
    }
    rows = []
    for model, (family, encoder_passes) in metadata.items():
        row = {
            "model": model,
            "family": family,
            "approximate_encoder_passes": encoder_passes,
        }
        for scope in ("all", "ood1000", "p8_targeted_hard"):
            selected = common_summary.loc[
                (common_summary.model.eq(model))
                & (common_summary.scope.eq(scope))
                & (common_summary.target.eq("average"))
            ]
            if not selected.empty:
                row[f"{scope}_average_mae_eV"] = float(selected.iloc[0].mae_eV)
                row[f"{scope}_average_delta_vs_v4_eV"] = float(
                    selected.iloc[0].delta_vs_routed_v4_500k_eV
                )
        selected_pcqm = pcqm_summary.loc[pcqm_summary.model.eq(model)]
        if not selected_pcqm.empty:
            row["pcqm_gap_mae_eV"] = float(selected_pcqm.iloc[0].gap_mae_eV)
            row["pcqm_gap_delta_vs_v4_eV"] = float(
                selected_pcqm.iloc[0].delta_vs_routed_v4_500k_eV
            )
            row["pcqm_comparison"] = "paired ETKDG-valid 4,981"
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    common = unified_common_predictions()
    metrics = common_metrics(common)
    candidate_names = sorted(
        {
            column.rsplit("_", 1)[0]
            for column in common
            if column.endswith("_gap")
            and column not in {"gap", "routed_v4_500k_gap"}
        }
    )
    residual_report, residual_strata = analyze_comparison(
        common,
        baseline="routed_v4_500k",
        candidates=candidate_names,
    )
    pcqm = unified_pcqm_predictions()
    pcqm_summary = pcqm_metrics(pcqm)
    distributions = dataset_distributions()
    chemical_distributions = sampled_chemical_distributions()
    dynamics = training_dynamics()
    composition = training_composition()
    deployment = deployment_matrix(metrics, pcqm_summary)
    atomic_csv(common, OUT / "unified_common_predictions.csv")
    atomic_csv(metrics, OUT / "unified_common_metrics.csv")
    atomic_json(residual_report, OUT / "residual_attribution.json")
    atomic_csv(residual_strata, OUT / "residual_strata.csv")
    atomic_csv(pcqm, OUT / "unified_pcqm_predictions.csv")
    atomic_csv(pcqm_summary, OUT / "unified_pcqm_metrics.csv")
    atomic_csv(distributions, OUT / "dataset_distributions.csv")
    atomic_csv(
        chemical_distributions, OUT / "sampled_chemical_distributions.csv"
    )
    atomic_csv(dynamics, OUT / "training_dynamics.csv")
    atomic_csv(composition, OUT / "training_composition.csv")
    atomic_csv(deployment, OUT / "deployment_matrix.csv")
    atomic_json(
        {
            "complete": True,
            "common_rows": len(common),
            "pcqm_rows": len(pcqm),
            "models": sorted(metrics.model.unique()),
            "distribution_segments": distributions.segment.tolist(),
            "sealed_opened": False,
        },
        OUT / "manifest.json",
    )


if __name__ == "__main__":
    main()
