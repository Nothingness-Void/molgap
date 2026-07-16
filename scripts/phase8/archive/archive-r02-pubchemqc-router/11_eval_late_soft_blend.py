"""Evaluate global, Gap-binned, and learned late soft blending with scaffold OOF."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from molgap.constants import RESULTS_DIR
from molgap.archive.phase8_r01_router.late_blend import (
    TARGETS, blend, fit_alpha_regressor, fit_fixed_alpha, fit_gap_binned_alpha,
    physics_project, predict_alpha_regressor, predict_gap_binned_alpha, target_metrics,
)
from molgap.router import paired_bootstrap_mean
from molgap.archive.phase8_r01_router.router_training import router_feature_sets


OUT_DIR = RESULTS_DIR / "phase8" / "archive" / "archive-r02-pubchemqc-router"
TARGET_WEIGHTS = np.asarray([0.25, 0.25, 0.50])


def arrays(frame: pd.DataFrame):
    y = frame[list(TARGETS)].to_numpy(dtype=np.float64)
    base = frame[[f"base_{target}" for target in TARGETS]].to_numpy(dtype=np.float64)
    expert = frame[[f"expert_{target}" for target in TARGETS]].to_numpy(dtype=np.float64)
    return y, base, expert


def consistency(predictions: np.ndarray) -> dict:
    residual = predictions[:, 2] - (predictions[:, 1] - predictions[:, 0])
    return {
        "mean_abs": float(np.mean(np.abs(residual))),
        "p95_abs": float(np.quantile(np.abs(residual), 0.95)),
        "max_abs": float(np.max(np.abs(residual))),
    }


def choose_variant(y, independent, weights):
    variants = {
        "independent_three_output": independent,
        "two_output_derived_gap": np.column_stack(
            [independent[:, 0], independent[:, 1], independent[:, 1] - independent[:, 0]]
        ),
    }
    best_gap_beta, best_gap_loss = None, np.inf
    best_combined_beta, best_combined_loss = None, np.inf
    for beta in np.linspace(0.0, 1.0, 21):
        projected = physics_project(independent, beta)
        combined_loss = float(np.average(np.abs(projected - y) @ TARGET_WEIGHTS, weights=weights))
        gap_loss = float(np.average(np.abs(projected[:, 2] - y[:, 2]), weights=weights))
        if gap_loss < best_gap_loss:
            best_gap_loss, best_gap_beta = gap_loss, float(beta)
        if combined_loss < best_combined_loss:
            best_combined_loss, best_combined_beta = combined_loss, float(beta)
    variants[f"physics_gap_beta_{best_gap_beta:.2f}"] = physics_project(independent, best_gap_beta)
    variants[f"physics_combined_beta_{best_combined_beta:.2f}"] = physics_project(independent, best_combined_beta)
    scores = {
        name: {
            "weighted_combined_mae": float(np.average(np.abs(prediction - y) @ TARGET_WEIGHTS, weights=weights)),
            "weighted_gap_mae": float(np.average(np.abs(prediction[:, 2] - y[:, 2]), weights=weights)),
            "consistency_mean_abs": consistency(prediction)["mean_abs"],
        }
        for name, prediction in variants.items()
    }
    selected = min(scores, key=lambda name: scores[name]["weighted_gap_mae"])
    return selected, variants, scores, best_gap_beta, best_combined_beta


def main() -> None:
    table = pd.read_parquet(OUT_DIR / "router_development_dataset_r5.parquet")
    late = pd.read_parquet(OUT_DIR / "late_blend_features.parquet")
    table = table.merge(late, on="probe_idx", how="left", validate="one_to_one")
    development = table[table.split.isin(["train", "validation"])].reset_index(drop=True)
    test = table[table.split == "dev_test"].reset_index(drop=True)
    r5_features = router_feature_sets(table.columns)["R5"]
    post_features = [
        *[f"expert_{target}" for target in TARGETS],
        *[f"expert_minus_base_{target}" for target in TARGETS],
        *[f"abs_expert_minus_base_{target}" for target in TARGETS],
        "base_prediction_consistency", "expert_prediction_consistency",
        *[column for column in late.columns if column != "probe_idx"],
    ]
    for target in TARGETS:
        table_column = table[f"expert_{target}"] - table[f"base_{target}"]
        table[f"expert_minus_base_{target}"] = table_column
        table[f"abs_expert_minus_base_{target}"] = table_column.abs()
    table["base_prediction_consistency"] = table.base_gap - (table.base_lumo - table.base_homo)
    table["expert_prediction_consistency"] = table.expert_gap - (table.expert_lumo - table.expert_homo)
    development = table[table.split.isin(["train", "validation"])].reset_index(drop=True)
    test = table[table.split == "dev_test"].reset_index(drop=True)
    features = list(dict.fromkeys(r5_features + post_features))

    n = len(development)
    oof_alpha = {
        method: np.zeros((n, 3), dtype=np.float64)
        for method in ("global", "gap_binned", "lightgbm")
    }
    fold_records = []
    splitter = GroupKFold(n_splits=5)
    for fold, (train_idx, val_idx) in enumerate(
        splitter.split(development, groups=development.scaffold), start=1
    ):
        train_fold = development.iloc[train_idx]
        val_fold = development.iloc[val_idx]
        for target_index, target in enumerate(TARGETS):
            y_train = train_fold[target].to_numpy(dtype=np.float64)
            base_train = train_fold[f"base_{target}"].to_numpy(dtype=np.float64)
            expert_train = train_fold[f"expert_{target}"].to_numpy(dtype=np.float64)
            weight_train = train_fold.training_weight.to_numpy(dtype=np.float64)
            fixed_alpha = fit_fixed_alpha(y_train, base_train, expert_train, weight_train)
            binned = fit_gap_binned_alpha(
                y_train, base_train, expert_train, train_fold.base_gap.to_numpy(), weight_train
            )
            model = fit_alpha_regressor(
                train_fold, features, target, seed=42 + fold * 10 + target_index
            )
            oof_alpha["global"][val_idx, target_index] = fixed_alpha
            oof_alpha["gap_binned"][val_idx, target_index] = predict_gap_binned_alpha(
                binned, val_fold.base_gap.to_numpy()
            )
            oof_alpha["lightgbm"][val_idx, target_index] = predict_alpha_regressor(
                model, val_fold, features
            )
        fold_records.append({
            "fold": fold, "train_n": len(train_idx), "validation_n": len(val_idx),
            "scaffold_overlap": len(set(train_fold.scaffold) & set(val_fold.scaffold)),
        })
        print(f"OOF fold {fold}/5", flush=True)

    y_dev, base_dev, expert_dev = arrays(development)
    weight_dev = development.training_weight.to_numpy(dtype=np.float64)
    oof_predictions = {
        method: blend(base_dev, expert_dev, alpha) for method, alpha in oof_alpha.items()
    }
    oof_metrics = {
        method: target_metrics(y_dev, prediction, weight_dev)
        for method, prediction in oof_predictions.items()
    }
    selected_methods = {}
    selected_oof = np.empty_like(y_dev)
    for target_index, target in enumerate(TARGETS):
        selected = min(
            oof_metrics,
            key=lambda method: oof_metrics[method][target]["weighted_mae"],
        )
        selected_methods[target] = selected
        selected_oof[:, target_index] = oof_predictions[selected][:, target_index]
    selected_variant, oof_variants, variant_scores, gap_beta, combined_beta = choose_variant(
        y_dev, selected_oof, weight_dev
    )

    final_alpha = {method: np.zeros((len(test), 3)) for method in oof_alpha}
    fitted_models = {}
    final_parameters = {}
    for target_index, target in enumerate(TARGETS):
        y_train = development[target].to_numpy(dtype=np.float64)
        base_train = development[f"base_{target}"].to_numpy(dtype=np.float64)
        expert_train = development[f"expert_{target}"].to_numpy(dtype=np.float64)
        weights = development.training_weight.to_numpy(dtype=np.float64)
        fixed_alpha = fit_fixed_alpha(y_train, base_train, expert_train, weights)
        binned = fit_gap_binned_alpha(
            y_train, base_train, expert_train, development.base_gap.to_numpy(), weights
        )
        model = fit_alpha_regressor(development, features, target, seed=100 + target_index)
        final_alpha["global"][:, target_index] = fixed_alpha
        final_alpha["gap_binned"][:, target_index] = predict_gap_binned_alpha(
            binned, test.base_gap.to_numpy()
        )
        final_alpha["lightgbm"][:, target_index] = predict_alpha_regressor(model, test, features)
        fitted_models[target] = {"global": fixed_alpha, "gap_binned": binned, "lightgbm": model}
        final_parameters[target] = {
            "global_alpha": float(fixed_alpha),
            "gap_binned_alpha": binned,
        }

    y_test, base_test, expert_test = arrays(test)
    weight_test = test.training_weight.to_numpy(dtype=np.float64)
    test_predictions = {
        method: blend(base_test, expert_test, alpha) for method, alpha in final_alpha.items()
    }
    selected_independent = np.column_stack([
        test_predictions[selected_methods[target]][:, index]
        for index, target in enumerate(TARGETS)
    ])
    test_variants = {
        "independent_three_output": selected_independent,
        "two_output_derived_gap": np.column_stack([
            selected_independent[:, 0], selected_independent[:, 1],
            selected_independent[:, 1] - selected_independent[:, 0],
        ]),
        f"physics_gap_beta_{gap_beta:.2f}": physics_project(selected_independent, gap_beta),
        f"physics_combined_beta_{combined_beta:.2f}": physics_project(selected_independent, combined_beta),
    }
    selected_test = test_variants[selected_variant]

    fixed_route = base_test[:, 2] < 4.0
    fixed_v4 = base_test.copy(); fixed_v4[fixed_route] = expert_test[fixed_route]
    baselines = {
        "base": base_test,
        "expert": expert_test,
        "fixed_v4": fixed_v4,
        "alpha_0.5": blend(base_test, expert_test, np.full_like(base_test, 0.5)),
        **test_predictions,
        "selected_independent": selected_independent,
        **{f"variant_{name}": prediction for name, prediction in test_variants.items()},
        "selected_final": selected_test,
    }
    test_metrics = {name: target_metrics(y_test, pred, weight_test) for name, pred in baselines.items()}
    gap_delta = np.abs(selected_test[:, 2] - y_test[:, 2]) - np.abs(fixed_v4[:, 2] - y_test[:, 2])
    bootstrap = paired_bootstrap_mean(gap_delta, n_bootstrap=10_000, seed=42)
    improvement = -float(bootstrap["delta"])
    verdict = "go" if improvement >= 0.001 and bootstrap["ci95"][1] < 0 else "stop"

    output = test[["probe_idx", "cid", "sampling_source", "scaffold"]].copy()
    for index, target in enumerate(TARGETS):
        output[f"y_{target}"] = y_test[:, index]
        output[f"fixed_v4_{target}"] = fixed_v4[:, index]
        output[f"late_blend_{target}"] = selected_test[:, index]
        output[f"late_blend_alpha_{target}"] = final_alpha[selected_methods[target]][:, index]
    output.to_parquet(OUT_DIR / "late_blend_dev_test_predictions.parquet", index=False)
    joblib.dump(
        {"features": features, "selected_methods": selected_methods,
         "selected_variant": selected_variant,
         "gap_projection_beta": gap_beta, "combined_projection_beta": combined_beta,
         "models": fitted_models},
        OUT_DIR / "late_blend_bundle.pkl",
    )
    label_consistency = consistency(y_test)
    prediction_consistency = {
        name: consistency(prediction) for name, prediction in baselines.items()
    }
    result = {
        "experiment": "archive-r02 late soft blend",
        "development_n": len(development),
        "dev_test_n": len(test),
        "oof": {
            "split": "5-fold scaffold-disjoint GroupKFold",
            "folds": fold_records,
            "method_metrics": oof_metrics,
            "selected_method_per_target": selected_methods,
            "variant_weighted_scores": variant_scores,
            "selected_variant": selected_variant,
            "physics_gap_projection_beta": gap_beta,
            "physics_combined_projection_beta": combined_beta,
            "final_global_and_binned_alpha": final_parameters,
        },
        "dev_test": {
            "metrics": test_metrics,
            "selected_gap_delta_vs_fixed_v4": bootstrap,
            "label_consistency": label_consistency,
            "prediction_consistency": prediction_consistency,
        },
        "decision": {
            "verdict": verdict,
            "gap_improvement_vs_fixed_v4_eV": improvement,
            "minimum_practical_improvement_eV": 0.001,
            "sealed_metrics_opened": False,
            "next_step": (
                "Repeat seeds and consider sealed evaluation." if verdict == "go"
                else "Stop late blending; do not open sealed sets."
            ),
        },
    }
    (OUT_DIR / "late_blend_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    selected_metrics = test_metrics["selected_final"]
    fixed_metrics = test_metrics["fixed_v4"]
    report = f"""# archive-r02 Late Soft Blend Decision

## Protocol

- 49,879 existing Base/Expert prediction pairs; no new GNN inference for labels.
- Five-fold scaffold-disjoint OOF on 45,478 development molecules.
- Independent 4,401-molecule dev-test; sealed random/hard sets remain unopened.
- Selection target: Gap MAE. Promotion gate: at least 0.001 eV Gap improvement over fixed v4.

## Results

| Model | HOMO MAE | LUMO MAE | Gap MAE |
|---|---:|---:|---:|
| Fixed v4 | {fixed_metrics['homo']['mae']:.6f} | {fixed_metrics['lumo']['mae']:.6f} | {fixed_metrics['gap']['mae']:.6f} |
| Late blend | {selected_metrics['homo']['mae']:.6f} | {selected_metrics['lumo']['mae']:.6f} | {selected_metrics['gap']['mae']:.6f} |

- OOF-selected alpha method: `{selected_methods}`.
- OOF-selected output structure: `{selected_variant}`.
- Gap improvement: {improvement:.6f} eV; paired bootstrap 95% CI for late-minus-v4 error: [{bootstrap['ci95'][0]:.6f}, {bootstrap['ci95'][1]:.6f}] eV.
- Labels satisfy `Gap = LUMO - HOMO` to numerical precision on dev-test (mean absolute residual {label_consistency['mean_abs']:.3g} eV).
- Physics projection is retained as a diagnostic; it is not selected unless it improves OOF Gap MAE.

## Decision

**{verdict.upper()}**. {result['decision']['next_step']} The production default remains fixed v4.
"""
    (OUT_DIR / "late_blend_decision.md").write_text(report, encoding="utf-8")
    print(json.dumps(result["decision"], indent=2), flush=True)


if __name__ == "__main__":
    main()
