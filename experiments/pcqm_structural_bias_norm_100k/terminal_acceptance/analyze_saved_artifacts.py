from __future__ import annotations

import json
import math
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from molgap.router import paired_bootstrap_mean


ARTIFACT_ROOT = Path(
    r"D:\molgap-artifacts\acceptance\structural-bias-norm-100k\20261020"
)
OUTPUT_DIR = Path(__file__).resolve().parent
N_BOOTSTRAP = 10_000
SEED = 42
MATERIAL_GATE_EV = 0.003

PAIRS = {
    "k1": {
        "title": "K1 · live development MAE",
        "control": "post_bn",
        "candidate": "post_ln",
        "primary_metric": "live_dev_metric",
    },
    "gptrans": {
        "title": "GPTrans · EMA development MAE",
        "control": "bond_local",
        "candidate": "resistance",
        "primary_metric": "ema_dev_metric",
    },
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def as_numpy(value, name: str) -> np.ndarray:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    return value.detach().cpu().numpy()


def load_arm(family: str, arm_id: str) -> dict:
    root = ARTIFACT_ROOT / family / "family_outputs" / arm_id
    manifest = read_json(root / "output_manifest.json")
    contract = read_json(root / "training_contract.json")
    trace = read_json(root / "canonical_trace.json")

    if manifest.get("format") != "molgap-family-output-v1":
        raise ValueError(f"{family}/{arm_id}: unexpected output format")
    context = manifest.get("context", {})
    if context.get("arm_id") != arm_id:
        raise ValueError(f"{family}/{arm_id}: output context arm mismatch")
    if manifest.get("progress", {}).get("status") != "complete":
        raise ValueError(f"{family}/{arm_id}: endpoint is not complete")
    if trace.get("run_id") != (
        context.get("logical_run_id") + ":" + arm_id + ":downstream"
    ):
        raise ValueError(f"{family}/{arm_id}: trace run identity mismatch")

    prediction_path = root / manifest["artifacts"]["predictions"]["path"]
    if prediction_path.name != "development_predictions.pt":
        raise ValueError(f"{family}/{arm_id}: unexpected predictions artifact")
    payload = torch.load(prediction_path, map_location="cpu", weights_only=True)
    required = {"source_idx", "target_eV", "prediction_eV", "context"}
    if not required.issubset(payload):
        raise ValueError(f"{family}/{arm_id}: predictions payload is incomplete")
    if payload["context"].get("arm_id") != arm_id:
        raise ValueError(f"{family}/{arm_id}: prediction context arm mismatch")

    source_idx = as_numpy(payload["source_idx"], "source_idx")
    target = as_numpy(payload["target_eV"], "target_eV").astype(np.float64, copy=False)
    prediction = as_numpy(payload["prediction_eV"], "prediction_eV").astype(
        np.float64, copy=False
    )
    if source_idx.ndim != 1 or target.ndim != 1 or prediction.ndim != 1:
        raise ValueError(f"{family}/{arm_id}: expected one-dimensional predictions")
    if not (len(source_idx) == len(target) == len(prediction) == 50_000):
        raise ValueError(f"{family}/{arm_id}: expected exactly 50,000 rows")
    if not np.isfinite(target).all() or not np.isfinite(prediction).all():
        raise ValueError(f"{family}/{arm_id}: target/prediction contains nonfinite values")

    metric_semantics = trace.get("metric_semantics", {})
    primary = PAIRS[family]["primary_metric"]
    semantics = metric_semantics.get(primary)
    if not isinstance(semantics, dict) or semantics.get("unit") != "eV":
        raise ValueError(f"{family}/{arm_id}: primary development metric is not eV")
    if semantics.get("target") != "Gap":
        raise ValueError(f"{family}/{arm_id}: primary metric target is not Gap")

    observations = [
        row for row in trace.get("observations", []) if row.get("event") == "observation"
    ]
    if not observations:
        raise ValueError(f"{family}/{arm_id}: trace has no observations")
    primary_values = np.asarray(
        [row[primary] for row in observations], dtype=np.float64
    )
    if not np.isfinite(primary_values).all():
        raise ValueError(f"{family}/{arm_id}: primary trace metric is nonfinite")
    best_index = int(np.argmin(primary_values))
    last = observations[-1]
    errors = prediction - target
    absolute_errors = np.abs(errors)

    train_key = "live_train_metric"
    train_semantics = metric_semantics.get(train_key)
    online_train_normalized = observations[-1].get(train_key)
    if not isinstance(train_semantics, dict) or train_semantics.get("unit") != "normalized-Gap":
        raise ValueError(f"{family}/{arm_id}: online train metric unit is not normalized-Gap")
    target_std = contract.get("target_std_eV")
    online_train_eV = (
        float(online_train_normalized) * float(target_std)
        if online_train_normalized is not None and target_std is not None
        else None
    )

    windows = {}
    for span in (10, 5):
        if len(observations) <= span:
            windows[f"last_{span}_epochs"] = {"status": "insufficient_trace"}
            continue
        start = observations[-(span + 1)]
        start_value = float(start[primary])
        final_value = float(last[primary])
        windows[f"last_{span}_epochs"] = {
            "start_epoch": start["epoch_or_pass"],
            "end_epoch": last["epoch_or_pass"],
            "span_epochs": span,
            "start_metric_eV": start_value,
            "end_metric_eV": final_value,
            "improvement_eV": start_value - final_value,
        }

    return {
        "arm_id": arm_id,
        "root": root,
        "manifest": manifest,
        "contract": contract,
        "trace": trace,
        "source_idx": source_idx,
        "target_eV": target,
        "prediction_eV": prediction,
        "absolute_error_eV": absolute_errors,
        "signed_error_eV": errors,
        "primary_metric": primary,
        "observations": observations,
        "curve_eV": primary_values,
        "selected_prediction_mae_eV": float(absolute_errors.mean()),
        "mean_signed_error_eV": float(errors.mean()),
        "absolute_error_quantiles_eV": {
            f"p{q}": float(np.quantile(absolute_errors, q / 100.0))
            for q in (50, 90, 95, 99)
        },
        "best_trace": {
            "epoch": observations[best_index]["epoch_or_pass"],
            "optimizer_step": observations[best_index]["optimizer_step"],
            "metric_eV": float(primary_values[best_index]),
        },
        "final_trace": {
            "epoch": last["epoch_or_pass"],
            "optimizer_step": last["optimizer_step"],
            "metric_eV": float(last[primary]),
        },
        "trailing_improvement": windows,
        "last_online_train_mae": {
            "value_normalized_gap": float(online_train_normalized),
            "value_eV_scaled_by_contract_target_std": online_train_eV,
            "unit": "normalized-Gap",
            "target_std_eV": target_std,
            "interpretation": (
                "Online training metric; eV scaling is descriptive and is not a "
                "fixed-checkpoint training evaluation."
            ),
        },
    }


def arm_summary(arm: dict) -> dict:
    manifest = arm["manifest"]
    return {
        "arm_id": arm["arm_id"],
        "selected_prediction_mae_eV": arm["selected_prediction_mae_eV"],
        "mean_signed_error_eV": arm["mean_signed_error_eV"],
        "absolute_error_quantiles_eV": arm["absolute_error_quantiles_eV"],
        "best_trace": arm["best_trace"],
        "final_trace": arm["final_trace"],
        "trailing_improvement": arm["trailing_improvement"],
        "last_online_train_mae": arm["last_online_train_mae"],
        "completed_exposure": manifest["progress"],
        "development_rows": len(arm["target_eV"]),
    }


def main() -> None:
    started = time.perf_counter()
    loaded = {
        family: {
            arm_id: load_arm(family, arm_id)
            for arm_id in (pair["control"], pair["candidate"])
        }
        for family, pair in PAIRS.items()
    }

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), constrained_layout=True)
    pair_results = {}
    for axis, (family, pair) in zip(axes, PAIRS.items()):
        control = loaded[family][pair["control"]]
        candidate = loaded[family][pair["candidate"]]
        for arm in (control, candidate):
            if not np.array_equal(control["source_idx"], arm["source_idx"]):
                raise ValueError(f"{family}: pair source_idx rows differ")
            if not np.array_equal(control["target_eV"], arm["target_eV"]):
                raise ValueError(f"{family}: pair target_eV rows differ")

        delta = candidate["absolute_error_eV"] - control["absolute_error_eV"]
        bootstrap = paired_bootstrap_mean(
            delta, n_bootstrap=N_BOOTSTRAP, seed=SEED
        )
        ci = bootstrap["ci95"]
        wins = int(np.count_nonzero(delta < 0))
        losses = int(np.count_nonzero(delta > 0))
        ties = int(np.count_nonzero(delta == 0))
        gain = -float(bootstrap["delta"])

        pair_results[family] = {
            "control_arm": pair["control"],
            "candidate_arm": pair["candidate"],
            "primary_development_metric": pair["primary_metric"],
            "development_row_alignment": {
                "row_count": int(len(delta)),
                "source_idx_equal_row_by_row": True,
                "target_eV_equal_row_by_row": True,
                "all_target_and_prediction_values_finite": True,
            },
            "candidate_minus_control_absolute_error_eV": bootstrap,
            "candidate_gain_eV": gain,
            "material_gate": {
                "threshold_eV": MATERIAL_GATE_EV,
                "point_estimate_gain_meets_threshold": gain >= MATERIAL_GATE_EV,
                "ci95_upper_below_minus_threshold": bool(ci is not None and ci[1] < -MATERIAL_GATE_EV),
                "interpretation": (
                    "Point estimate is checked against the frozen 0.003 eV threshold; "
                    "the CI is reported separately. Row bootstrap does not estimate "
                    "training-seed variance."
                ),
            },
            "row_outcomes": {
                "candidate_lower_absolute_error": wins,
                "candidate_higher_absolute_error": losses,
                "exact_ties": ties,
            },
            "arms": {
                pair["control"]: arm_summary(control),
                pair["candidate"]: arm_summary(candidate),
            },
        }

        epochs = [row["epoch_or_pass"] for row in control["observations"]]
        axis.plot(epochs, control["curve_eV"], label=f"{pair['control']} (control)")
        candidate_epochs = [row["epoch_or_pass"] for row in candidate["observations"]]
        axis.plot(
            candidate_epochs, candidate["curve_eV"], label=f"{pair['candidate']} (candidate)"
        )
        axis.set_title(pair["title"])
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Development MAE (eV)")
        axis.grid(True, alpha=0.25)
        axis.legend()

    plot_path = OUTPUT_DIR / "development_curves.png"
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)

    elapsed = time.perf_counter() - started
    result = {
        "format": "molgap-structural-bias-norm-saved-analysis-v1",
        "scope": "Saved development predictions and canonical traces only; no checkpoint/model execution.",
        "primary_metrics": {
            family: pair["primary_metric"] for family, pair in PAIRS.items()
        },
        "bootstrap": {
            "function": "molgap.router.paired_bootstrap_mean",
            "delta": "candidate absolute error minus control absolute error",
            "n_bootstrap": N_BOOTSTRAP,
            "seed": SEED,
            "ci": "percentile 95% interval returned by the existing helper",
            "interpretation": "Row bootstrap is not training-seed variance.",
        },
        "computation_seconds": elapsed,
        "analysis_script": "analyze_saved_artifacts.py",
        "development_curves": plot_path.name,
        "pairs": pair_results,
    }
    (OUTPUT_DIR / "analysis_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"summary": str(OUTPUT_DIR / "analysis_summary.json"),
                      "plot": str(plot_path), "computation_seconds": elapsed,
                      "pairs": {
                          family: {
                              "delta": record["candidate_minus_control_absolute_error_eV"]["delta"],
                              "ci95": record["candidate_minus_control_absolute_error_eV"]["ci95"],
                              "candidate_gain_eV": record["candidate_gain_eV"],
                              "wins": record["row_outcomes"]["candidate_lower_absolute_error"],
                              "losses": record["row_outcomes"]["candidate_higher_absolute_error"],
                              "ties": record["row_outcomes"]["exact_ties"],
                          }
                          for family, record in pair_results.items()
                      }}, indent=2))


if __name__ == "__main__":
    main()

