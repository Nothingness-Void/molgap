"""Tabulate accepted predictions and draw retained curves; no fitting or inference."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OWNER = HERE.parent


def main():
    helper_path = OWNER / "kaggle3_reconciliation_v1/analyze_saved_predictions.py"
    spec = importlib.util.spec_from_file_location("accepted_width_analysis", helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    accepted = helper.read_json(OWNER / "kaggle3_reconciliation_v1/scientific_metrics.json")
    records = {}
    for side in ("reference", "candidate"):
        item = accepted["artifacts"][side]["predictions"]
        path = ROOT / item["path"]
        if helper.sha256(path) != item["sha256"]:
            raise ValueError("Accepted prediction identity differs")
        records[side] = torch.load(path, map_location="cpu", weights_only=False)
    ref, cand = records["reference"], records["candidate"]
    assert torch.equal(ref["source_idx"], cand["source_idx"])
    assert torch.equal(ref["target_eV"], cand["target_eV"])
    target = ref["target_eV"].double().numpy()
    predictions = {side: record["prediction_eV"].double().numpy() for side, record in records.items()}
    assert all(np.isfinite(a).all() for a in (target, *predictions.values()))
    errors = {side: np.abs(values - target) for side, values in predictions.items()}
    delta = errors["candidate"] - errors["reference"]
    result = dict(scope="posthoc_descriptive_tabulation_of_already_accepted_saved_predictions_no_fit_no_model_execution",
        sign_definition="positive means candidate worse", rows=len(target), mean_delta_eV=float(delta.mean()),
        source_artifacts={side: accepted["artifacts"][side]["predictions"] for side in records},
        groups={}, gap_bins=[], prediction_change=dict(
            mean_absolute_eV=float(np.abs(predictions["candidate"] - predictions["reference"]).mean()),
            pearson_correlation=float(np.corrcoef(predictions["reference"], predictions["candidate"])[0, 1])),
        limitations=["posthoc bins are descriptive, not chemical structure slices or causal evidence",
                     "single seed and runtime differences remain", "no calibration, routing or new model comparison"])
    for label, mask in [("candidate_better", delta < 0), ("candidate_worse", delta > 0), ("ties", delta == 0)]:
        result["groups"][label] = dict(rows=int(mask.sum()), fraction=float(mask.mean()),
            mean_delta_eV=float(delta[mask].mean()) if mask.any() else None,
            contribution_to_full_mean_eV=float(delta[mask].sum() / len(target)))
    for low, high in [(-np.inf, 2), (2, 4), (4, 6), (6, 8), (8, np.inf)]:
        mask = (target >= low) & (target < high)
        if mask.any():
            result["gap_bins"].append(dict(lower_eV=float(low) if np.isfinite(low) else None,
                upper_eV=float(high) if np.isfinite(high) else None, rows=int(mask.sum()),
                reference_mae_eV=float(errors["reference"][mask].mean()),
                candidate_mae_eV=float(errors["candidate"][mask].mean()),
                candidate_worse_by_eV=float(delta[mask].mean()),
                candidate_better_fraction=float((delta[mask] < 0).mean())))
    (HERE / "descriptive_stats.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    draw(accepted["paired_curve"])
    print(json.dumps({"rows": len(target), "candidate_worse_by_eV": float(delta.mean()), "model_execution": False}))


def draw(curve):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    epochs = [point["epoch"] for point in curve]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    for side, label, color in [("reference", "192 dimensions", "#2166ac"), ("candidate", "256 dimensions", "#b35806")]:
        axes[0].plot(epochs, [point[f"{side}_development_mae_eV"] for point in curve], label=label, color=color, linewidth=2)
    axes[0].set(title="Same exposure: development MAE", xlabel="Epoch", ylabel="MAE (eV, lower is better)")
    axes[0].legend(frameon=False)
    gains = [1000 * point["development_gain_eV"] for point in curve]
    axes[1].plot(epochs, gains, color="#6a51a3", linewidth=2)
    axes[1].axhline(0, color="#777777", linewidth=1)
    axes[1].scatter([epochs[-1]], [gains[-1]], color="#b35806", zorder=4)
    axes[1].set(title="Reference MAE minus candidate MAE", xlabel="Epoch", ylabel="Gain (meV, positive favors 256)")
    for axis in axes:
        axis.grid(alpha=.2)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_xlim(1, 40)
    fig.suptitle("K1 width screen: 100K training / 50K development, seed42", fontsize=13)
    fig.savefig(HERE / "curves.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
