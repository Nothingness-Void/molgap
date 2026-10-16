"""Frozen-gate analysis of retained predictions; no model execution."""
import json
from pathlib import Path

import torch

from molgap.research_memory.trace import atomic_write, file_digest, json_bytes
from molgap.router import paired_bootstrap_mean


def main():
    root = Path.cwd()
    base = root / "experiments/pcqm_k1_prenorm_norm_100k"
    output = base / "terminal_acceptance"
    storage = root / "platforms/_records/kaggle/training/k1_prenorm_norm_100k_s42_v1"
    retained = json.loads((base / "reference_binding/post_norm/manifest.json").read_text())
    for entry in retained["files"]:
        assert file_digest(root / entry["path"]) == entry["sha256"], entry["path"]
    folders = {
        "pre_bn": storage / "family_outputs/k1_pre_bn",
        "pre_ada": storage / "family_outputs/k1_pre_ada",
        "post_bn": base / "reference_binding/post_norm/k1_single_bn",
        "post_ada": base / "reference_binding/post_norm/k1_ada_rmsnorm",
    }
    predictions, traces, arms = {}, {}, {}
    for name, folder in folders.items():
        payload = torch.load(folder / "development_predictions.pt", map_location="cpu", weights_only=True)
        for key in ("source_idx", "target_eV", "prediction_eV"):
            assert payload[key].shape == (50000,) and torch.isfinite(payload[key]).all()
        if predictions:
            reference = next(iter(predictions.values()))
            assert torch.equal(payload["source_idx"], reference["source_idx"])
            assert torch.equal(payload["target_eV"], reference["target_eV"])
        predictions[name] = payload
        trace = json.loads((folder / "canonical_trace.json").read_text())
        rows = [row for row in trace["observations"] if row["event"] == "observation"]
        assert len(rows) == 40 and rows[-1]["optimizer_step"] == 31240
        assert rows[-1]["sample_presentations"] == 3998720
        traces[name] = rows
        best = min(rows, key=lambda row: row["live_dev_metric"])
        errors = (payload["prediction_eV"].double() - payload["target_eV"].double()).abs()
        arms[name] = {
            "mae_eV": errors.mean().item(), "selected_pass": best["epoch_or_pass"],
            "selected_step": best["optimizer_step"], "last_live_mae_eV": rows[-1]["live_dev_metric"],
            "live_30_mae_eV": rows[29]["live_dev_metric"],
            "live_30_to_last_improvement_eV": rows[29]["live_dev_metric"] - rows[-1]["live_dev_metric"],
            "last_online_train_mae_eV": rows[-1]["live_train_metric"] * 1.275090217590332,
            "prediction_sha256": file_digest(folder / "development_predictions.pt"),
        }
        if all(row.get("ema_dev_metric") is not None for row in rows):
            ema = min(rows, key=lambda row: row["ema_dev_metric"])
            arms[name].update(secondary_best_ema_mae_eV=ema["ema_dev_metric"],
                              secondary_best_ema_pass=ema["epoch_or_pass"])
        if name.startswith("pre_"):
            manifest = json.loads((folder / "output_manifest.json").read_text())
            arms[name]["invocation_wall_seconds"] = next(
                cost["value"] for cost in manifest["costs"] if cost["metric"] == "wall_seconds")
    comparisons = {}
    pairs = (("primary", "pre_ada", "pre_bn"),
             ("context_pre_bn_minus_post_bn", "pre_bn", "post_bn"),
             ("context_pre_ada_minus_post_ada", "pre_ada", "post_ada"))
    for label, candidate, reference in pairs:
        def error(name):
            value = predictions[name]
            return (value["prediction_eV"].double() - value["target_eV"].double()).abs().numpy()
        delta = error(candidate) - error(reference)
        comparisons[label] = {
            **paired_bootstrap_mean(delta, n_bootstrap=10000, seed=42),
            "candidate": candidate, "reference": reference, "rows": len(delta),
            "candidate_better_rows": int((delta < 0).sum()),
            "reference_better_rows": int((delta > 0).sum()), "tie_rows": int((delta == 0).sum()),
            "scope": "same_job_frozen_gate" if label == "primary" else "cross_job_context_only",
        }
    primary = comparisons["primary"]
    primary["noninferiority_passed"] = primary["delta"] <= .0005 and primary["ci95"][1] <= .0005
    primary["material_superiority_passed"] = primary["delta"] <= -.001 and primary["ci95"][1] < 0
    entry = json.loads((storage / "metadata/experiment/allocation_entry_observation.json").read_text())
    assert entry["status"] == "complete" and entry["observed_gpu_count"] == 2
    assert abs(entry["allocated_device_seconds"] - entry["monotonic_seconds"] * 2) < 1e-6
    result = {
        "format": "molgap-retained-pair-analysis-v1", "kernel_id": 138980611, "version": 1,
        "arms": arms, "comparisons": comparisons, "allocation_entry": entry,
        "entry_assigned_t4_device_hours": entry["allocated_device_seconds"] / 3600,
        "candidate_over_reference_invocation_wall_ratio": arms["pre_ada"]["invocation_wall_seconds"] / arms["pre_bn"]["invocation_wall_seconds"],
        "scientific_outcome": "NEGATIVE_UNDER_CONTRACT" if not (primary["noninferiority_passed"] or primary["material_superiority_passed"]) else "POSITIVE_UNDER_CONTRACT",
        "limits": ["paired row bootstrap is not seed variance", "historical placement comparisons have source/job/final-norm confounding", "online train MAE is not a same-checkpoint generalization gap", "no new training or inference"],
    }
    atomic_write(output / "analysis_summary.json", json_bytes(result))
    print(json.dumps(result, indent=2))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    colors = {"pre_bn": "#176b87", "pre_ada": "#c35830", "post_bn": "#4f8a43", "post_ada": "#8661a0"}
    for name, rows in traces.items():
        x = [row["optimizer_step"] for row in rows]
        for ax in axes:
            ax.plot(x, [row["live_dev_metric"] for row in rows], label=name.replace("_", " "), color=colors[name])
        if name.startswith("pre_"):
            axes[1].plot(x, [row["ema_dev_metric"] for row in rows], "--", color=colors[name], alpha=.7, label=name.replace("_", " ") + " EMA (secondary)")
    axes[0].set(title="Live development curves", xlabel="Optimizer steps", ylabel="Gap MAE (eV)", ylim=(.135, .35))
    axes[1].set(title="Late training: historical cells are context only", xlabel="Optimizer steps", ylabel="Gap MAE (eV)", xlim=(15620, 31240), ylim=(.138, .165))
    for ax in axes:
        ax.grid(alpha=.2)
    axes[1].legend(fontsize=7)
    fig.savefig(output / "development_curves.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
