"""Analyze retained development predictions; never load a model or run inference."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

from molgap.research_memory.trace import atomic_write, file_digest, json_bytes

ROOT = Path(__file__).resolve().parents[2]
EXP = Path(__file__).resolve().parent
ARMS = ("k1_pretrained_consistency", "gptrans_g1_bond_local_ema999")


def main() -> None:
    spec = importlib.util.spec_from_file_location(
        "retained_pair", ROOT / "experiments/pcqm_gptrans_100k_transfer_control/analyze_pair.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    raw = ROOT / "platforms/_records/kaggle/staging/pcqm_k1_gptrans_package_transfer_500k/v5-checked-artifacts"
    checked = EXP / "submission_v5/terminal_inspection"
    predictions, traces, bindings = {}, {}, {}
    for arm in ARMS:
        stage = json.loads((checked / "evidence/stages" / arm / "stage_manifest.json").read_text())
        path = raw / arm / "best_predictions.pt"
        digest = file_digest(path)
        # Stage manifest declarations differ from the canonical prediction keys only.
        if digest not in json.dumps(stage):
            raise ValueError("Prediction SHA256 absent from accepted stage manifest")
        pred = torch.load(path, map_location="cpu", weights_only=False)
        predictions[arm] = {"source_idx": pred["source_idx"],
                            "prediction_eV": pred["prediction"], "target_eV": pred["target"]}
        traces[arm] = json.loads((checked / "evidence/stages" / arm / "trace.json").read_text())["epochs"]
        bindings[arm] = {"selected_predictions_sha256": digest,
                         "stage_manifest_ref": (checked / "evidence/stages" / arm / "stage_manifest.json").relative_to(ROOT).as_posix(),
                         "stage_manifest_sha256": file_digest(checked / "evidence/stages" / arm / "stage_manifest.json")}
    k1, gp = (predictions[arm] for arm in ARMS)
    component = module.paired_metrics(k1, gp, 500000, 550000)
    stronger = gp if component["candidate_mae_eV"] <= component["reference_mae_eV"] else k1
    # Promote vectors to float64 before arithmetic; no fitting or selection of weights.
    fusion = {"source_idx": gp["source_idx"], "target_eV": gp["target_eV"],
              "prediction_eV": .5 * (k1["prediction_eV"].double() + gp["prediction_eV"].double())}
    paired = module.paired_metrics(stronger, fusion, 500000, 550000)
    gate = paired["paired_gain_eV"] >= .001 and paired["row_bootstrap_95pct_eV"][0] > 0
    live = torch.load(raw / ARMS[1] / "best_live_predictions.pt", map_location="cpu", weights_only=False)
    ema_snapshot = module.paired_metrics(
        {"source_idx": live["source_idx"], "prediction_eV": live["prediction"], "target_eV": live["target"]},
        gp, 500000, 550000,
    )
    target = gp["target_eV"].view(-1).double().numpy()
    residuals = {name: pred["prediction_eV"].view(-1).double().numpy() - target
                 for name, pred in (("k1", k1), ("gptrans", gp), ("fixed50", fusion))}
    errors = {name: np.abs(value) for name, value in residuals.items()}
    opposite = residuals["k1"] * residuals["gptrans"] < 0
    windows = {}
    for arm, rows in traces.items():
        best = min(rows, key=lambda row: row["development_mae_eV"])
        windows[arm] = {
            "best_epoch_one_based": best["epoch"] + 1,
            "best_dev_mae_eV": best["development_mae_eV"],
            "final_dev_mae_eV": rows[-1]["development_mae_eV"],
            "final_minus_best_meV": 1000 * (rows[-1]["development_mae_eV"] - best["development_mae_eV"]),
            "epochs_since_best": 60 - best["epoch"] - 1,
            "measured_one_worker_epoch_seconds": sum(row["seconds"] for row in rows),
            "epoch_windows": [{"first_epoch": start + 1, "last_epoch": stop,
                               "mean_dev_mae_eV": float(np.mean([r["development_mae_eV"] for r in rows[start:stop]])),
                               "minimum_dev_mae_eV": min(r["development_mae_eV"] for r in rows[start:stop]),
                               "mean_recorded_train_mae_eV": float(np.mean([r["train_mae_eV"] for r in rows[start:stop]]))}
                              for start, stop in ((0, 10), (20, 30), (40, 50), (50, 60))],
            "train_metric_limit": "Training dropout supervised L1, different prediction mode from clean development; no fit diagnosis from their gap alone.",
        }
    result = {
        "format": "molgap-retained-500k-package-terminal-analysis-v1",
        "role": "selection_used_internal_development_500000_550000",
        "prediction_bindings": bindings,
        "k1_vs_gptrans": component,
        "fixed50_vs_stronger_component": paired,
        "frozen_complementarity_gate": {"minimum_gain_eV": .001, "positive_row_95pct_lower_bound": True, "passed": gate},
        "selected_gptrans_ema_vs_same_checkpoint_live": ema_snapshot,
        "ema_diagnostic_limit": "Within-selected-checkpoint observation, not a randomized EMA intervention or independent checkpoint-selection comparison.",
        "residual_diagnostics": {
            "signed_error_pearson": float(np.corrcoef(residuals["k1"], residuals["gptrans"])[0, 1]),
            "absolute_error_pearson": float(np.corrcoef(errors["k1"], errors["gptrans"])[0, 1]),
            "opposite_error_sign_fraction": float(opposite.mean()),
            "k1_better_row_fraction": float((errors["k1"] < errors["gptrans"]).mean()),
            "signed_error_mean_eV": {name: float(values.mean()) for name, values in residuals.items()},
            "absolute_error_quantiles_eV": {name: dict(zip(("median", "p90", "p95", "p99"), map(float, np.quantile(values, [.5, .9, .95, .99])))) for name, values in errors.items()},
            "label_informed_best_of_two_mae_eV": float(np.minimum(errors["k1"], errors["gptrans"]).mean()),
            "oracle_limit": "Label-informed upper bound only, not a learnable routing result or promotion evidence.",
            "fixed50_gain_on_opposite_sign_rows_eV": float((errors["gptrans"] - errors["fixed50"])[opposite].mean()),
            "fixed50_gain_on_same_sign_rows_eV": float((errors["gptrans"] - errors["fixed50"])[~opposite].mean()),
        },
        "trace_diagnostics": windows,
        "limits": ["Development selected checkpoints; row bootstrap does not measure seed variance or independent evaluation.",
                   "No qualified prelaunch component reference; whole-package contextual gains do not identify module causality.",
                   "No model execution, protected-role access, fusion weight fitting or remote training in this analysis."],
    }
    atomic_write(EXP / "terminal_acceptance/analysis.json", json_bytes(result))
    print(json.dumps({"paired": paired, "gate": gate, "residuals": result["residual_diagnostics"], "traces": windows}, indent=2))


if __name__ == "__main__":
    main()
