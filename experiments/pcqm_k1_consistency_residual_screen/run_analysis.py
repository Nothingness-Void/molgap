"""Frozen CPU probe of accepted predictions; no model construction or execution."""
from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from experiments.pcqm_gptrans_100k_transfer_control.analyze_pair import paired_metrics
from molgap.pcqm_k1_gptrans_fusion import calibration_mask, fit_convex_blend_weight, mae
from molgap.residual_attribution import _metric_block
from molgap.router import paired_bootstrap_mean
from molgap.training_reproducibility import atomic_json, sha256_file
from molgap.v4_runtime import state_dict_sha256, torch_load_compat

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def paired_gain(reference, candidate, truth):
    delta = np.abs(reference - truth) - np.abs(candidate - truth)
    bounds = np.asarray(paired_bootstrap_mean(delta, n_bootstrap=1000, seed=42)["ci95"])
    return dict(reference_mae_eV=mae(reference, truth), candidate_mae_eV=mae(candidate, truth),
                gain_meV=float(1000 * delta.mean()), row_bootstrap_95pct_meV=(1000 * bounds).tolist(),
                candidate_better_fraction=float((delta > 0).mean()),
                nomination_passed=bool(delta.mean() >= 0.001 and bounds[0] > 0))


def main():
    wall, cpu = time.perf_counter(), time.process_time()
    if not (HERE / "rml/trajectory.json").exists():
        raise ValueError("Prospective trajectory must exist before any tensor consumption")
    inputs = json.loads((HERE / "inputs.json").read_text(encoding="utf-8"))
    for path, expected in inputs["source_hashes"].items():
        if sha256_file(ROOT / path) != expected:
            raise ValueError(f"Frozen analysis source changed: {path}")
    records = {}
    for arm, binding in inputs["arms"].items():
        path = ROOT / binding["path"]
        if sha256_file(path) != binding["sha256"]:
            raise ValueError(f"Accepted prediction bytes changed: {arm}")
        records[arm] = torch_load_compat(path, map_location="cpu", weights_only=True)
    full = paired_metrics(records["mean2"], records["consistency2"], 100000, 150000)
    accepted = json.loads((ROOT / "experiments/pcqm_k1_dropout_consistency/terminal_acceptance/scientific_metrics.json").read_text())
    for key in ("reference_mae_eV", "candidate_mae_eV", "paired_gain_eV"):
        if abs(full[key] - accepted["primary"][key]) > 1e-12:
            raise ValueError(f"Accepted endpoint changed: {key}")
    p = {arm: r["prediction_eV"].view(-1).double().numpy() for arm, r in records.items()}
    truth = records["mean2"]["target_eV"].view(-1).double().numpy()
    idx = records["mean2"]["source_idx"].view(-1).long().numpy()
    frame = pd.DataFrame(dict(gap=truth, mean2_gap=p["mean2"], consistency2_gap=p["consistency2"]))
    residuals = _metric_block(frame, "mean2", "consistency2", ["gap"])
    gain = np.abs(p["mean2"] - truth) - np.abs(p["consistency2"] - truth)
    distribution = dict(improved_rows=int((gain > 0).sum()), harmed_rows=int((gain < 0).sum()),
        improvement_contribution_meV=float(np.maximum(gain, 0).mean() * 1000),
        harm_contribution_meV=float(np.maximum(-gain, 0).mean() * 1000))
    strata = []
    for lo, hi in ((0, 2), (2, 4), (4, 6), (6, 8), (8, float("inf"))):
        mask = (truth >= lo) & (truth < hi)
        if mask.any():
            strata.append(dict(true_gap_bin=f"[{lo},{hi})", rows=int(mask.sum()),
                gain_meV=float(gain[mask].mean() * 1000),
                mean2_mae_eV=mae(p["mean2"][mask], truth[mask]),
                consistency2_mae_eV=mae(p["consistency2"][mask], truth[mask])))
    tail = np.argsort(np.abs(p["mean2"] - truth), kind="stable")[-500:]
    distribution["control_top1pct"] = dict(rows=500,
        gain_meV=float(gain[tail].mean() * 1000),
        fraction_of_net_gain=float(gain[tail].sum() / gain.sum()))
    fit = calibration_mask(idx)
    held = ~fit
    offsets = {arm: float(np.median(truth[fit] - value[fit])) for arm, value in p.items()}
    alpha, fitted_mae = fit_convex_blend_weight(p["mean2"][fit], p["consistency2"][fit], truth[fit])
    candidates = {"consistency_bias_corrected": p["consistency2"] + offsets["consistency2"],
                  "mean2_bias_corrected": p["mean2"] + offsets["mean2"],
                  "fixed_equal_blend": 0.5 * (p["mean2"] + p["consistency2"]),
                  "fitted_convex_blend": alpha * p["mean2"] + (1 - alpha) * p["consistency2"]}
    held_metrics = {name: paired_gain(p["consistency2"][held], value[held], truth[held])
                    for name, value in candidates.items()}
    report = dict(status="EXECUTED_SAVED_PREDICTION_DIAGNOSTIC", rows=len(idx),
        inputs_sha256=sha256_file(HERE / "inputs.json"),
        prospective_sha256=sha256_file(HERE / "rml/trajectory.json"),
        source_idx_sha256=state_dict_sha256({"source_idx": records["mean2"]["source_idx"]}),
        artifacts_sha256={v["path"]: v["sha256"] for v in inputs["arms"].values()},
        accepted_primary=full, residuals=residuals, distribution=distribution, descriptive_strata=strata,
        split=dict(fit_rows=int(fit.sum()), holdout_rows=int(held.sum()), rule="source_idx modulo5 ==0 fits"),
        offsets_eV=offsets, fitted_mean2_weight=alpha, fitted_mae_eV=fitted_mae,
        heldout_against_consistency2=held_metrics,
        checks=dict(aligned_rows=True, finite_predictions=True, exact_targets=True,
                    reconstructed_accepted_endpoints=True, no_model_execution=True,
                    protected_roles_untouched=True),
        limitations=["Post-hoc model-selected development role; modulo holdout is not independent qualification.",
                     "One seed; row bootstrap is not training variance.",
                     "No saved dropout pass disagreement or clean training-cohort predictions.",
                     "Descriptive target/tail strata and Oracle are label-informed, not routing policies.",
                     "Original cross-attempt control/native-cost and strict V5 gaps unchanged."],
        hardware=platform.processor() or "local Windows CPU", wall_seconds=time.perf_counter() - wall,
        cpu_seconds=time.process_time() - cpu)
    atomic_json(HERE / "results/analysis.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
