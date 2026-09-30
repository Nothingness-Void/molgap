"""Thin, frozen scientific caller of existing saved-prediction/Oracle helpers."""
from __future__ import annotations

import importlib.util
import json
import shutil
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from molgap.hierarchical_oracle import hierarchical_oracle_analysis
from molgap.multi2d import targetwise_oracle
from molgap.training_reproducibility import atomic_json, sha256_file

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
GEOM = Path("C:/Users/17449/.codex/worktrees/geometry-v4-500k/molgap")
LEGACY = Path("D:/文档/molgap-exp/molgap-500k-v4-evidence")


def main():
    if (HERE / "analysis.json").exists():
        raise RuntimeError("Reconcile retained analysis before a repeat")
    wall, cpu = time.perf_counter(), time.process_time()
    helper_path = ROOT / "experiments/pcqm_500k_v4_evidence/analyze_local_ablation.py"
    spec = importlib.util.spec_from_file_location("matched_v4_analysis", helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    refs = GEOM / "platforms/_records/kaggle/training/pcqm_500k_v4_evidence_reference_predictions"
    geom = GEOM / "platforms/_records/kaggle/training/geometry_v4_500k_continuation_attempt_003/output/evidence"
    inputs = {
        "gptrans_2d": (refs / "gptrans__best_predictions.pt", "0608084eb2a7a90923a652235572ee69578c4e138d5df3c457e5170f7da613bf"),
        "k1_2d": (refs / "neural_atom_k1__best_predictions.pt", "68fba785a0b028a8445fe8d94d348df2c3a8d7d8caab708acb574d13cac9831b"),
        "edge_state": (LEGACY / "platforms/_records/kaggle/training/pcqm_500k_v4_stage6/full-gps-final-epoch-v8/evidence/full_gps/best_predictions.pt", "c31fafb651fcfef08381627ac86f6639cae91f664ae7a39574d4ca0383b91ce0"),
        "gptrans_distance": (geom / "gptrans_distance_only/best_predictions.pt", "85488eb225721274e37097554a9ceb81371267d5b404e939bcfb96403d982105"),
        "k1_geometry": (geom / "k1_distance_angle/best_predictions.pt", "a5363aaf2654996e169cf62277db68e42221a08102d94b905bec9aeb66ff0a23"),
    }
    retained = ROOT / "platforms/_records/local/expert_oracle_feasibility_20260930"
    retained.mkdir(parents=True, exist_ok=True)
    payloads, bindings = {}, {}
    for name, (path, digest) in inputs.items():
        if sha256_file(path) != digest:
            raise ValueError(f"Unaccepted bytes: {name}")
        local = retained / f"{name}.pt"
        shutil.copyfile(path, local)
        item = helper._load(local)
        if not np.array_equal(item["source_idx"], np.arange(500000, 550000)):
            raise ValueError(f"Wrong ordered source rows: {name}")
        payloads[name] = item
        bindings[name] = dict(sha256=digest, retained_path=local.relative_to(ROOT).as_posix(), source_path=str(path))
    anchor = payloads["k1_2d"]
    if any(not np.array_equal(x["target"], anchor["target"]) for x in payloads.values()):
        raise ValueError("Target identity mismatch")
    y = anchor["target"].astype(np.float64)
    p = {n: x["prediction"].astype(np.float64) for n, x in payloads.items()}
    original = (p["k1_2d"] + p["gptrans_2d"]) / 2
    geometry = (p["k1_geometry"] + p["gptrans_distance"]) / 2
    mae = lambda prediction: float(np.abs(prediction - y).mean())
    oracles = {}
    for name, names, fallback in (
        ("original_pair", ["k1_2d", "gptrans_2d"], False),
        ("five_arms", list(p), False),
        ("five_arms_with_original_blend_fallback", list(p), True),
    ):
        choices = {n: p[n][:, None] for n in names}
        if fallback:
            choices["original_blend"] = original[:, None]
        oracle, winners = targetwise_oracle(y[:, None], choices)
        oracles[name] = dict(mae_eV=mae(oracle[:, 0]), chosen_fraction={n: float((winners[:, 0] == i).mean()) for i, n in enumerate(choices)}, label_informed_not_deployable=True)
    budget, _ = hierarchical_oracle_analysis(y[:, None], p["k1_2d"][:, None], p["gptrans_2d"][:, None], target_names=("gap",), budgets=(.05, .10, .20, .50, 1.), base_encoder_passes=1., expert_encoder_passes=1.)

    k, g = p["k1_2d"], p["gptrans_2d"]
    delta = k - g
    x = np.column_stack([k, g, delta, np.abs(delta), (k + g) / 2])
    k_error, g_error = np.abs(k-y), np.abs(g-y)
    winner = (k_error < g_error).astype(int)
    oracle_alpha = np.zeros_like(y)
    np.divide(y-g, delta, out=oracle_alpha, where=delta != 0)
    oracle_alpha = np.clip(oracle_alpha, 0, 1)
    folds = anchor["source_idx"].astype(np.int64) % 5
    predictions = {n: np.empty_like(y) for n in ("global_crossfit", "hard_gate", "soft_gate")}
    probabilities = np.empty_like(y)
    reports = []
    for fold in range(5):
        train, test = folds != fold, folds == fold
        weight = helper._weighted_median_weight(k[train], g[train], y[train])
        predictions["global_crossfit"][test] = weight*k[test] + (1-weight)*g[test]
        hard = make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=500))
        hard.fit(x[train], winner[train])
        probabilities[test] = hard.predict_proba(x[test])[:, 1]
        predictions["hard_gate"][test] = np.where(probabilities[test] >= .5, k[test], g[test])
        soft = HistGradientBoostingRegressor(loss="absolute_error", max_iter=100, max_leaf_nodes=7, max_depth=3, min_samples_leaf=200, l2_regularization=1, learning_rate=.1, random_state=42)
        soft.fit(x[train], oracle_alpha[train], sample_weight=np.abs(delta[train]))
        alpha = np.clip(soft.predict(x[test]), 0, 1)
        predictions["soft_gate"][test] = g[test] + alpha*delta[test]
        reports.append(dict(fold=fold, n=int(test.sum()), global_weight_k1=weight, mae_eV={n: float(np.abs(v[test]-y[test]).mean()) for n,v in predictions.items()}))
    comparator_error = np.abs(predictions["global_crossfit"]-y)
    gates = {}
    for name in ("hard_gate", "soft_gate"):
        gain = comparator_error - np.abs(predictions[name]-y)
        paired = helper._bootstrap(gain)
        positive_folds = sum(r["mae_eV"][name] < r["mae_eV"]["global_crossfit"] for r in reports)
        gates[name] = dict(mae_eV=mae(predictions[name]), improvement_over_global=paired, improving_folds=positive_folds, nomination_rule_pass=bool(paired["mean_eV"] >= .001 and paired["bootstrap_95pct_eV"][0] > 0 and positive_folds >= 4))
    opposition = (k-y)*(g-y) < 0
    result = dict(format="molgap-expert-oracle-feasibility-v1", rows=len(y), identity=dict(ordered_source_idx=[500000,549999], exact_target_alignment=True, inputs=bindings), scope=dict(encoder_training=False, model_inference=False, diagnostic_fit=True, consumed_development=True, official_validation_read=False, test_dev_read=False, test_challenge_read=False, geometry_training_replay_qualified=False), individual_mae_eV={n:mae(v) for n,v in p.items()}, fixed_blend_mae_eV=dict(original=mae(original), geometry=mae(geometry)), oracles=oracles, conditional_pair_oracle=budget, crossfit=dict(global_mae_eV=mae(predictions["global_crossfit"]), gates=gates, hard_winner_auc=float(roc_auc_score(winner, probabilities)), folds=reports, semantics="Only gate labels cross-fitted; models were selected on this consumed development role; exploratory, not independent promotion"), complementarity=dict(opposite_signed_error_fraction=float(opposition.mean()), signed_residual_correlation=float(np.corrcoef(k-y,g-y)[0,1]), fixed_blend_minus_discrete_oracle_eV=mae(original)-oracles["original_pair"]["mae_eV"], fixed_blend_better_than_both_fraction=float((np.abs(original-y) < np.minimum(k_error,g_error)).mean())), provenance=dict(analyzer_sha256=sha256_file(Path(__file__)), protocol_sha256=sha256_file(HERE/"protocol.md"), helper_sha256=sha256_file(helper_path), bootstrap_seed=helper.BOOTSTRAP_SEED, bootstrap_replicates=helper.BOOTSTRAP_REPLICATES, bootstrap_semantics="paired row sampling, not training-seed uncertainty", encoder_passes_semantics="hypothetical workload counts, not measured native timing"))
    np.savez_compressed(retained / "gate_oof_predictions.npz", source_idx=anchor["source_idx"], target=y, **predictions)
    result["retained_oof"] = dict(path=(retained/"gate_oof_predictions.npz").relative_to(ROOT).as_posix(), sha256=sha256_file(retained/"gate_oof_predictions.npz"))
    result["local_cost"] = dict(wall_seconds=time.perf_counter()-wall, process_cpu_seconds=time.process_time()-cpu, timing_scope="loading/copying/analysis/output prediction writing; imports and literature review excluded")
    atomic_json(HERE / "analysis.json", result)
    print(json.dumps({n:result[n] for n in ("individual_mae_eV", "fixed_blend_mae_eV", "oracles", "crossfit", "complementarity", "local_cost")}, indent=2))


if __name__ == "__main__":
    main()
