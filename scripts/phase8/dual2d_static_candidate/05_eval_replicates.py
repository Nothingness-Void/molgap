"""Evaluate complete Local/GPS/Fusion seeds for the dual-2D stop gate."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch

from molgap.constants import RESULTS_DIR
from molgap.router import paired_bootstrap_mean
from molgap.dual2d_static_candidate.diagnostics import router_diagnostics
from molgap.dual2d_static_candidate.dual2d_training import predict_dual2d, train_dual2d
from molgap.dual2d_static_candidate.evaluation import apply_static_weights, fit_static_weights, metrics


OUT_DIR = RESULTS_DIR / "phase8" / "dual2d_static_candidate"


def main() -> None:
    original = pd.read_parquet(OUT_DIR / "pilot_30k.parquet").set_index("source_idx")
    result = {
        "experiment": "dual-2D static candidate complete-stack dual-2D three-seed audit",
        "geometry_expert_used": False,
        "seeds": {},
    }
    for seed in (42, 43, 44):
        with np.load(OUT_DIR / f"dual2d_features_seed{seed}.npz") as raw:
            source_idx = raw["source_idx"]
            local = raw["local_embedding"].copy()
            global_value = raw["global_embedding"].copy()
            expert_predictions = np.stack([
                raw["local_prediction"], raw["global_prediction"]
            ], axis=-1).astype(np.float32)
        table = original.loc[source_idx].reset_index()
        targets = table[["homo", "lumo", "gap"]].to_numpy(dtype=np.float32)
        splits = {
            split: np.flatnonzero(table.split.eq(split).to_numpy())
            for split in ("train", "validation", "internal_test")
        }
        for embedding in (local, global_value):
            mean = embedding[splits["train"]].mean(axis=0)
            std = embedding[splits["train"]].std(axis=0).clip(1e-6)
            embedding -= mean
            embedding /= std
        validation, test = splits["validation"], splits["internal_test"]
        static_weights = fit_static_weights(targets[validation], expert_predictions[validation])
        predictions = {
            "local": expert_predictions[test, :, 0],
            "global": expert_predictions[test, :, 1],
            "equal_average": expert_predictions[test].mean(axis=-1),
            "static_weights": apply_static_weights(expert_predictions[test], static_weights),
        }
        best_single = min(("local", "global"), key=lambda name: metrics(targets[test], predictions[name])["gap"]["mae"])
        record = {
            "static_weights": static_weights.tolist(),
            "best_single": best_single,
            "metrics": {name: metrics(targets[test], value) for name, value in predictions.items()},
            "learned": {},
        }
        reference = predictions[best_single]
        for kind in ("concat_fusion", "soft_gate", "static_centered_gate"):
            model, training = train_dual2d(
                kind=kind,
                local_embedding=local,
                global_embedding=global_value,
                expert_predictions=expert_predictions,
                targets=targets,
                train_indices=splits["train"],
                validation_indices=validation,
                validation_sources=table.loc[validation, "sampling_source"].to_numpy(),
                seed=seed,
                prior_weights=(static_weights if kind == "static_centered_gate" else None),
            )
            prediction, weights = predict_dual2d(
                model=model,
                kind=kind,
                local_embedding=local,
                global_embedding=global_value,
                expert_predictions=expert_predictions,
                indices=test,
                target_mean=np.asarray(training["target_mean"], dtype=np.float32),
                target_std=np.asarray(training["target_std"], dtype=np.float32),
            )
            predictions[kind] = prediction
            entry = {
                "training": training,
                "metrics": metrics(targets[test], prediction),
            }
            if weights is not None:
                entry["gate_diagnostics"] = router_diagnostics(torch.from_numpy(weights))
            record["learned"][kind] = entry
        for name in ("equal_average", "static_weights", "concat_fusion", "soft_gate", "static_centered_gate"):
            value = predictions[name]
            delta = np.abs(value[:, 2] - targets[test, 2]) - np.abs(
                reference[:, 2] - targets[test, 2]
            )
            comparison = paired_bootstrap_mean(delta, seed=seed + len(name))
            record.setdefault("comparisons", {})[name] = {
                "gap_improvement_vs_best_single_eV": -comparison["delta"],
                "paired_bootstrap": comparison,
            }
        result["seeds"][str(seed)] = record
        print(seed, {name: record["comparisons"][name]["gap_improvement_vs_best_single_eV"] for name in record["comparisons"]}, flush=True)
    result["summary"] = {}
    for name in ("equal_average", "static_weights", "concat_fusion", "soft_gate", "static_centered_gate"):
        gains = [
            result["seeds"][str(seed)]["comparisons"][name]["gap_improvement_vs_best_single_eV"]
            for seed in (42, 43, 44)
        ]
        result["summary"][name] = {
            "gains_eV": gains,
            "mean_gain_eV": float(np.mean(gains)),
            "all_seeds_improve": bool(all(value > 0 for value in gains)),
            "all_seeds_reach_0.001_eV": bool(all(value >= 0.001 for value in gains)),
        }
    pass_names = [
        name for name, summary in result["summary"].items()
        if summary["all_seeds_reach_0.001_eV"]
    ]
    dynamic_pass = result["summary"]["static_centered_gate"]["all_seeds_reach_0.001_eV"]
    result["decision"] = {
        "threshold_eV": 0.001,
        "passing_methods": pass_names,
        "dual2d_complementarity_pass": bool(pass_names),
        "dynamic_gate_pass": dynamic_pass,
        "continue_dynamic_moe": dynamic_pass,
        "retain_static_dual2d_candidate": "static_weights" in pass_names,
        "sealed_metrics_opened": False,
    }
    (OUT_DIR / "dual2d_three_seed_metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps({"summary": result["summary"], "decision": result["decision"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
