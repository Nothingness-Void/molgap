"""Evaluate Local+GPS equal/static/dense/gated dual-2D controls."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch

from molgap.constants import RESULTS_DIR
from molgap.router import paired_bootstrap_mean
from molgap.archive.phase8_r04_static_dual2d.diagnostics import router_diagnostics
from molgap.archive.phase8_r04_static_dual2d.dual2d_training import predict_dual2d, train_dual2d
from molgap.archive.phase8_r04_static_dual2d.evaluation import apply_static_weights, fit_static_weights, metrics


OUT_DIR = RESULTS_DIR / "phase8" / "archive" / "archive-r03-three-expert-moe"


def main() -> None:
    table = pd.read_parquet(OUT_DIR / "pilot_30k.parquet")
    with np.load(OUT_DIR / "pilot_expert_features.npz") as raw:
        source_idx = raw["source_idx"]
        local = raw["local_embedding"].copy()
        global_value = raw["global_embedding"].copy()
        expert_predictions = np.stack([
            raw["local_prediction"], raw["global_prediction"]
        ], axis=-1).astype(np.float32)
    table = table.set_index("source_idx").loc[source_idx].reset_index()
    targets = table[["homo", "lumo", "gap"]].to_numpy(dtype=np.float32)
    split_indices = {
        split: np.flatnonzero(table.split.eq(split).to_numpy())
        for split in ("train", "validation", "internal_test")
    }
    train = split_indices["train"]
    for embedding in (local, global_value):
        mean = embedding[train].mean(axis=0)
        std = embedding[train].std(axis=0).clip(1e-6)
        embedding -= mean
        embedding /= std
    validation = split_indices["validation"]
    test = split_indices["internal_test"]
    static_weights = fit_static_weights(
        targets[validation], expert_predictions[validation]
    )
    baselines = {
        "local": expert_predictions[test, :, 0],
        "global": expert_predictions[test, :, 1],
        "equal_average": expert_predictions[test].mean(axis=-1),
        "static_weights": apply_static_weights(
            expert_predictions[test], static_weights
        ),
    }
    result = {
        "experiment": "preliminary dual-2D candidate dual-2D Local GINE6 + Global GPS9 gate",
        "geometry_expert_used": False,
        "static_weights": static_weights.tolist(),
        "baselines": {name: metrics(targets[test], value) for name, value in baselines.items()},
        "learned": {},
    }
    best_single = min(("local", "global"), key=lambda name: result["baselines"][name]["gap"]["mae"])
    reference = baselines[best_single]
    for kind in ("concat_fusion", "soft_gate"):
        result["learned"][kind] = {}
        for seed in (42, 43, 44):
            model, training = train_dual2d(
                kind=kind,
                local_embedding=local,
                global_embedding=global_value,
                expert_predictions=expert_predictions,
                targets=targets,
                train_indices=train,
                validation_indices=validation,
                validation_sources=table.loc[validation, "sampling_source"].to_numpy(),
                seed=seed,
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
            delta = np.abs(prediction[:, 2] - targets[test, 2]) - np.abs(
                reference[:, 2] - targets[test, 2]
            )
            record = {
                "training": training,
                "metrics": metrics(targets[test], prediction),
                "gap_delta_vs_best_single": paired_bootstrap_mean(delta, seed=seed),
                "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            }
            if weights is not None:
                record["gate_diagnostics"] = router_diagnostics(torch.from_numpy(weights))
            result["learned"][kind][str(seed)] = record
            model_dir = OUT_DIR / "dual2d" / kind
            model_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), model_dir / f"seed{seed}.pt")
            print(kind, seed, record["metrics"]["gap"]["mae"], record["gap_delta_vs_best_single"]["delta"], flush=True)
        improvements = [
            -result["learned"][kind][str(seed)]["gap_delta_vs_best_single"]["delta"]
            for seed in (42, 43, 44)
        ]
        result["learned"][kind]["summary"] = {
            "mean_gap_improvement_vs_best_single_eV": float(np.mean(improvements)),
            "all_seeds_improve": bool(all(value > 0 for value in improvements)),
            "all_seeds_reach_0.001_eV": bool(all(value >= 0.001 for value in improvements)),
        }
    static_delta = np.abs(baselines["static_weights"][:, 2] - targets[test, 2]) - np.abs(
        reference[:, 2] - targets[test, 2]
    )
    static_comparison = paired_bootstrap_mean(static_delta, seed=41)
    static_gain = -static_comparison["delta"]
    fusion_summary = result["learned"]["concat_fusion"]["summary"]
    result["comparison"] = {
        "best_single": best_single,
        "static_gap_delta_vs_best_single": static_comparison,
        "static_gap_improvement_eV": static_gain,
    }
    result["decision"] = {
        "threshold_eV": 0.001,
        "dual2d_complementarity_pass": bool(
            (static_gain >= 0.001 and static_comparison["ci95"][1] < 0)
            or fusion_summary["all_seeds_reach_0.001_eV"]
        ),
        "train_dynamic_moe": False,
    }
    result["decision"]["train_dynamic_moe"] = result["decision"]["dual2d_complementarity_pass"]
    (OUT_DIR / "dual2d_metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result["decision"], indent=2), flush=True)


if __name__ == "__main__":
    main()
