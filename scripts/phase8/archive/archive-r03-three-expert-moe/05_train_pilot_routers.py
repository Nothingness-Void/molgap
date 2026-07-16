"""Run the ordered frozen-expert archive-r03 Router ablations on three seeds."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch

from molgap.constants import RESULTS_DIR
from molgap.router import paired_bootstrap_mean
from molgap.dual2d_static_candidate.diagnostics import router_diagnostics
from molgap.dual2d_static_candidate.evaluation import apply_static_weights, metrics
from molgap.archive.phase8_r03_three_expert.router_training import (
    predict_frozen_router, train_frozen_router,
)


OUT_DIR = RESULTS_DIR / "phase8" / "archive" / "archive-r03-three-expert-moe"
VARIANTS = {
    "shared_no_desc_m0": {"shared_targets": True, "descriptors": False, "residual": False},
    "target_no_desc_m0": {"shared_targets": False, "descriptors": False, "residual": False},
    "target_desc_m0": {"shared_targets": False, "descriptors": True, "residual": False},
    "target_desc_m1": {"shared_targets": False, "descriptors": True, "residual": True},
}


def main() -> None:
    table = pd.read_parquet(OUT_DIR / "pilot_30k.parquet")
    with np.load(OUT_DIR / "pilot_expert_features.npz") as raw:
        source_idx = raw["source_idx"]
        embeddings = [raw[f"{kind}_embedding"].copy() for kind in ("local", "global", "geometry")]
        expert_predictions = np.stack([
            raw[f"{kind}_prediction"] for kind in ("local", "global", "geometry")
        ], axis=-1).astype(np.float32)
        descriptors = raw["descriptors"].copy()
    table = table.set_index("source_idx").loc[source_idx].reset_index()
    train = table.split.eq("train").to_numpy()
    means = np.nanmean(descriptors[train], axis=0)
    stds = np.nanstd(descriptors[train], axis=0)
    descriptors = np.where(np.isfinite(descriptors), descriptors, means)
    descriptors = ((descriptors - means) / np.maximum(stds, 1e-6)).astype(np.float32)
    targets = table[["homo", "lumo", "gap"]].to_numpy(dtype=np.float32)
    split_indices = {
        split: np.flatnonzero(table.split.eq(split).to_numpy())
        for split in ("train", "validation", "internal_test")
    }
    static_weights = np.asarray(json.load(open(
        OUT_DIR / "pilot_expert_complementarity.json"
    ))["static_weights"])
    test_idx = split_indices["internal_test"]
    static = apply_static_weights(expert_predictions[test_idx], static_weights)
    result = {
        "experiment": "archive-r03 frozen-expert Router pilot",
        "selection": "validation robust score; internal test opened after each fitted seed",
        "variants": {},
    }
    for variant_name, config in VARIANTS.items():
        result["variants"][variant_name] = {}
        for seed in (42, 43, 44):
            used_descriptors = descriptors if config["descriptors"] else descriptors[:, :0]
            arrays = {
                "embeddings": embeddings,
                "expert_predictions": expert_predictions,
                "descriptors": used_descriptors,
                "targets": targets,
            }
            model, training = train_frozen_router(
                arrays=arrays,
                train_indices=split_indices["train"],
                validation_indices=split_indices["validation"],
                validation_sources=table.loc[split_indices["validation"], "sampling_source"].to_numpy(),
                n_descriptors=used_descriptors.shape[1],
                shared_targets=config["shared_targets"],
                use_residual=config["residual"],
                seed=seed,
            )
            prediction, weights, residual = predict_frozen_router(model, arrays, test_idx)
            delta = np.abs(prediction[:, 2] - targets[test_idx, 2]) - np.abs(
                static[:, 2] - targets[test_idx, 2]
            )
            record = {
                "training": training,
                "test_metrics": metrics(targets[test_idx], prediction),
                "gap_delta_vs_static": paired_bootstrap_mean(delta, seed=seed),
                "router": router_diagnostics(torch.from_numpy(weights)),
                "residual_mean_abs_eV": float(np.abs(residual).mean()),
            }
            result["variants"][variant_name][str(seed)] = record
            model_dir = OUT_DIR / "moe" / variant_name
            model_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), model_dir / f"seed{seed}.pt")
            print(
                variant_name, seed,
                record["test_metrics"]["gap"]["mae"],
                record["gap_delta_vs_static"]["delta"],
                flush=True,
            )
    for variant_name, seeds in result["variants"].items():
        improvements = [-row["gap_delta_vs_static"]["delta"] for row in seeds.values()]
        result["variants"][variant_name]["summary"] = {
            "mean_gap_improvement_vs_static_eV": float(np.mean(improvements)),
            "all_seeds_improve": bool(all(value > 0 for value in improvements)),
        }
    m0 = result["variants"]["target_desc_m0"]["summary"]
    m1 = result["variants"]["target_desc_m1"]["summary"]
    best_gain = max(
        m0["mean_gap_improvement_vs_static_eV"],
        m1["mean_gap_improvement_vs_static_eV"],
    )
    result["decision"] = {
        "best_router_gain_vs_static_eV": best_gain,
        "router_effective_threshold_eV": 0.001,
        "stop_threshold_eV": 0.0005,
        "proceed_to_joint_finetune": bool(
            best_gain >= 0.001 and (m0["all_seeds_improve"] or m1["all_seeds_improve"])
        ),
    }
    (OUT_DIR / "pilot_router_metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result["decision"], indent=2), flush=True)


if __name__ == "__main__":
    main()
