"""Evaluate archive-r03 expert complementarity before training a Router."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from molgap.constants import RESULTS_DIR
from molgap.gps import GPSWrapper
from molgap.router import paired_bootstrap_mean
from molgap.schnet import SchNetWrapper
from molgap.dual2d_static_candidate.evaluation import (
    apply_static_weights, expert_complementarity, fit_static_weights, metrics,
)
from molgap.dual2d_static_candidate.local_gine import LocalGINEExpert
from molgap.dual2d_static_candidate.training import predict_expert


OUT_DIR = RESULTS_DIR / "phase8" / "archive" / "archive-r03-three-expert-moe"
DUAL2D_DIR = RESULTS_DIR / "phase8" / "dual2d_static_candidate"
EXPERTS = ("local", "global", "geometry")


def model_for(kind):
    if kind == "local":
        return LocalGINEExpert()
    if kind == "global":
        return GPSWrapper(
            hidden_channels=192, num_layers=9, num_heads=4,
            dropout=0.05, pooling="mean_max",
        )
    return SchNetWrapper(
        hidden_channels=192, num_filters=192, num_interactions=6,
        num_gaussians=50, cutoff=6.0, dropout=0.05,
    )


def main() -> None:
    table = pd.read_parquet(DUAL2D_DIR / "pilot_30k.parquet")
    graphs_2d = torch.load(DUAL2D_DIR / "pilot_30k_graphs_2d.pt", weights_only=False)
    graphs_3d = torch.load(OUT_DIR / "pilot_30k_graphs_3d.pt", weights_only=False)
    maps = {
        "2d": {int(graph.source_idx.item()): graph for graph in graphs_2d},
        "3d": {int(graph.source_idx.item()): graph for graph in graphs_3d},
    }
    common = set(maps["2d"]) & set(maps["3d"])
    table = table[table.source_idx.isin(common)].copy()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictions = {}
    for kind in EXPERTS:
        model = model_for(kind).to(device)
        model.load_state_dict(torch.load(
            (OUT_DIR if kind == "geometry" else DUAL2D_DIR)
            / f"expert_{kind}" / "seed42.pt",
            map_location=device,
            weights_only=True,
        ))
        graph_map = maps["3d" if kind == "geometry" else "2d"]
        ordered = [graph_map[int(index)] for index in table.source_idx]
        prediction, target, indices = predict_expert(
            kind, model, DataLoader(ordered, batch_size=256), device
        )
        if not np.array_equal(indices, table.source_idx.to_numpy()):
            raise ValueError(f"{kind} prediction order mismatch")
        predictions[kind] = prediction
        del model
        torch.cuda.empty_cache()
    stacked = np.stack([predictions[kind] for kind in EXPERTS], axis=-1)
    y = table[["homo", "lumo", "gap"]].to_numpy()
    validation = table.split.eq("validation").to_numpy()
    test = table.split.eq("internal_test").to_numpy()
    static_weights = fit_static_weights(y[validation], stacked[validation])
    equal = stacked[test].mean(axis=-1)
    static = apply_static_weights(stacked[test], static_weights)
    oracle = np.take_along_axis(
        stacked[test],
        np.abs(stacked[test] - y[test, :, None]).argmin(axis=-1)[..., None],
        axis=-1,
    ).squeeze(-1)
    model_predictions = {
        **{kind: predictions[kind][test] for kind in EXPERTS},
        "equal_average": equal,
        "static_weights": static,
        "oracle": oracle,
    }
    all_metrics = {name: metrics(y[test], value) for name, value in model_predictions.items()}
    best_single_name = min(EXPERTS, key=lambda name: all_metrics[name]["gap"]["mae"])
    static_delta = np.abs(static[:, 2] - y[test, 2]) - np.abs(
        model_predictions[best_single_name][:, 2] - y[test, 2]
    )
    oracle_delta = np.abs(oracle[:, 2] - y[test, 2]) - np.abs(static[:, 2] - y[test, 2])
    result = {
        "experiment": "archive-r03 30k from-scratch heterogeneous expert feasibility gate",
        "expert_order": list(EXPERTS),
        "validation_n": int(validation.sum()),
        "internal_test_n": int(test.sum()),
        "static_weights": static_weights.tolist(),
        "metrics": all_metrics,
        "complementarity": expert_complementarity(
            y[test], stacked[test], table.loc[test, "sampling_source"].to_numpy()
        ),
        "comparisons": {
            "best_single": best_single_name,
            "static_minus_best_single_gap": paired_bootstrap_mean(static_delta, seed=42),
            "oracle_minus_static_gap": paired_bootstrap_mean(oracle_delta, seed=43),
        },
    }
    oracle_headroom = -result["comparisons"]["oracle_minus_static_gap"]["delta"]
    expert_wins = result["complementarity"]["targets"]["gap"]["win_fraction"]
    result["decision"] = {
        "oracle_gap_headroom_eV": oracle_headroom,
        "at_least_two_experts_win_10pct": sum(value >= 0.10 for value in expert_wins) >= 2,
        "proceed_to_router_pilot": bool(
            oracle_headroom >= 0.005 and sum(value >= 0.10 for value in expert_wins) >= 2
        ),
        "note": "This gate tests complementarity only; it cannot promote a production model.",
    }
    (OUT_DIR / "pilot_expert_complementarity.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    output = table.loc[test, ["source_idx", "cid", "sampling_source", "scaffold"]].copy()
    for target_index, target_name in enumerate(("homo", "lumo", "gap")):
        output[f"y_{target_name}"] = y[test, target_index]
        for name, value in model_predictions.items():
            output[f"{name}_{target_name}"] = value[:, target_index]
    output.to_parquet(OUT_DIR / "pilot_internal_predictions.parquet", index=False)
    print(json.dumps({
        "metrics": all_metrics,
        "comparisons": result["comparisons"],
        "decision": result["decision"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
