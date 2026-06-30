"""
Common-evaluate the full Phase 7 baseline against the full Phase 8 replacement300k model.

This uses the same common eval rows created for the 30k decision test, but loads
the full 300k checkpoints:

  P7 baseline: models/gps_2d_300k.pt + models/gnn_schnet_3d_300k.pt
               + models/hybrid_fusion_optuna.pt
  P8 full:     models/phase8_gps_replacement_300k.pt
               + models/phase8_schnet_replacement_300k.pt
               + models/phase8_hybrid_fusion_replacement_300k.pt

Usage:
  .venv\\Scripts\\python.exe scripts/phase8/eval_full_replacement_common.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, r2_score
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import TensorDataset
from torch_geometric.loader import DataLoader as GeometricDataLoader
from tqdm import tqdm

from molgap.constants import MODELS_DIR, PARAMS_GPS_2D, PARAMS_SCHNET_300K, RESULTS_DIR
from molgap.fusion import FusionHead
from molgap.gps import GPSWrapper
from molgap.graphs import smiles_to_2d_pyg, smiles_to_pyg
from molgap.schnet import SchNetWrapper
from molgap.utils import ensure_dirs

PHASE8_DIR = RESULTS_DIR / "phase8"
TARGETS = ["homo", "lumo", "gap"]
DISPLAY_TARGETS = ["HOMO", "LUMO", "Gap"]


def _build_graphs(df: pd.DataFrame):
    kept, graphs_2d, graphs_3d = [], [], []
    for i, smi in tqdm(list(enumerate(df["smiles"].tolist())), desc="build eval graphs"):
        graph_2d = smiles_to_2d_pyg(smi)
        graph_3d = smiles_to_pyg(smi)
        if graph_2d is None or graph_3d is None:
            continue
        kept.append(i)
        graphs_2d.append(graph_2d)
        graphs_3d.append(graph_3d)
    return df.iloc[kept].reset_index(drop=True), graphs_2d, graphs_3d


def _load_trio(gps_path: Path, schnet_path: Path, fusion_path: Path, device):
    gps = GPSWrapper(**PARAMS_GPS_2D).to(device)
    gps.load_state_dict(torch.load(gps_path, weights_only=True, map_location=device))
    gps.eval()

    schnet = SchNetWrapper(**PARAMS_SCHNET_300K, use_charges=True).to(device)
    schnet.load_state_dict(torch.load(schnet_path, weights_only=True, map_location=device))
    schnet.eval()

    fusion = FusionHead("gate", 192, 0.0).to(device)
    fusion.load_state_dict(torch.load(fusion_path, weights_only=True, map_location=device))
    fusion.eval()
    return gps, schnet, fusion


@torch.no_grad()
def _predict(gps, schnet, fusion, graphs_2d, graphs_3d, args, device):
    emb2, pred2 = [], []
    for batch in GeometricDataLoader(graphs_2d, batch_size=args.bs_2d, shuffle=False):
        batch = batch.to(device)
        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
            emb = gps.encode(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            pred = gps(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        emb2.append(emb.float().cpu())
        pred2.append(pred.float().cpu())
    emb2 = torch.cat(emb2)
    pred2 = torch.cat(pred2).numpy()

    emb3, pred3 = [], []
    for batch in GeometricDataLoader(graphs_3d, batch_size=args.bs_3d, shuffle=False):
        batch = batch.to(device)
        charges = batch.charges if hasattr(batch, "charges") else None
        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
            emb = schnet.encode(batch.z, batch.pos, batch.batch, charges=charges)
            pred = schnet(batch.z, batch.pos, batch.batch, charges=charges)
        emb3.append(emb.float().cpu())
        pred3.append(pred.float().cpu())
    emb3 = torch.cat(emb3)
    pred3 = torch.cat(pred3).numpy()

    hybrid = []
    for batch_2d, batch_3d in TorchDataLoader(
        TensorDataset(emb2, emb3), batch_size=args.bs_fusion, shuffle=False
    ):
        hybrid.append(fusion(batch_2d.to(device), batch_3d.to(device)).float().cpu())
    return {
        "gps_2d": pred2,
        "schnet_3d": pred3,
        "hybrid": torch.cat(hybrid).numpy(),
    }


def _metrics(y_true: np.ndarray, y_pred: np.ndarray):
    out = {}
    for i, name in enumerate(DISPLAY_TARGETS):
        out[name] = {
            "mae": float(mean_absolute_error(y_true[:, i], y_pred[:, i])),
            "r2": float(r2_score(y_true[:, i], y_pred[:, i])),
        }
    out["average"] = {
        "mae": float(np.mean([out[name]["mae"] for name in DISPLAY_TARGETS])),
        "r2": float(np.mean([out[name]["r2"] for name in DISPLAY_TARGETS])),
    }
    return out


def _metric_blocks(eval_df: pd.DataFrame, pred: np.ndarray):
    y_true = eval_df[TARGETS].to_numpy(dtype=np.float32)
    blocks = {"all": _metrics(y_true, pred)}
    for eval_set in sorted(eval_df["eval_set"].unique()):
        mask = eval_df["eval_set"].to_numpy() == eval_set
        blocks[eval_set] = _metrics(y_true[mask], pred[mask])
    return blocks


def main():
    parser = argparse.ArgumentParser(description="Common eval for full replacement300k")
    parser.add_argument("--common-csv", type=Path, default=PHASE8_DIR / "common_eval_30k_predictions.csv")
    parser.add_argument("--out", type=Path, default=PHASE8_DIR / "full_replacement_common_eval_metrics.json")
    parser.add_argument("--predictions", type=Path, default=PHASE8_DIR / "full_replacement_common_eval_predictions.csv")
    parser.add_argument("--bs-2d", type=int, default=256)
    parser.add_argument("--bs-3d", type=int, default=128)
    parser.add_argument("--bs-fusion", type=int, default=2048)
    args = parser.parse_args()

    ensure_dirs(PHASE8_DIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    eval_df = pd.read_csv(args.common_csv)
    eval_df, graphs_2d, graphs_3d = _build_graphs(eval_df)
    print(
        f"Common eval valid N={len(eval_df)} "
        f"sets={eval_df['eval_set'].value_counts().to_dict()}",
        flush=True,
    )

    model_specs = {
        "phase7_full": {
            "gps": MODELS_DIR / "gps_2d_300k.pt",
            "schnet": MODELS_DIR / "gnn_schnet_3d_300k.pt",
            "fusion": MODELS_DIR / "hybrid_fusion_optuna.pt",
        },
        "replacement300k_full": {
            "gps": MODELS_DIR / "phase8_gps_replacement_300k.pt",
            "schnet": MODELS_DIR / "phase8_schnet_replacement_300k.pt",
            "fusion": MODELS_DIR / "phase8_hybrid_fusion_replacement_300k.pt",
        },
    }
    expansion500k = {
        "gps": MODELS_DIR / "phase8_gps_expansion_500k.pt",
        "schnet": MODELS_DIR / "phase8_schnet_expansion_500k.pt",
        "fusion": MODELS_DIR / "phase8_hybrid_fusion_expansion_500k.pt",
    }
    if all(path.exists() for path in expansion500k.values()):
        model_specs["expansion500k_full"] = expansion500k
    tail_probe = {
        "gps": MODELS_DIR / "phase8_gps_expansion_500k.pt",
        "schnet": MODELS_DIR / "phase8_schnet_expansion_500k.pt",
        "fusion": MODELS_DIR / "phase8_hybrid_fusion_tail_probe_30k.pt",
    }
    if all(path.exists() for path in tail_probe.values()):
        model_specs["tail_probe30k_fusion"] = tail_probe

    metrics = {
        "n_eval": int(len(eval_df)),
        "eval_set_counts": {k: int(v) for k, v in eval_df["eval_set"].value_counts().items()},
        "models": {},
    }
    pred_df = eval_df.copy()
    for name, paths in model_specs.items():
        print(f"Predicting {name}", flush=True)
        trio = _load_trio(paths["gps"], paths["schnet"], paths["fusion"], device)
        preds = _predict(*trio, graphs_2d, graphs_3d, args, device)
        metrics["models"][name] = {}
        for pred_name, pred in preds.items():
            metrics["models"][name][pred_name] = _metric_blocks(eval_df, pred)
            for i, target in enumerate(TARGETS):
                pred_df[f"{name}_{pred_name}_{target}"] = pred[:, i]

    old = metrics["models"]["phase7_full"]["hybrid"]
    for model_name, model_metrics in metrics["models"].items():
        if model_name == "phase7_full":
            continue
        new = model_metrics["hybrid"]
        deltas = {}
        for block in old:
            deltas[block] = {
                "average_mae_delta": float(new[block]["average"]["mae"] - old[block]["average"]["mae"]),
                "gap_mae_delta": float(new[block]["Gap"]["mae"] - old[block]["Gap"]["mae"]),
            }
        metrics[f"{model_name}_minus_phase7_hybrid"] = deltas

    args.out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pred_df.to_csv(args.predictions, index=False, encoding="utf-8")
    print(f"Metrics -> {args.out}", flush=True)
    print(f"Predictions -> {args.predictions}", flush=True)
    print(
        "Hybrid all avg: "
        + " ".join(
            f"{name}={metrics['models'][name]['hybrid']['all']['average']['mae']:.5f}"
            for name in metrics["models"]
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
