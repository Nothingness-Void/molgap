"""
Evaluate a trained intermediate-layer fusion head on the Phase 8 common set.

Usage:
  .venv\\Scripts\\python.exe scripts/phase8/archive/legacy/head_probes/eval_layer_fusion_common.py --tag replacement30k
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

from molgap.constants import PARAMS_GPS_2D, PARAMS_SCHNET_300K, RESULTS_DIR
from molgap.fusion import FusionHead
from molgap.gps import GPSWrapper
from molgap.graphs import smiles_to_2d_pyg, smiles_to_pyg
from molgap.schnet import SchNetWrapper

PHASE8_DIR = RESULTS_DIR / "phase8"
TARGETS = ["homo", "lumo", "gap"]
DISPLAY_TARGETS = ["HOMO", "LUMO", "Gap"]


def _parse_layers(text: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in text.split(",") if x.strip())


def _metrics(y_true: np.ndarray, y_pred: np.ndarray):
    out = {}
    for i, name in enumerate(DISPLAY_TARGETS):
        out[name] = {
            "mae": float(mean_absolute_error(y_true[:, i], y_pred[:, i])),
            "r2": float(r2_score(y_true[:, i], y_pred[:, i])),
        }
    out["average"] = {
        "mae": float(np.mean([out[k]["mae"] for k in DISPLAY_TARGETS])),
        "r2": float(np.mean([out[k]["r2"] for k in DISPLAY_TARGETS])),
    }
    return out


def _build_graphs(df: pd.DataFrame):
    kept, g2d, g3d = [], [], []
    for i, smi in tqdm(list(enumerate(df["smiles"].tolist())), desc="build eval graphs"):
        graph_2d = smiles_to_2d_pyg(smi)
        graph_3d = smiles_to_pyg(smi)
        if graph_2d is None or graph_3d is None:
            continue
        kept.append(i)
        g2d.append(graph_2d)
        g3d.append(graph_3d)
    return df.iloc[kept].reset_index(drop=True), g2d, g3d


@torch.no_grad()
def _predict(df, g2d, g3d, args, device):
    gps_layers = _parse_layers(args.gps_layers)
    schnet_layers = _parse_layers(args.schnet_layers)

    gps = GPSWrapper(**PARAMS_GPS_2D).to(device)
    gps.load_state_dict(torch.load(PHASE8_DIR / f"gps_{args.tag}.pt", weights_only=True, map_location=device))
    gps.eval()

    schnet = SchNetWrapper(**PARAMS_SCHNET_300K, use_charges=True).to(device)
    schnet.load_state_dict(torch.load(PHASE8_DIR / f"schnet_{args.tag}.pt", weights_only=True, map_location=device))
    schnet.eval()

    emb2 = []
    for batch in GeometricDataLoader(g2d, batch_size=args.bs_2d, shuffle=False):
        batch = batch.to(device)
        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
            emb = gps.encode_layers(batch.x, batch.edge_index, batch.edge_attr, batch.batch, layers=gps_layers)
        emb2.append(emb.float().cpu())
    emb2 = torch.cat(emb2)

    emb3 = []
    for batch in GeometricDataLoader(g3d, batch_size=args.bs_3d, shuffle=False):
        batch = batch.to(device)
        charges = batch.charges if hasattr(batch, "charges") else None
        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
            emb = schnet.encode_layers(batch.z, batch.pos, batch.batch, charges=charges, layers=schnet_layers)
        emb3.append(emb.float().cpu())
    emb3 = torch.cat(emb3)

    fusion = FusionHead("gate", hidden=192, dim_2d=emb2.shape[1], dim_3d=emb3.shape[1]).to(device)
    fusion.load_state_dict(torch.load(args.model, weights_only=True, map_location=device))
    fusion.eval()

    pred = []
    for b2, b3 in TorchDataLoader(TensorDataset(emb2, emb3), batch_size=args.bs_fusion, shuffle=False):
        pred.append(fusion(b2.to(device), b3.to(device)).float().cpu())
    return torch.cat(pred).numpy()


def main():
    parser = argparse.ArgumentParser(description="Evaluate layer fusion on common eval set")
    parser.add_argument("--tag", default="replacement30k")
    pilot_archive = PHASE8_DIR / "archive" / "legacy" / "pilots_30k"
    head_archive = PHASE8_DIR / "archive" / "legacy" / "head_posthoc"
    parser.add_argument("--common-csv", type=Path, default=pilot_archive / "common_eval_30k_predictions.csv")
    parser.add_argument("--model", type=Path, default=head_archive / "layer_fusion_replacement30k.pt")
    parser.add_argument("--gps-layers", default="2,4,-1")
    parser.add_argument("--schnet-layers", default="2,4,-1")
    parser.add_argument("--bs-2d", type=int, default=256)
    parser.add_argument("--bs-3d", type=int, default=128)
    parser.add_argument("--bs-fusion", type=int, default=2048)
    parser.add_argument("--out", type=Path, default=head_archive / "layer_fusion_common_eval_metrics.json")
    parser.add_argument("--predictions", type=Path, default=head_archive / "layer_fusion_common_eval_predictions.csv")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | tag={args.tag}", flush=True)
    df = pd.read_csv(args.common_csv)
    df, g2d, g3d = _build_graphs(df)
    pred = _predict(df, g2d, g3d, args, device)

    y = df[TARGETS].to_numpy(dtype=np.float32)
    result = {
        "tag": args.tag,
        "n_eval": int(len(df)),
        "eval_set_counts": {k: int(v) for k, v in df["eval_set"].value_counts().items()},
        "all": _metrics(y, pred),
        "by_eval_set": {},
    }
    for eval_set in sorted(df["eval_set"].unique()):
        mask = df["eval_set"].to_numpy() == eval_set
        result["by_eval_set"][eval_set] = _metrics(y[mask], pred[mask])

    for i, target in enumerate(TARGETS):
        df[f"{args.tag}_layer_fusion_{target}"] = pred[:, i]
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    df.to_csv(args.predictions, index=False, encoding="utf-8")
    print(f"Metrics -> {args.out}", flush=True)
    print(f"Predictions -> {args.predictions}", flush=True)
    print(
        f"all avg={result['all']['average']['mae']:.5f} gap={result['all']['Gap']['mae']:.5f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
