"""Evaluate ETKDG conformer-ensemble inference for the v3 B3LYP Hybrid.

This is a B3LYP-level inference probe only:

- no LoRA / GW Delta;
- no PM6 or external geometry;
- same v3 GPS + SchNet + FusionHead checkpoints;
- multiple ETKDG+MMFF conformers per molecule, averaged after fusion.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, r2_score
from torch_geometric.loader import DataLoader

from molgap.constants import RESULTS_DIR, TARGET_COLS
from molgap.graphs import smiles_to_2d_pyg, smiles_to_pyg_ensemble
from molgap.inference import load_hybrid

PHASE8 = RESULTS_DIR / "phase8"
BASE_PREFIX = "expansion500k_full_hybrid"
TARGETS_DISPLAY = ("HOMO", "LUMO", "Gap")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--common-csv", type=Path, default=PHASE8 / "full_expansion500k_common_eval_predictions.csv")
    archive = PHASE8 / "archive" / "legacy" / "conformer_ensemble"
    parser.add_argument("--out-json", type=Path, default=archive / "v3_conformer_ensemble_metrics.json")
    parser.add_argument("--out-md", type=Path, default=archive / "v3_conformer_ensemble_decision.md")
    parser.add_argument("--out-predictions", type=Path, default=archive / "v3_conformer_ensemble_predictions.csv")
    parser.add_argument("--hybrid-key", default="phase8_expansion_hybrid")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bs-2d", type=int, default=256)
    parser.add_argument("--bs-3d", type=int, default=128)
    parser.add_argument("--bs-fusion", type=int, default=2048)
    return parser.parse_args()


def metrics(y_true: np.ndarray, pred: np.ndarray) -> dict:
    out = {}
    for i, name in enumerate(TARGETS_DISPLAY):
        out[name] = {
            "mae": float(mean_absolute_error(y_true[:, i], pred[:, i])),
            "r2": float(r2_score(y_true[:, i], pred[:, i])),
            "bias": float(np.mean(pred[:, i] - y_true[:, i])),
        }
    out["average"] = {
        "mae": float(np.mean([out[t]["mae"] for t in TARGETS_DISPLAY])),
        "r2": float(np.mean([out[t]["r2"] for t in TARGETS_DISPLAY])),
    }
    return out


def metric_blocks(df: pd.DataFrame, pred: np.ndarray) -> dict:
    y = df[list(TARGET_COLS)].to_numpy(dtype=np.float32)
    blocks = {"all": metrics(y, pred)}
    for scope, sub in df.groupby("eval_set"):
        idx = sub.index.to_numpy()
        blocks[str(scope)] = metrics(y[idx], pred[idx])
    return blocks


def build_graphs(df: pd.DataFrame, k: int, seed: int):
    kept_rows = []
    g2d_list = []
    g3d_list = []
    conf_owner = []
    n_confs = []
    for row_i, smi in enumerate(df["smiles"].astype(str).tolist()):
        g2d = smiles_to_2d_pyg(smi)
        if g2d is None:
            continue
        confs = smiles_to_pyg_ensemble(smi, k=k, random_seed=seed + row_i * 1000)
        if not confs:
            continue
        kept_rows.append(row_i)
        g2d_list.append(g2d)
        n_confs.append(len(confs))
        local_idx = len(kept_rows) - 1
        for conf in confs:
            g3d_list.append(conf)
            conf_owner.append(local_idx)
        if len(kept_rows) % 100 == 0:
            print(f"  built {len(kept_rows)}/{len(df)} molecules, {len(g3d_list)} conformers", flush=True)
    return kept_rows, g2d_list, g3d_list, np.asarray(conf_owner, dtype=np.int64), np.asarray(n_confs, dtype=np.int64)


@torch.no_grad()
def encode_2d(gps, graphs, batch_size: int, device: torch.device) -> torch.Tensor:
    rows = []
    for batch in DataLoader(graphs, batch_size=batch_size, shuffle=False):
        batch = batch.to(device)
        rows.append(gps.encode(batch.x, batch.edge_index, batch.edge_attr, batch.batch).float().cpu())
    return torch.cat(rows, dim=0)


@torch.no_grad()
def encode_3d(schnet, graphs, batch_size: int, device: torch.device) -> torch.Tensor:
    rows = []
    for batch in DataLoader(graphs, batch_size=batch_size, shuffle=False):
        batch = batch.to(device)
        charges = batch.charges if hasattr(batch, "charges") else None
        rows.append(schnet.encode(batch.z, batch.pos, batch.batch, charges=charges).float().cpu())
    return torch.cat(rows, dim=0)


@torch.no_grad()
def predict_conformer_ensemble(gps, schnet, fusion, g2d, g3d, conf_owner, batch_size: int, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    e2d = encode_2d(gps, g2d, batch_size, device)
    e3d = encode_3d(schnet, g3d, min(batch_size, 128), device)
    owner = torch.tensor(conf_owner, dtype=torch.long)
    pred_conf = []
    for start in range(0, len(owner), batch_size):
        end = min(start + batch_size, len(owner))
        pred_conf.append(
            fusion(e2d[owner[start:end]].to(device), e3d[start:end].to(device)).float().cpu()
        )
    pred_conf_arr = torch.cat(pred_conf, dim=0).numpy()
    n_mol = len(g2d)
    mean = np.zeros((n_mol, len(TARGET_COLS)), dtype=np.float32)
    std = np.zeros_like(mean)
    for i in range(n_mol):
        p = pred_conf_arr[conf_owner == i]
        mean[i] = p.mean(axis=0)
        std[i] = p.std(axis=0)
    return mean, std


def write_decision(path: Path, result: dict, metrics_path: Path, predictions_path: Path) -> None:
    base = result["metrics"]["stored_single"]["all"]
    ens = result["metrics"]["conformer_ensemble"]["all"]
    delta_avg = ens["average"]["mae"] - base["average"]["mae"]
    delta_gap = ens["Gap"]["mae"] - base["Gap"]["mae"]
    verdict = "positive" if delta_avg < -0.001 and delta_gap <= 0 else "negative"
    lines = [
        "# Phase 8 v3 ETKDG Conformer Ensemble Probe",
        "",
        "Date: 2026-07-06",
        "",
        "## Setup",
        "",
        "- Base: `phase8_expansion_hybrid` B3LYP v3.",
        f"- Inference: average up to `{result['k']}` seeded ETKDG+MMFF conformers per molecule.",
        "- 2D graph is unchanged; only the SchNet 3D leg sees conformer variants.",
        "- Evaluation: Phase 8 common eval with the same B3LYP labels.",
        "",
        "## Common Eval MAE",
        "",
        "| model | HOMO | LUMO | Gap | avg |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, label in [("stored_single", "stored v3 single"), ("conformer_ensemble", "ETKDG ensemble")]:
        row = result["metrics"][name]["all"]
        lines.append(
            f"| {label} | {row['HOMO']['mae']:.4f} | {row['LUMO']['mae']:.4f} | "
            f"{row['Gap']['mae']:.4f} | {row['average']['mae']:.4f} |"
        )
    lines.extend([
        "",
        "## Decision",
        "",
        f"Probe verdict: **{verdict}**. Ensemble changes avg/GAP MAE by `{delta_avg:+.5f}/{delta_gap:+.5f}` eV.",
    ])
    if verdict == "negative":
        lines.append("Do not promote conformer-ensemble inference for the B3LYP baseline.")
    else:
        lines.append("Keep as an inference candidate, but benchmark speed before changing default prediction.")
    lines.extend([
        "",
        "Artifacts:",
        "",
        f"- `{metrics_path}`",
        f"- `{predictions_path}`",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    df = pd.read_csv(args.common_csv)
    kept, g2d, g3d, conf_owner, n_confs = build_graphs(df, args.k, args.seed)
    eval_df = df.iloc[kept].reset_index(drop=True)
    print(f"Valid molecules: {len(eval_df)}/{len(df)} | conformers: {len(g3d)}", flush=True)

    gps, schnet, fusion, _ = load_hybrid(device, key=args.hybrid_key)
    pred_mean, pred_std = predict_conformer_ensemble(
        gps, schnet, fusion, g2d, g3d, conf_owner, args.bs_fusion, device
    )
    stored = eval_df[[f"{BASE_PREFIX}_{t}" for t in TARGET_COLS]].to_numpy(dtype=np.float32)

    result = {
        "kind": "v3_etkdg_conformer_ensemble_probe",
        "hybrid_key": args.hybrid_key,
        "k": args.k,
        "seed": args.seed,
        "n_input": int(len(df)),
        "n_valid": int(len(eval_df)),
        "n_conformers": int(len(g3d)),
        "n_confs_summary": {
            "min": int(n_confs.min()) if len(n_confs) else 0,
            "mean": float(n_confs.mean()) if len(n_confs) else 0.0,
            "max": int(n_confs.max()) if len(n_confs) else 0,
        },
        "metrics": {
            "stored_single": metric_blocks(eval_df, stored),
            "conformer_ensemble": metric_blocks(eval_df, pred_mean),
        },
        "conformer_std_mean": {
            target: float(pred_std[:, i].mean())
            for i, target in enumerate(TARGET_COLS)
        },
    }
    args.out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    pred_df = eval_df.copy()
    pred_df["n_conformers"] = n_confs
    for i, target in enumerate(TARGET_COLS):
        pred_df[f"conformer_ensemble_{target}"] = pred_mean[:, i]
        pred_df[f"conformer_ensemble_std_{target}"] = pred_std[:, i]
    pred_df.to_csv(args.out_predictions, index=False, encoding="utf-8")
    write_decision(args.out_md, result, args.out_json, args.out_predictions)

    base = result["metrics"]["stored_single"]["all"]
    ens = result["metrics"]["conformer_ensemble"]["all"]
    print(
        f"Stored avg/Gap={base['average']['mae']:.5f}/{base['Gap']['mae']:.5f} | "
        f"Ensemble avg/Gap={ens['average']['mae']:.5f}/{ens['Gap']['mae']:.5f} | "
        f"delta={ens['average']['mae'] - base['average']['mae']:+.5f}/"
        f"{ens['Gap']['mae'] - base['Gap']['mae']:+.5f}",
        flush=True,
    )
    print(f"Metrics -> {args.out_json}", flush=True)
    print(f"Decision -> {args.out_md}", flush=True)


if __name__ == "__main__":
    main()
