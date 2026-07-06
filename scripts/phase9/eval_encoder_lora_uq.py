"""Evaluate an Encoder-LoRA ensemble with calibrated uncertainty.

This turns the v3 Encoder-LoRA accuracy probe into a UQ candidate:

1. load N separately trained LoRA checkpoints;
2. predict the same scaffold-split validation/test molecules;
3. calibrate ensemble std on the validation split;
4. report MAE/R2, ENCE, 1σ/2σ coverage on the held-out test split;
5. validate a simple embedding-distance OOD score against LoRA ensemble error.

The B3LYP base is not modified. Checkpoints are local artifacts under models/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.special import erfinv
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.neighbors import NearestNeighbors

from molgap.constants import MODELS_DIR, RESULTS_DIR, TARGET_COLS
from molgap.inference import load_hybrid

from train_encoder_lora_delta import (
    freeze,
    inject_lora,
    load_or_build_graphs,
    scaffold_split,
    evaluate,
)

PHASE9 = RESULTS_DIR / "phase9"
OUT_DIR = RESULTS_DIR / "phase10_lora_v3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=PHASE9 / "delta_oe62_v3.csv")
    parser.add_argument("--graph-cache", type=Path, default=PHASE9 / "delta_oe62_v3_graphs.pt")
    parser.add_argument("--embedding-npz", type=Path, default=PHASE9 / "delta_oe62_v3_embeddings.npz")
    parser.add_argument("--hybrid-key", default="phase8_expansion_hybrid")
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def resolve_device(text: str | None) -> torch.device:
    if text:
        return torch.device(text)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_lora_member(path: Path, hybrid_key: str, device: torch.device):
    ckpt = torch.load(path, weights_only=False, map_location=device)
    targets = tuple(ckpt.get("targets", ("fusion", "gps", "schnet")))
    rank = int(ckpt.get("rank", 4))
    alpha = float(ckpt.get("alpha", 8.0))

    gps, schnet, fusion, _ = load_hybrid(device, key=hybrid_key)
    freeze(gps)
    freeze(schnet)
    freeze(fusion)
    if "gps" in targets:
        inject_lora(gps, rank, alpha, 0.0)
    if "schnet" in targets:
        inject_lora(schnet, rank, alpha, 0.0)
    if "fusion" in targets:
        inject_lora(fusion, rank, alpha, 0.0)
    state = ckpt["state"]
    gps.load_state_dict(state["gps"])
    schnet.load_state_dict(state["schnet"])
    fusion.load_state_dict(state["fusion"])
    gps.to(device)
    schnet.to(device)
    fusion.to(device)
    gps.eval()
    schnet.eval()
    fusion.eval()
    return gps, schnet, fusion


def ensemble_predict(members, g2d, g3d, y, indices, device, batch_size: int) -> np.ndarray:
    preds = []
    for i, (gps, schnet, fusion) in enumerate(members, start=1):
        print(f"Predicting member {i}/{len(members)}", flush=True)
        preds.append(evaluate(gps, schnet, fusion, g2d, g3d, y, indices, device, batch_size))
    return np.stack(preds, axis=0)


def ence(errors: np.ndarray, sigmas: np.ndarray, n_bins: int = 10) -> float:
    order = np.argsort(sigmas)
    sigmas = sigmas[order]
    abs_err = np.abs(errors)[order]
    total, weight = 0.0, 0
    for b in np.array_split(np.arange(len(sigmas)), n_bins):
        if len(b) == 0:
            continue
        rmv = np.sqrt(np.mean(sigmas[b] ** 2))
        rmse = np.sqrt(np.mean(abs_err[b] ** 2))
        if rmv > 1e-9:
            total += len(b) * abs(rmv - rmse) / rmv
            weight += len(b)
    return float(total / max(weight, 1))


def coverage(errors: np.ndarray, sigmas: np.ndarray, p: float) -> float:
    z = np.sqrt(2.0) * erfinv(p)
    return float(np.mean(np.abs(errors) <= z * sigmas))


def reliability_curve(errors: np.ndarray, sigmas: np.ndarray, target: str, path: Path) -> None:
    levels = np.linspace(0.05, 0.95, 19)
    obs = [coverage(errors, sigmas, p) for p in levels]
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.plot(levels, obs, "o-", ms=4)
    ax.set_xlabel("expected coverage")
    ax.set_ylabel("observed coverage")
    ax.set_title(f"LoRA reliability - {target.upper()}")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def metrics_block(y_true: np.ndarray, mean: np.ndarray, raw_std: np.ndarray, scale: np.ndarray, out_dir: Path) -> dict:
    result = {}
    for i, target in enumerate(TARGET_COLS):
        err = y_true[:, i] - mean[:, i]
        sig_raw = np.clip(raw_std[:, i], 1e-6, None)
        sig = sig_raw * scale[i]
        reliability_curve(err, sig, target, out_dir / f"reliability_{target}.png")
        result[target] = {
            "mae": float(mean_absolute_error(y_true[:, i], mean[:, i])),
            "r2": float(r2_score(y_true[:, i], mean[:, i])),
            "raw_sigma_mean": float(sig_raw.mean()),
            "scale": float(scale[i]),
            "sigma_mean": float(sig.mean()),
            "ence_pre": ence(err, sig_raw),
            "ence_post": ence(err, sig),
            "coverage_1sigma": coverage(err, sig, 0.6827),
            "coverage_2sigma": coverage(err, sig, 0.9545),
        }
    result["average"] = {
        "mae": float(np.mean([result[t]["mae"] for t in TARGET_COLS])),
        "r2": float(np.mean([result[t]["r2"] for t in TARGET_COLS])),
    }
    return result


def ood_features(npz_path: Path) -> np.ndarray:
    npz = np.load(npz_path, allow_pickle=True)
    return np.hstack([npz["emb_2d"], npz["emb_3d"]]).astype(np.float32)


def ood_analysis(X: np.ndarray, train: np.ndarray, test: np.ndarray, abs_gap_err: np.ndarray) -> dict:
    mu = X[train].mean(axis=0)
    sd = X[train].std(axis=0) + 1e-8
    ref = (X[train] - mu) / sd
    query = (X[test] - mu) / sd
    nn = NearestNeighbors(n_neighbors=5, metric="euclidean").fit(ref)
    dist, _ = nn.kneighbors(query)
    d = dist.mean(axis=1)
    self_nn = NearestNeighbors(n_neighbors=6, metric="euclidean").fit(ref)
    self_dist, _ = self_nn.kneighbors(ref)
    threshold = float(np.percentile(self_dist[:, 1:].mean(axis=1), 95))
    order = np.argsort(d)
    bins = np.array_split(order, 10)
    binned = [float(abs_gap_err[b].mean()) for b in bins]
    return {
        "metric": "euclidean",
        "k": 5,
        "threshold_p95": threshold,
        "ood_fraction": float((d > threshold).mean()),
        "spearman_dist_abs_gap_error": float(spearmanr(d, abs_gap_err).correlation),
        "binned_gap_mae_deciles": binned,
        "distance": d.astype(float).tolist(),
        "mu": mu.astype(float).tolist(),
        "sd": sd.astype(float).tolist(),
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    print(f"Device: {device}", flush=True)
    print(f"Members: {len(args.checkpoint)}", flush=True)

    df, g2d, g3d = load_or_build_graphs(args.csv, args.graph_cache)
    y = torch.tensor(df[[f"gw_{t}" for t in TARGET_COLS]].values, dtype=torch.float32)
    gw = y.numpy()
    train, val, test, n_scaffolds = scaffold_split(df["smiles"].tolist(), 42)

    members = [load_lora_member(path, args.hybrid_key, device) for path in args.checkpoint]
    pred_val = ensemble_predict(members, g2d, g3d, y, val, device, args.batch_size)
    pred_test = ensemble_predict(members, g2d, g3d, y, test, device, args.batch_size)

    mean_val = pred_val.mean(axis=0)
    std_val = np.clip(pred_val.std(axis=0), 1e-6, None)
    err_val = gw[val] - mean_val
    scale = np.sqrt(np.mean((err_val / std_val) ** 2, axis=0))

    mean_test = pred_test.mean(axis=0)
    std_test = np.clip(pred_test.std(axis=0), 1e-6, None)
    metrics = metrics_block(gw[test], mean_test, std_test, scale, args.out_dir)

    abs_gap_err = np.abs(gw[test, TARGET_COLS.index("gap")] - mean_test[:, TARGET_COLS.index("gap")])
    ood = ood_analysis(ood_features(args.embedding_npz), train, test, abs_gap_err)

    result = {
        "hybrid_key": args.hybrid_key,
        "csv": str(args.csv),
        "graph_cache": str(args.graph_cache),
        "embedding_npz": str(args.embedding_npz),
        "checkpoints": [str(p) for p in args.checkpoint],
        "n_members": len(args.checkpoint),
        "n": int(len(df)),
        "n_scaffolds": int(n_scaffolds),
        "n_train": int(len(train)),
        "n_val_calib": int(len(val)),
        "n_test": int(len(test)),
        "scale": {t: float(scale[i]) for i, t in enumerate(TARGET_COLS)},
        "metrics": metrics,
        "ood": {k: v for k, v in ood.items() if k not in {"distance", "mu", "sd"}},
    }
    (args.out_dir / "lora_uq_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    pred_df = df.iloc[test].reset_index(drop=True).copy()
    for i, target in enumerate(TARGET_COLS):
        pred_df[f"gw_pred_lora_ensemble_{target}"] = mean_test[:, i]
        pred_df[f"gw_sigma_lora_ensemble_{target}"] = std_test[:, i] * scale[i]
    pred_df["ood_distance"] = ood["distance"]
    pred_df.to_csv(args.out_dir / "lora_uq_predictions.csv", index=False, encoding="utf-8")

    np.savez(
        args.out_dir / "ood_reference.npz",
        ref_std=((ood_features(args.embedding_npz)[train] - np.asarray(ood["mu"])) / np.asarray(ood["sd"])).astype(np.float32),
        mu=np.asarray(ood["mu"], dtype=np.float32),
        sd=np.asarray(ood["sd"], dtype=np.float32),
        threshold=np.asarray([ood["threshold_p95"]], dtype=np.float32),
        k=np.asarray([5], dtype=np.int64),
    )

    print("\nLoRA ensemble UQ")
    for target in TARGET_COLS:
        row = metrics[target]
        print(
            f"  {target}: MAE={row['mae']:.4f} R2={row['r2']:.4f} "
            f"sigma={row['sigma_mean']:.4f} cov1={row['coverage_1sigma']:.3f} "
            f"cov2={row['coverage_2sigma']:.3f} ENCE={row['ence_post']:.3f}",
            flush=True,
        )
    print(f"  avg: MAE={metrics['average']['mae']:.4f} R2={metrics['average']['r2']:.4f}")
    print(
        f"  OOD gap rho={result['ood']['spearman_dist_abs_gap_error']:.3f} "
        f"near->far={result['ood']['binned_gap_mae_deciles'][0]:.3f}->"
        f"{result['ood']['binned_gap_mae_deciles'][-1]:.3f}",
        flush=True,
    )
    print(f"Saved {args.out_dir}")


if __name__ == "__main__":
    main()
