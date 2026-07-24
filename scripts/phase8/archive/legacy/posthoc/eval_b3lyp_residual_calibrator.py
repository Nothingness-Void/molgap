"""Probe a lightweight B3LYP-level residual calibrator for the v3 baseline.

This does not touch the GPS/SchNet/Fusion checkpoints. It learns a post-hoc
residual correction from v3 predictions plus lightweight RDKit context features
on the expansion500k training split, then evaluates whether that correction
generalizes to the held-out internal test and the Phase 8 common-eval slices.
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

from molgap.constants import MODELS_DIR, RAW_DIR, RESULTS_DIR, SEED
from molgap.fusion import FusionHead
from molgap.inference import load_hybrid
from molgap.utils import FUSION_CONTEXT_FEATURES, calc_fusion_context_features

PHASE8 = RESULTS_DIR / "phase8"
TARGETS = ("homo", "lumo", "gap")
PRED_PREFIX = "expansion500k_full_hybrid"

warnings.filterwarnings("ignore", message="X does not have valid feature names")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, default=RAW_DIR / "phase8_expansion_500k.csv")
    parser.add_argument("--common-csv", type=Path, default=PHASE8 / "full_expansion500k_common_eval_predictions.csv")
    parser.add_argument("--emb-2d", type=Path, default=PHASE8 / "gps_expansion_500k_embeddings.pt")
    parser.add_argument("--emb-3d", type=Path, default=PHASE8 / "schnet_expansion_500k_embeddings.pt")
    parser.add_argument("--fusion", type=Path, default=MODELS_DIR / "phase8_hybrid_fusion_expansion_500k.pt")
    parser.add_argument("--hybrid-key", default="phase8_expansion_hybrid")
    archive = PHASE8 / "archive" / "legacy" / "head_posthoc"
    parser.add_argument("--out-json", type=Path, default=archive / "b3lyp_residual_calibrator_metrics.json")
    parser.add_argument("--out-md", type=Path, default=archive / "b3lyp_residual_calibrator_decision.md")
    parser.add_argument("--out-predictions", type=Path, default=archive / "b3lyp_residual_calibrator_common_predictions.csv")
    parser.add_argument("--cache-dir", type=Path, default=PHASE8)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--max-train-rows", type=int, default=None)
    return parser.parse_args()


def load_embedding_payload(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    payload = torch.load(path, weights_only=False, map_location="cpu")
    return payload["embeddings"].float(), payload["source_idx"].long()


def load_aligned_embeddings(args: argparse.Namespace) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    h2, idx2 = load_embedding_payload(args.emb_2d)
    h3, idx3 = load_embedding_payload(args.emb_3d)
    pos2 = {int(v): i for i, v in enumerate(idx2.tolist())}
    pos3 = {int(v): i for i, v in enumerate(idx3.tolist())}
    common = np.array(sorted(set(pos2).intersection(pos3)), dtype=np.int64)
    ii2 = torch.tensor([pos2[int(i)] for i in common], dtype=torch.long)
    ii3 = torch.tensor([pos3[int(i)] for i in common], dtype=torch.long)
    return h2[ii2], h3[ii3], common


@torch.no_grad()
def predict_fusion(h2: torch.Tensor, h3: torch.Tensor, fusion_path: Path, batch_size: int) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FusionHead("gate", 192, 0.0).to(device)
    model.load_state_dict(torch.load(fusion_path, weights_only=True, map_location=device))
    model.eval()
    rows = []
    for start in range(0, h2.shape[0], batch_size):
        end = min(start + batch_size, h2.shape[0])
        rows.append(model(h2[start:end].to(device), h3[start:end].to(device)).float().cpu().numpy())
    return np.concatenate(rows, axis=0)


@torch.no_grad()
def predict_v3_heads(h2: torch.Tensor, h3: torch.Tensor, hybrid_key: str, batch_size: int) -> dict[str, np.ndarray]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gps, schnet, fusion, _ = load_hybrid(device, key=hybrid_key)
    gps.eval()
    schnet.eval()
    fusion.eval()
    rows = {"gps": [], "schnet": [], "hybrid": []}
    for start in range(0, h2.shape[0], batch_size):
        end = min(start + batch_size, h2.shape[0])
        b2 = h2[start:end].to(device)
        b3 = h3[start:end].to(device)
        rows["gps"].append(gps.head(b2).float().cpu().numpy())
        rows["schnet"].append(schnet.head(b3).float().cpu().numpy())
        rows["hybrid"].append(fusion(b2, b3).float().cpu().numpy())
    return {name: np.concatenate(values, axis=0).astype(np.float32) for name, values in rows.items()}


def make_split(n: int, max_rows: int | None = None) -> dict[str, np.ndarray]:
    idx = np.random.RandomState(SEED).permutation(n)
    if max_rows is not None:
        idx = idx[:max_rows]
    n_train, n_val = int(0.8 * len(idx)), int(0.1 * len(idx))
    return {
        "train": idx[:n_train],
        "val": idx[n_train:n_train + n_val],
        "test": idx[n_train + n_val:],
    }


def context_features(smiles: pd.Series, cache_path: Path | None) -> np.ndarray:
    if cache_path is not None and cache_path.exists():
        return np.load(cache_path).astype(np.float32)
    rows = []
    for i, smi in enumerate(smiles.astype(str), start=1):
        rows.append(calc_fusion_context_features(smi))
        if i % 50000 == 0:
            print(f"  descriptors {i}/{len(smiles)}", flush=True)
    arr = np.stack(rows).astype(np.float32)
    if cache_path is not None:
        np.save(cache_path, arr)
    return arr


def feature_matrix(pred: np.ndarray, desc: np.ndarray) -> np.ndarray:
    return np.hstack([pred.astype(np.float32), desc.astype(np.float32)])


def stack_feature_matrix(heads: dict[str, np.ndarray], desc: np.ndarray) -> np.ndarray:
    return np.hstack([
        heads["gps"].astype(np.float32),
        heads["schnet"].astype(np.float32),
        heads["hybrid"].astype(np.float32),
        desc.astype(np.float32),
    ])


def target_metrics(y_true: np.ndarray, pred: np.ndarray) -> dict:
    rows = {}
    for i, target in enumerate(TARGETS):
        rows[target] = {
            "mae": float(mean_absolute_error(y_true[:, i], pred[:, i])),
            "r2": float(r2_score(y_true[:, i], pred[:, i])),
            "bias": float(np.mean(pred[:, i] - y_true[:, i])),
        }
    rows["average"] = {
        "mae": float(np.mean([rows[t]["mae"] for t in TARGETS])),
        "r2": float(np.mean([rows[t]["r2"] for t in TARGETS])),
    }
    return rows


def fit_ridge(X: np.ndarray, residual: np.ndarray, train: np.ndarray) -> list[Ridge]:
    models = []
    for i in range(residual.shape[1]):
        model = Ridge(alpha=10.0)
        model.fit(X[train], residual[train, i])
        models.append(model)
    return models


def fit_lgbm(X: np.ndarray, residual: np.ndarray, train: np.ndarray, val: np.ndarray) -> list[lgb.LGBMRegressor]:
    params = dict(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.9,
        reg_lambda=5.0,
        min_child_samples=200,
        random_state=SEED,
        n_jobs=-1,
        verbose=-1,
    )
    models = []
    for i in range(residual.shape[1]):
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X[train],
            residual[train, i],
            eval_set=[(X[val], residual[val, i])],
            callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(0)],
        )
        models.append(model)
    return models


def fit_lgbm_targets(X: np.ndarray, y: np.ndarray, train: np.ndarray, val: np.ndarray) -> list[lgb.LGBMRegressor]:
    params = dict(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.9,
        reg_lambda=5.0,
        min_child_samples=200,
        random_state=SEED,
        n_jobs=-1,
        verbose=-1,
    )
    models = []
    for i in range(y.shape[1]):
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X[train],
            y[train, i],
            eval_set=[(X[val], y[val, i])],
            callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(0)],
        )
        models.append(model)
    return models


def predict_models(models, X: np.ndarray) -> np.ndarray:
    return np.column_stack([m.predict(X) for m in models]).astype(np.float32)


def evaluate_by_bucket(df: pd.DataFrame, y: np.ndarray, baseline: np.ndarray, calibrated: np.ndarray) -> dict:
    out = {"all": {"baseline": target_metrics(y, baseline), "calibrated": target_metrics(y, calibrated)}}
    for name, sub in df.groupby("eval_set", dropna=False):
        idx = sub.index.to_numpy()
        out[str(name)] = {
            "baseline": target_metrics(y[idx], baseline[idx]),
            "calibrated": target_metrics(y[idx], calibrated[idx]),
        }
    return out


def delta_row(metrics: dict) -> dict:
    return {
        "avg_mae_delta": metrics["calibrated"]["average"]["mae"] - metrics["baseline"]["average"]["mae"],
        "gap_mae_delta": metrics["calibrated"]["gap"]["mae"] - metrics["baseline"]["gap"]["mae"],
    }


def write_decision(path: Path, result: dict) -> None:
    common = result["common_eval"]
    best_name = min(result["common_eval_models"], key=lambda n: common["all"][n]["average"]["mae"])
    baseline = common["all"]["baseline"]
    best = common["all"][best_name]
    delta = best["average"]["mae"] - baseline["average"]["mae"]
    verdict = "negative" if delta >= -0.001 else "positive"
    lines = [
        "# Phase 8 B3LYP Residual Calibrator Probe",
        "",
        "Date: 2026-07-06",
        "",
        "## Setup",
        "",
        "- Base: `phase8_expansion_hybrid` B3LYP v3.",
        "- Correction target: B3LYP labels only, not GW and not LoRA.",
        "- Residual features: v3 Hybrid HOMO/LUMO/Gap predictions + lightweight RDKit context descriptors.",
        "- Stack features: v3 GPS, SchNet, and Hybrid B3LYP outputs + the same descriptors.",
        "- Fit split: the same RandomState(42) 80/10/10 aligned expansion500k embedding split used by fusion-head probes.",
        "- External check: Phase 8 common eval (`ood1000` + `p8_targeted_hard`).",
        "",
        "## Common Eval MAE",
        "",
        "| model | HOMO | LUMO | Gap | avg |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in result["common_eval_models"]:
        row = common["all"][name]
        label = {
            "baseline": "v3 baseline",
            "constant_bias": "constant residual",
            "ridge": "ridge residual",
            "lightgbm": "LightGBM residual",
            "ridge_output_stack": "ridge output stack",
            "lightgbm_output_stack": "LightGBM output stack",
        }[name]
        lines.append(
            f"| {label} | {row['homo']['mae']:.4f} | {row['lumo']['mae']:.4f} | "
            f"{row['gap']['mae']:.4f} | {row['average']['mae']:.4f} |"
        )
    lines.extend([
        "",
        "## Common Eval Deltas Vs V3",
        "",
        "| scope | best model | avg delta | Gap delta |",
        "|---|---|---:|---:|",
    ])
    for scope in ["all", "ood1000", "p8_targeted_hard"]:
        candidates = {name: common[scope][name] for name in result["common_eval_models"] if name != "baseline"}
        best_scope = min(candidates, key=lambda n: candidates[n]["average"]["mae"])
        base_scope = common[scope]["baseline"]
        best_scope_metrics = common[scope][best_scope]
        lines.append(
            f"| {scope} | {best_scope} | "
            f"{best_scope_metrics['average']['mae'] - base_scope['average']['mae']:+.5f} | "
            f"{best_scope_metrics['gap']['mae'] - base_scope['gap']['mae']:+.5f} |"
        )
    lines.extend([
        "",
        "## Decision",
        "",
        f"Probe verdict: **{verdict}**. Best common-eval avg MAE delta versus v3 is `{delta:+.5f}` eV.",
    ])
    if verdict == "negative":
        lines.append("Do not promote a B3LYP residual calibrator unless a future version wins the external common eval, not just the internal split.")
    else:
        lines.append("Keep as a B3LYP-level candidate, but require a second validation slice before changing default inference.")
    lines.extend([
        "",
        "Artifacts:",
        "",
        "- `results/phase8/archive/legacy/head_posthoc/b3lyp_residual_calibrator_metrics.json`",
        "- `results/phase8/archive/legacy/head_posthoc/b3lyp_residual_calibrator_common_predictions.csv`",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    print("Loading aligned v3 embeddings", flush=True)
    h2, h3, source_idx = load_aligned_embeddings(args)
    train_df = pd.read_csv(args.train_csv)
    aligned_df = train_df.iloc[source_idx].reset_index(drop=True)
    y = aligned_df[list(TARGETS)].to_numpy(dtype=np.float32)
    pred_cache = args.cache_dir / "b3lyp_residual_calibrator_internal_preds.npy"
    if pred_cache.exists():
        pred = np.load(pred_cache).astype(np.float32)
    else:
        print("Predicting v3 internal aligned rows", flush=True)
        pred = predict_fusion(h2, h3, args.fusion, args.batch_size).astype(np.float32)
        np.save(pred_cache, pred)
    head_cache = args.cache_dir / "b3lyp_residual_calibrator_internal_head_preds.npz"
    if head_cache.exists():
        with np.load(head_cache) as d:
            internal_heads = {name: d[name].astype(np.float32) for name in ("gps", "schnet", "hybrid")}
    else:
        print("Predicting v3 GPS/SchNet/Hybrid internal rows", flush=True)
        internal_heads = predict_v3_heads(h2, h3, args.hybrid_key, args.batch_size)
        np.savez(head_cache, **internal_heads)
    pred = internal_heads["hybrid"]
    split = make_split(len(aligned_df), args.max_train_rows)
    print(f"Internal split {len(split['train'])}/{len(split['val'])}/{len(split['test'])}", flush=True)

    desc_cache = args.cache_dir / "b3lyp_residual_calibrator_train_context.npy"
    print("Loading/computing internal descriptors", flush=True)
    desc = context_features(aligned_df["canonical_smiles"].fillna(aligned_df["smiles"]), desc_cache)
    X = feature_matrix(pred, desc)
    scaler = StandardScaler().fit(X[split["train"]])
    Xs = scaler.transform(X).astype(np.float32)
    residual = y - pred

    const = residual[split["train"]].mean(axis=0, keepdims=True)
    print("Training ridge residual calibrator", flush=True)
    ridge = fit_ridge(Xs, residual, split["train"])
    print("Training LightGBM residual calibrator", flush=True)
    lgbm_models = fit_lgbm(Xs, residual, split["train"], split["val"])
    stack_X = stack_feature_matrix(internal_heads, desc)
    stack_scaler = StandardScaler().fit(stack_X[split["train"]])
    stack_Xs = stack_scaler.transform(stack_X).astype(np.float32)
    print("Training ridge output stack", flush=True)
    ridge_stack = fit_ridge(stack_Xs, y, split["train"])
    print("Training LightGBM output stack", flush=True)
    lgbm_stack = fit_lgbm_targets(stack_Xs, y, split["train"], split["val"])

    internal = {}
    for split_name, idx in split.items():
        base = pred[idx]
        internal[split_name] = {
            "baseline": target_metrics(y[idx], base),
            "constant_bias": target_metrics(y[idx], base + const),
            "ridge": target_metrics(y[idx], base + predict_models(ridge, Xs[idx])),
            "lightgbm": target_metrics(y[idx], base + predict_models(lgbm_models, Xs[idx])),
            "ridge_output_stack": target_metrics(y[idx], predict_models(ridge_stack, stack_Xs[idx])),
            "lightgbm_output_stack": target_metrics(y[idx], predict_models(lgbm_stack, stack_Xs[idx])),
        }

    print("Evaluating common eval", flush=True)
    common_df = pd.read_csv(args.common_csv)
    common_y = common_df[list(TARGETS)].to_numpy(dtype=np.float32)
    common_base = common_df[[f"{PRED_PREFIX}_{t}" for t in TARGETS]].to_numpy(dtype=np.float32)
    common_desc_cache = args.cache_dir / "b3lyp_residual_calibrator_common_context.npy"
    common_desc = context_features(common_df["canonical_smiles"].fillna(common_df["smiles"]), common_desc_cache)
    common_X = scaler.transform(feature_matrix(common_base, common_desc)).astype(np.float32)
    common_heads = {
        "gps": common_df[[f"expansion500k_full_gps_2d_{t}" for t in TARGETS]].to_numpy(dtype=np.float32),
        "schnet": common_df[[f"expansion500k_full_schnet_3d_{t}" for t in TARGETS]].to_numpy(dtype=np.float32),
        "hybrid": common_base,
    }
    common_stack_X = stack_scaler.transform(stack_feature_matrix(common_heads, common_desc)).astype(np.float32)
    common_preds = {
        "baseline": common_base,
        "constant_bias": common_base + const,
        "ridge": common_base + predict_models(ridge, common_X),
        "lightgbm": common_base + predict_models(lgbm_models, common_X),
        "ridge_output_stack": predict_models(ridge_stack, common_stack_X),
        "lightgbm_output_stack": predict_models(lgbm_stack, common_stack_X),
    }
    common_eval = {}
    for scope, idx in {"all": common_df.index.to_numpy()}.items():
        common_eval[scope] = {
            name: target_metrics(common_y[idx], value[idx])
            for name, value in common_preds.items()
        }
    for scope, sub in common_df.groupby("eval_set"):
        idx = sub.index.to_numpy()
        common_eval[str(scope)] = {
            name: target_metrics(common_y[idx], value[idx])
            for name, value in common_preds.items()
        }

    pred_df = common_df.copy()
    for name, value in common_preds.items():
        if name == "baseline":
            continue
        for i, target in enumerate(TARGETS):
            pred_df[f"{name}_{target}"] = value[:, i]
    pred_df.to_csv(args.out_predictions, index=False, encoding="utf-8")

    result = {
        "kind": "b3lyp_residual_calibrator_probe",
        "base": "phase8_expansion_hybrid",
        "n_aligned": int(len(aligned_df)),
        "source_idx_min": int(source_idx.min()),
        "source_idx_max": int(source_idx.max()),
        "max_train_rows": args.max_train_rows,
        "features": ["pred_homo", "pred_lumo", "pred_gap", *FUSION_CONTEXT_FEATURES],
        "stack_features": [
            "gps_homo", "gps_lumo", "gps_gap",
            "schnet_homo", "schnet_lumo", "schnet_gap",
            "hybrid_homo", "hybrid_lumo", "hybrid_gap",
            *FUSION_CONTEXT_FEATURES,
        ],
        "split": {k: int(len(v)) for k, v in split.items()},
        "internal": internal,
        "common_eval_models": list(common_preds.keys()),
        "common_eval": common_eval,
    }
    args.out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_decision(args.out_md, result)
    print(f"Metrics -> {args.out_json}", flush=True)
    print(f"Decision -> {args.out_md}", flush=True)
    best = min(common_preds, key=lambda n: common_eval["all"][n]["average"]["mae"])
    row = common_eval["all"][best]
    base = common_eval["all"]["baseline"]
    print(
        f"Best common eval: {best} avg={row['average']['mae']:.5f} "
        f"Gap={row['gap']['mae']:.5f} "
        f"delta_avg={row['average']['mae'] - base['average']['mae']:+.5f} "
        f"delta_gap={row['gap']['mae'] - base['gap']['mae']:+.5f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
