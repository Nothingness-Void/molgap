"""Test P0 physics-consistent FusionHeads on the frozen expansion500k embeddings.

This is a head-only gate before porting a successful design to the routed v4
two-GPS path. It compares a soft algebraic penalty with a head that derives
LUMO from HOMO plus a non-negative Gap. The external common/OOD/P8-hard rows
are evaluated only after lambda selection on the internal validation split.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, r2_score
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.phase8.archive.legacy.selection.eval_full_replacement_common import (
    _build_graphs,
    _load_trio,
    _metric_blocks,
    _predict,
)
from molgap.constants import MODELS_DIR, RAW_DIR, RESULTS_DIR, SEED
from molgap.fusion import FusionHead
from molgap.archive.phase8_r05_physics_consistency.fusion import (
    StructuredPhysicsFusionHead,
    homo_lumo_gap_consistency_loss,
)

PHASE8 = RESULTS_DIR / "phase8"
ARCHIVE_DIR = PHASE8 / "archive" / "archive-r05-physics-consistency"
ARCHIVE_MODELS_DIR = ARCHIVE_DIR / "models"
TARGETS = ("homo", "lumo", "gap")
DISPLAY_TARGETS = ("HOMO", "LUMO", "Gap")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, default=RAW_DIR / "phase8_expansion_500k.csv")
    parser.add_argument(
        "--common-csv",
        type=Path,
        default=PHASE8 / "archive" / "legacy" / "pilots_30k" / "common_eval_30k_predictions.csv",
    )
    parser.add_argument("--emb-2d", type=Path, default=PHASE8 / "gps_expansion_500k_embeddings.pt")
    parser.add_argument("--emb-3d", type=Path, default=PHASE8 / "schnet_expansion_500k_embeddings.pt")
    parser.add_argument("--base-fusion", type=Path, default=MODELS_DIR / "phase8_hybrid_fusion_expansion_500k.pt")
    parser.add_argument("--soft-lambdas", default="0.01,0.05,0.1,0.25")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--bs-2d", type=int, default=256)
    parser.add_argument("--bs-3d", type=int, default=128)
    parser.add_argument("--bs-fusion", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--out-json", type=Path, default=ARCHIVE_DIR / "p0_physics_consistency_metrics.json")
    parser.add_argument("--out-md", type=Path, default=ARCHIVE_DIR / "p0_physics_consistency_decision.md")
    parser.add_argument("--predictions", type=Path, default=ARCHIVE_DIR / "p0_physics_consistency_common_predictions.csv")
    return parser.parse_args()


def _load_embeddings(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    payload = torch.load(path, weights_only=False, map_location="cpu")
    return payload["embeddings"].float(), payload["source_idx"].long()


def load_aligned(args: argparse.Namespace) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    h2, idx2 = _load_embeddings(args.emb_2d)
    h3, idx3 = _load_embeddings(args.emb_3d)
    pos2 = {int(value): i for i, value in enumerate(idx2.tolist())}
    pos3 = {int(value): i for i, value in enumerate(idx3.tolist())}
    common = np.array(sorted(set(pos2).intersection(pos3)), dtype=np.int64)
    i2 = torch.tensor([pos2[int(i)] for i in common], dtype=torch.long)
    i3 = torch.tensor([pos3[int(i)] for i in common], dtype=torch.long)
    labels = pd.read_csv(args.train_csv).iloc[common][list(TARGETS)].to_numpy(dtype=np.float32)
    return h2[i2], h3[i3], torch.tensor(labels)


def make_split(n: int) -> dict[str, np.ndarray]:
    indices = np.random.RandomState(SEED).permutation(n)
    n_train, n_val = int(0.8 * n), int(0.1 * n)
    return {
        "train": indices[:n_train],
        "val": indices[n_train:n_train + n_val],
        "test": indices[n_train + n_val:],
    }


def make_loader(h2, h3, y, idx, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        TensorDataset(h2[idx], h3[idx], y[idx]),
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=0,
    )


def metrics(y_true: np.ndarray, prediction: np.ndarray) -> dict:
    result = {}
    for index, name in enumerate(DISPLAY_TARGETS):
        result[name] = {
            "mae": float(mean_absolute_error(y_true[:, index], prediction[:, index])),
            "r2": float(r2_score(y_true[:, index], prediction[:, index])),
        }
    result["average"] = {
        "mae": float(np.mean([result[name]["mae"] for name in DISPLAY_TARGETS])),
        "r2": float(np.mean([result[name]["r2"] for name in DISPLAY_TARGETS])),
    }
    return result


def consistency(prediction: np.ndarray) -> dict:
    residual = prediction[:, 2] - (prediction[:, 1] - prediction[:, 0])
    absolute = np.abs(residual)
    return {
        "mean_abs_ev": float(absolute.mean()),
        "p50_abs_ev": float(np.quantile(absolute, 0.5)),
        "p90_abs_ev": float(np.quantile(absolute, 0.9)),
        "p99_abs_ev": float(np.quantile(absolute, 0.99)),
        "max_abs_ev": float(absolute.max()),
    }


@torch.no_grad()
def evaluate(model, h2, h3, y, idx, batch_size: int, device: torch.device) -> dict:
    model.eval()
    predictions, labels = [], []
    for batch_2d, batch_3d, batch_y in make_loader(h2, h3, y, idx, batch_size, False):
        predictions.append(model(batch_2d.to(device), batch_3d.to(device)).float().cpu().numpy())
        labels.append(batch_y.numpy())
    prediction = np.concatenate(predictions)
    result = metrics(np.concatenate(labels), prediction)
    result["physical_consistency"] = consistency(prediction)
    return result


def _transfer_structured_initialization(model: StructuredPhysicsFusionHead, base_state: dict) -> None:
    target_state = model.state_dict()
    transferable = {
        key: value for key, value in base_state.items()
        if key in target_state and target_state[key].shape == value.shape
    }
    model.load_state_dict(transferable, strict=False)
    with torch.no_grad():
        model.head[-1].weight[0].copy_(base_state["head.5.weight"][0])
        model.head[-1].bias[0].copy_(base_state["head.5.bias"][0])
        model.head[-1].weight[1].copy_(base_state["head.5.weight"][2])
        model.head[-1].bias[1].copy_(base_state["head.5.bias"][2])


def train_candidate(
    *,
    name: str,
    model: torch.nn.Module,
    soft_lambda: float | None,
    args: argparse.Namespace,
    h2: torch.Tensor,
    h3: torch.Tensor,
    y: torch.Tensor,
    split: dict[str, np.ndarray],
    device: torch.device,
) -> tuple[dict, dict]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=4, factor=0.5, min_lr=1e-6
    )
    train_loader = make_loader(h2, h3, y, split["train"], args.batch_size, True)
    best_gap, best_state, best_epoch, waiting = float("inf"), None, -1, 0
    log = []
    for epoch in range(args.epochs):
        started = time.time()
        model.train()
        for batch_2d, batch_3d, batch_y in train_loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch_2d.to(device), batch_3d.to(device))
            target_loss = F.l1_loss(prediction, batch_y.to(device))
            if soft_lambda is None:
                loss = target_loss
            else:
                loss = target_loss + soft_lambda * homo_lumo_gap_consistency_loss(prediction)
            loss.backward()
            optimizer.step()
        validation = evaluate(model, h2, h3, y, split["val"], args.batch_size, device)
        val_gap = validation["Gap"]["mae"]
        scheduler.step(val_gap)
        if val_gap < best_gap:
            best_gap = val_gap
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_epoch = epoch
            waiting = 0
        else:
            waiting += 1
        log.append({
            "epoch": epoch,
            "validation_gap_mae": float(val_gap),
            "validation_average_mae": float(validation["average"]["mae"]),
            "validation_consistency_mean_abs_ev": float(validation["physical_consistency"]["mean_abs_ev"]),
            "time_s": time.time() - started,
        })
        print(f"{name} ep{epoch:03d} val_gap={val_gap:.5f} best={best_gap:.5f}@{best_epoch}", flush=True)
        if waiting >= args.patience:
            break
    model.load_state_dict(best_state)
    return {
        "best_epoch": int(best_epoch),
        "best_validation_gap_mae": float(best_gap),
        "validation": evaluate(model, h2, h3, y, split["val"], args.batch_size, device),
        "internal_test": evaluate(model, h2, h3, y, split["test"], args.batch_size, device),
        "log": log,
    }, model.state_dict()


def save_state(name: str, state: dict) -> Path:
    ARCHIVE_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = ARCHIVE_MODELS_DIR / f"phase8_p0_{name}.pt"
    torch.save(state, path)
    return path


def external_evaluate(args, candidates: dict[str, torch.nn.Module], device: torch.device) -> tuple[dict, pd.DataFrame]:
    print("Building common evaluation graphs once", flush=True)
    evaluation = pd.read_csv(args.common_csv)
    evaluation, graphs_2d, graphs_3d = _build_graphs(evaluation)
    metrics_by_candidate = {}
    prediction_frame = evaluation.copy()
    for name, model in candidates.items():
        print(f"Common evaluation: {name}", flush=True)
        gps_models, schnet, _ = _load_trio(
            MODELS_DIR / "phase8_gps_expansion_500k.pt",
            MODELS_DIR / "phase8_schnet_expansion_500k.pt",
            args.base_fusion,
            device,
        )
        model = model.to(device).eval()
        prediction = _predict(gps_models, schnet, model, graphs_2d, graphs_3d, args, device)["hybrid"]
        block = _metric_blocks(evaluation, prediction)
        block["physical_consistency"] = consistency(prediction)
        metrics_by_candidate[name] = block
        for index, target in enumerate(TARGETS):
            prediction_frame[f"{name}_{target}"] = prediction[:, index]
    return metrics_by_candidate, prediction_frame


def delta_vs_baseline(external: dict, name: str) -> dict:
    baseline = external["baseline"]
    result = {}
    for block in ("all", "ood1000", "p8_targeted_hard"):
        result[block] = {
            "average_mae": float(external[name][block]["average"]["mae"] - baseline[block]["average"]["mae"]),
            "gap_mae": float(external[name][block]["Gap"]["mae"] - baseline[block]["Gap"]["mae"]),
        }
    return result


def write_decision(path: Path, result: dict) -> None:
    external = result["external_common"]
    lines = [
        "# Phase 8 P0 Physics-Consistent Head Probe",
        "",
        "## Design",
        "",
        "- Labels were audited on all 500,000 expansion rows: `Gap = LUMO - HOMO` holds within `3.56e-15 eV`.",
        "- Encoders stay frozen. The probe starts from the v3 FusionHead and selects the soft-loss lambda only by internal validation Gap MAE.",
        "- The structured head emits HOMO and a non-negative Gap, then derives LUMO exactly.",
        "",
        "## Common Evaluation Deltas vs Re-evaluated v3 Baseline",
        "",
        "| candidate | all Gap | OOD Gap | P8 hard Gap | all avg |",
        "|---|---:|---:|---:|---:|",
    ]
    advances = []
    for name, delta in result["external_delta"].items():
        lines.append(
            f"| {name} | {delta['all']['gap_mae']:+.5f} | {delta['ood1000']['gap_mae']:+.5f} | "
            f"{delta['p8_targeted_hard']['gap_mae']:+.5f} | {delta['all']['average_mae']:+.5f} |"
        )
        if (
            delta["all"]["gap_mae"] <= -0.001
            and delta["ood1000"]["gap_mae"] <= 0
            and delta["p8_targeted_hard"]["gap_mae"] <= 0
        ):
            advances.append(name)
    lines.extend(["", "## Decision", ""])
    if advances:
        lines.append(
            "Advance only " + ", ".join(f"`{name}`" for name in advances)
            + " to the routed-v4 port; no default checkpoint changes in this probe."
        )
    else:
        lines.append(
            "Negative at the v3 gate: no candidate reaches a >=0.001 eV common Gap improvement "
            "without OOD/P8-hard regression. Do not port P0 to routed v4 or change defaults."
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    h2, h3, y = load_aligned(args)
    split = make_split(len(y))
    print(f"Aligned split {len(split['train'])}/{len(split['val'])}/{len(split['test'])}", flush=True)
    base_state = torch.load(args.base_fusion, weights_only=True, map_location="cpu")

    baseline = FusionHead("gate", 192, 0.0).to(device)
    baseline.load_state_dict(base_state)
    trained = {
        "baseline": {
            "validation": evaluate(baseline, h2, h3, y, split["val"], args.batch_size, device),
            "internal_test": evaluate(baseline, h2, h3, y, split["test"], args.batch_size, device),
        }
    }
    candidates = {"baseline": baseline}
    soft_names = []
    for value in (float(item) for item in args.soft_lambdas.split(",") if item.strip()):
        name = f"soft_lambda_{value:g}".replace(".", "p")
        model = FusionHead("gate", 192, 0.0).to(device)
        model.load_state_dict(base_state)
        report, state = train_candidate(
            name=name, model=model, soft_lambda=value, args=args, h2=h2, h3=h3, y=y,
            split=split, device=device,
        )
        report["lambda"] = value
        report["model"] = str(save_state(name, state))
        trained[name] = report
        soft_names.append(name)

    selected_soft = min(soft_names, key=lambda name: trained[name]["best_validation_gap_mae"])
    soft_model = FusionHead("gate", 192, 0.0).to(device)
    soft_model.load_state_dict(torch.load(trained[selected_soft]["model"], weights_only=True, map_location=device))
    candidates[selected_soft] = soft_model

    structured_name = "structured_physics"
    structured = StructuredPhysicsFusionHead("gate", 192, 0.0).to(device)
    _transfer_structured_initialization(structured, base_state)
    report, state = train_candidate(
        name=structured_name, model=structured, soft_lambda=None, args=args, h2=h2, h3=h3, y=y,
        split=split, device=device,
    )
    report["model"] = str(save_state(structured_name, state))
    trained[structured_name] = report
    structured.load_state_dict(state)
    candidates[structured_name] = structured

    external, predictions = external_evaluate(args, candidates, device)
    predictions.to_csv(args.predictions, index=False, encoding="utf-8")
    deltas = {name: delta_vs_baseline(external, name) for name in candidates if name != "baseline"}
    result = {
        "kind": "p0_physics_consistent_fusion_head_probe",
        "base": "phase8_expansion_hybrid_v3",
        "label_audit": {
            "source": "data/raw/phase8_expansion_500k.csv",
            "n_rows": 500000,
            "max_abs_gap_minus_lumo_minus_homo_ev": 3.552713678800501e-15,
        },
        "split": {key: int(len(value)) for key, value in split.items()},
        "soft_lambda_selection": selected_soft,
        "trained": trained,
        "external_common": external,
        "external_delta": deltas,
    }
    args.out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_decision(args.out_md, result)
    print(f"Metrics -> {args.out_json}", flush=True)
    print(f"Decision -> {args.out_md}", flush=True)


if __name__ == "__main__":
    main()
