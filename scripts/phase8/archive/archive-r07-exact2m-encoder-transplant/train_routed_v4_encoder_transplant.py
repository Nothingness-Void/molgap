"""Paired head-only test of exact-2M GPS encoders in the routed-v4 architecture.

The 500K SchNet embeddings, ETKDG labels, fusion architecture, split, and route
threshold stay fixed.  For every seed, the script trains matching 500K-control
and exact-2M-transplant base/dual heads so encoder effects are not confused with
head initialization noise.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from molgap.constants import MODELS_DIR, RESULTS_DIR, SEED
from molgap.fusion import FusionHead


PHASE8 = RESULTS_DIR / "phase8"
DEFAULT_OUT = PHASE8 / "archive" / "archive-r07-exact2m-encoder-transplant"
TARGETS = ("HOMO", "LUMO", "Gap")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-gps7", type=Path, default=PHASE8 / "gps_expansion_500k_embeddings.pt")
    parser.add_argument("--control-gps9", type=Path, default=PHASE8 / "gps_arch_depth9_embeddings.pt")
    parser.add_argument("--candidate-gps", type=Path, required=True,
                        help="payload containing exact-2M gps7/gps9/source_idx")
    parser.add_argument("--schnet", type=Path, default=PHASE8 / "schnet_expansion_500k_embeddings.pt")
    parser.add_argument("--graphs", type=Path,
                        default=PHASE8 / "pyg_3d_graphs_etkdg_expansion_500k.pt")
    parser.add_argument("--label-cache", type=Path, default=None)
    parser.add_argument("--production-base", type=Path,
                        default=MODELS_DIR / "phase8_hybrid_fusion_expansion_500k.pt")
    parser.add_argument("--production-dual", type=Path,
                        default=MODELS_DIR / "phase8_hybrid_fusion_expansion_500k_dualgps.pt")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--variants", choices=("both", "control", "candidate"), default="both")
    parser.add_argument("--seeds", type=int, nargs="+", default=(42, 43, 44))
    parser.add_argument("--split-seed", type=int, default=SEED)
    parser.add_argument("--prefix-rows", type=int, default=500_000)
    parser.add_argument("--threshold", type=float, default=4.0)
    parser.add_argument("--hidden", type=int, default=192)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--eval-batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=5.4e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="bounded preflight only; samples after alignment")
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def atomic_torch_save(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def atomic_json_dump(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def load_standard_embedding(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not {"embeddings", "source_idx"} <= set(payload):
        raise ValueError(f"{path} must contain embeddings and source_idx")
    emb = payload["embeddings"].float().contiguous()
    idx = payload["source_idx"].long().view(-1)
    if len(emb) != len(idx):
        raise ValueError(f"row mismatch in {path}: {len(emb)} != {len(idx)}")
    return emb, idx


def load_candidate(path: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not {"gps7", "gps9", "source_idx"} <= set(payload):
        raise ValueError(f"{path} must contain gps7, gps9, and source_idx")
    gps7 = payload["gps7"].float().contiguous()
    gps9 = payload["gps9"].float().contiguous()
    idx = payload["source_idx"].long().view(-1)
    if len(gps7) != len(gps9) or len(gps7) != len(idx):
        raise ValueError("candidate GPS payload row counts differ")
    return gps7, gps9, idx


def load_or_build_labels(graphs_path: Path, cache_path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        if isinstance(payload, dict) and {"labels", "source_idx"} <= set(payload):
            return payload["labels"].float(), payload["source_idx"].long().view(-1)

    print(f"Extracting labels from {graphs_path} ...", flush=True)
    labels: list[torch.Tensor] = []
    source_idx: list[int] = []
    for graph in torch.load(graphs_path, map_location="cpu", weights_only=False):
        labels.append(graph.y.view(-1, 3)[0].float())
        source_idx.append(int(graph.source_idx.view(-1)[0]))
    payload = {
        "labels": torch.stack(labels),
        "source_idx": torch.tensor(source_idx, dtype=torch.long),
        "source_graphs": str(graphs_path),
    }
    atomic_torch_save(payload, cache_path)
    print(f"Label cache -> {cache_path}", flush=True)
    return payload["labels"], payload["source_idx"]


def aligned_positions(indices: torch.Tensor, common: np.ndarray, name: str) -> torch.Tensor:
    """Locate sorted common source ids without materializing Python dictionaries."""
    values = indices.numpy()
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    if len(sorted_values) > 1 and np.any(sorted_values[1:] == sorted_values[:-1]):
        raise ValueError(f"duplicate source_idx in {name}")
    locations = np.searchsorted(sorted_values, common)
    if np.any(locations >= len(sorted_values)) or not np.array_equal(sorted_values[locations], common):
        raise ValueError(f"failed to align source_idx in {name}")
    return torch.from_numpy(order[locations].copy()).long()


def align_inputs(args: argparse.Namespace) -> dict[str, torch.Tensor]:
    control7, idx7 = load_standard_embedding(args.control_gps7)
    control9, idx9 = load_standard_embedding(args.control_gps9)
    candidate7, candidate9, idx_candidate = load_candidate(args.candidate_gps)
    schnet, idx_schnet = load_standard_embedding(args.schnet)
    label_cache = args.label_cache or (args.out_dir / "labels_expansion500k_etkdg.pt")
    labels, idx_labels = load_or_build_labels(args.graphs, label_cache)

    sources = {
        "control7": (control7, idx7),
        "control9": (control9, idx9),
        "candidate7": (candidate7, idx_candidate),
        "candidate9": (candidate9, idx_candidate),
        "schnet": (schnet, idx_schnet),
        "labels": (labels, idx_labels),
    }
    common = np.arange(args.prefix_rows, dtype=np.int64)
    for _, indices in sources.values():
        common = np.intersect1d(common, indices.numpy(), assume_unique=False)
    if len(common) == 0:
        raise ValueError("no aligned rows")
    if args.max_samples is not None:
        rng = np.random.RandomState(args.split_seed)
        common = np.sort(rng.permutation(common)[:args.max_samples])

    aligned: dict[str, torch.Tensor] = {}
    for name, (tensor, indices) in sources.items():
        take = aligned_positions(indices, common, name)
        aligned[name] = tensor[take].contiguous()
    aligned["source_idx"] = torch.from_numpy(common.copy()).long()

    y = aligned["labels"]
    if y.shape[1] != 3 or not torch.isfinite(y).all():
        raise ValueError("labels must be finite [N,3]")
    for name in ("control7", "control9", "candidate7", "candidate9", "schnet"):
        tensor = aligned[name]
        if tensor.shape != (len(common), 192) or not torch.isfinite(tensor).all():
            raise ValueError(f"{name} must be finite [N,192], got {tuple(tensor.shape)}")
    return aligned


def make_split(n: int, seed: int) -> dict[str, np.ndarray]:
    idx = np.random.RandomState(seed).permutation(n)
    n_train, n_val = int(0.8 * n), int(0.1 * n)
    return {
        "train": idx[:n_train],
        "val": idx[n_train:n_train + n_val],
        "test": idx[n_train + n_val:],
    }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict(model: nn.Module, h2: torch.Tensor, h3: torch.Tensor, idx: np.ndarray,
            batch_size: int, device: torch.device) -> np.ndarray:
    loader = DataLoader(TensorDataset(h2[idx], h3[idx]), batch_size=batch_size,
                        shuffle=False, pin_memory=device.type == "cuda")
    model.eval()
    result = []
    for batch_2d, batch_3d in loader:
        result.append(model(batch_2d.to(device), batch_3d.to(device)).float().cpu())
    return torch.cat(result).numpy()


def metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    error = np.abs(pred - y)
    result = {
        target: {"mae": float(error[:, i].mean())}
        for i, target in enumerate(TARGETS)
    }
    result["average"] = {"mae": float(error.mean())}
    return result


def train_head(name: str, dim_2d: int, h2: torch.Tensor, h3: torch.Tensor,
               y: torch.Tensor, split: dict[str, np.ndarray], seed: int,
               args: argparse.Namespace, device: torch.device) -> tuple[nn.Module, dict]:
    set_seed(seed)
    model = FusionHead("gate", args.hidden, 0.0, dim_2d=dim_2d, dim_3d=192).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=8, factor=0.5, min_lr=1e-6,
    )
    criterion = nn.L1Loss()
    run_dir = args.out_dir / f"seed{seed}"
    last_path = run_dir / f"{name}_last.pt"
    best_path = run_dir / f"{name}_best.pt"
    start_epoch, best_val, best_epoch, wait, log = 0, float("inf"), -1, 0, []

    if last_path.exists() and not args.no_resume:
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val = float(checkpoint["best_val"])
        best_epoch = int(checkpoint["best_epoch"])
        wait = int(checkpoint["wait"])
        log = list(checkpoint["log"])
        print(f"{name} seed={seed}: resume epoch {start_epoch}", flush=True)

    train_idx = split["train"]
    val_idx = split["val"]
    train_dataset = TensorDataset(h2[train_idx], h3[train_idx], y[train_idx])
    val_loader = DataLoader(TensorDataset(h2[val_idx], h3[val_idx], y[val_idx]),
                            batch_size=args.eval_batch_size, shuffle=False,
                            pin_memory=device.type == "cuda")
    for epoch in range(start_epoch, args.epochs):
        started = time.time()
        generator = torch.Generator().manual_seed(seed * 100_000 + epoch)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size, shuffle=True, generator=generator,
            pin_memory=device.type == "cuda", num_workers=0,
        )
        model.train()
        train_total, train_n = 0.0, 0
        for batch_2d, batch_3d, batch_y in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_2d.to(device), batch_3d.to(device)), batch_y.to(device))
            loss.backward()
            optimizer.step()
            train_total += float(loss.item()) * len(batch_y)
            train_n += len(batch_y)

        model.eval()
        val_total, val_n = 0.0, 0
        with torch.no_grad():
            for batch_2d, batch_3d, batch_y in val_loader:
                loss = criterion(model(batch_2d.to(device), batch_3d.to(device)), batch_y.to(device))
                val_total += float(loss.item()) * len(batch_y)
                val_n += len(batch_y)
        val = val_total / val_n
        scheduler.step(val)
        improved = val < best_val
        if improved:
            best_val, best_epoch, wait = val, epoch, 0
            atomic_torch_save(model.state_dict(), best_path)
        else:
            wait += 1
        row = {
            "epoch": epoch,
            "train_mae": train_total / train_n,
            "val_mae": val,
            "best_val_mae": best_val,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "time_s": time.time() - started,
        }
        log.append(row)
        atomic_torch_save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_val": best_val,
            "best_epoch": best_epoch,
            "wait": wait,
            "log": log,
        }, last_path)
        atomic_json_dump(log, run_dir / f"{name}_train_log.json")
        print(
            f"{name} seed={seed} ep={epoch:03d} train={row['train_mae']:.5f} "
            f"val={val:.5f} best={best_val:.5f}@{best_epoch} {row['time_s']:.1f}s"
            + (" *" if improved else ""), flush=True,
        )
        if wait >= args.patience:
            break

    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    return model, {
        "best_val_mae": best_val,
        "best_epoch": best_epoch,
        "epochs_completed": len(log),
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "n_params": sum(parameter.numel() for parameter in model.parameters()),
    }


def load_production_heads(args: argparse.Namespace, device: torch.device) -> tuple[nn.Module, nn.Module]:
    base = FusionHead("gate", args.hidden, 0.0, dim_2d=192, dim_3d=192).to(device)
    dual = FusionHead("gate", args.hidden, 0.0, dim_2d=384, dim_3d=192).to(device)
    base.load_state_dict(torch.load(args.production_base, map_location=device, weights_only=True))
    dual.load_state_dict(torch.load(args.production_dual, map_location=device, weights_only=True))
    return base.eval(), dual.eval()


def routed_predictions(base: np.ndarray, dual: np.ndarray, route: np.ndarray) -> np.ndarray:
    result = base.copy()
    result[route] = dual[route]
    return result


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    data = align_inputs(args)
    split = make_split(len(data["labels"]), args.split_seed)
    print(
        f"Aligned={len(data['labels']):,} split="
        f"{len(split['train']):,}/{len(split['val']):,}/{len(split['test']):,}", flush=True,
    )

    test_idx = split["test"]
    y_test = data["labels"][test_idx].numpy()
    production_base, production_dual = load_production_heads(args, device)
    production_base_pred = predict(
        production_base, data["control7"], data["schnet"], test_idx,
        args.eval_batch_size, device,
    )
    production_dual_pred = predict(
        production_dual, torch.cat([data["control7"], data["control9"]], dim=1),
        data["schnet"], test_idx, args.eval_batch_size, device,
    )
    fixed_route = production_base_pred[:, 2] < args.threshold
    production_routed = routed_predictions(production_base_pred, production_dual_pred, fixed_route)

    result = {
        "experiment": "routed_v4_exact2m_encoder_transplant",
        "device": str(device),
        "n_aligned": len(data["labels"]),
        "source_idx_min": int(data["source_idx"].min()),
        "source_idx_max": int(data["source_idx"].max()),
        "split_seed": args.split_seed,
        "split": {name: len(values) for name, values in split.items()},
        "threshold_eV": args.threshold,
        "fixed_route_n": int(fixed_route.sum()),
        "production_reference": metrics(y_test, production_routed),
        "seeds": {},
    }
    atomic_json_dump(result, args.out_dir / "metrics.partial.json")

    variants = ("control", "candidate") if args.variants == "both" else (args.variants,)
    for seed in args.seeds:
        seed_result: dict[str, object] = {}
        for variant in variants:
            gps7 = data[f"{variant}7"]
            gps9 = data[f"{variant}9"]
            base, base_train = train_head(
                f"{variant}_base", 192, gps7, data["schnet"], data["labels"],
                split, seed, args, device,
            )
            dual, dual_train = train_head(
                f"{variant}_dual", 384, torch.cat([gps7, gps9], dim=1), data["schnet"],
                data["labels"], split, seed, args, device,
            )
            base_pred = predict(base, gps7, data["schnet"], test_idx, args.eval_batch_size, device)
            dual_pred = predict(
                dual, torch.cat([gps7, gps9], dim=1), data["schnet"], test_idx,
                args.eval_batch_size, device,
            )
            self_route = base_pred[:, 2] < args.threshold
            seed_result[variant] = {
                "base_training": base_train,
                "dual_training": dual_train,
                "base": metrics(y_test, base_pred),
                "dual": metrics(y_test, dual_pred),
                "fixed_route": metrics(y_test, routed_predictions(base_pred, dual_pred, fixed_route)),
                "self_route": metrics(y_test, routed_predictions(base_pred, dual_pred, self_route)),
                "self_route_n": int(self_route.sum()),
            }
        if "control" in seed_result and "candidate" in seed_result:
            seed_result["paired_delta_candidate_minus_control"] = {}
            for route_name in ("fixed_route", "self_route"):
                seed_result["paired_delta_candidate_minus_control"][route_name] = {
                    target: float(
                        seed_result["candidate"][route_name][target]["mae"]
                        - seed_result["control"][route_name][target]["mae"]
                    )
                    for target in (*TARGETS, "average")
                }
        result["seeds"][str(seed)] = seed_result
        atomic_json_dump(result, args.out_dir / "metrics.partial.json")

    if args.variants == "both":
        summary = {}
        for route_name in ("fixed_route", "self_route"):
            summary[route_name] = {}
            for target in (*TARGETS, "average"):
                values = [
                    result["seeds"][str(seed)]["paired_delta_candidate_minus_control"][route_name][target]
                    for seed in args.seeds
                ]
                summary[route_name][target] = {
                    "mean_delta_mae": float(np.mean(values)),
                    "std_delta_mae": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    "all_seeds_better": bool(all(value < 0 for value in values)),
                    "values": values,
                }
        result["paired_summary"] = summary

    atomic_json_dump(result, args.out_dir / "metrics.json")
    partial = args.out_dir / "metrics.partial.json"
    if partial.exists():
        partial.unlink()
    print(f"Metrics -> {args.out_dir / 'metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
