"""Train a bounded SchNetPack 2.x three-target model on cached ETKDG graphs."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from molgap.schnetpack import SchNetPackRegressor
from molgap.utils import ensure_dirs

TARGETS = ("HOMO", "LUMO", "Gap")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graphs",
        type=Path,
        default=Path("results/phase8/pyg_3d_graphs_etkdg_expansion_500k.pt"),
    )
    parser.add_argument("--max-samples", type=int, default=30_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--lr", type=float, default=2.1150972021685588e-4)
    parser.add_argument("--weight-decay", type=float, default=1.4656553886225336e-5)
    parser.add_argument("--hidden-channels", type=int, default=192)
    parser.add_argument("--num-interactions", type=int, default=6)
    parser.add_argument("--num-gaussians", type=int, default=50)
    parser.add_argument("--cutoff", type=float, default=6.0)
    parser.add_argument("--model-out", type=Path, required=True)
    parser.add_argument("--metrics-out", type=Path, required=True)
    parser.add_argument("--split-out", type=Path, required=True)
    return parser.parse_args()


def batches(graphs, batch_size: int, shuffle: bool, rng: np.random.RandomState):
    order = rng.permutation(len(graphs)) if shuffle else np.arange(len(graphs))
    for start in range(0, len(order), batch_size):
        yield [graphs[index] for index in order[start:start + batch_size]]


@torch.no_grad()
def evaluate(model, graphs, batch_size: int, device: torch.device) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    predictions, targets, total, count = [], [], 0.0, 0
    criterion = nn.L1Loss()
    rng = np.random.RandomState(0)
    for batch in batches(graphs, batch_size, False, rng):
        prediction, target = model(batch, device)
        loss = criterion(prediction, target)
        total += float(loss.detach().cpu()) * len(batch)
        count += len(batch)
        predictions.append(prediction.float().cpu().numpy())
        targets.append(target.float().cpu().numpy())
    return np.concatenate(predictions), np.concatenate(targets), total / max(count, 1)


def metrics(prediction: np.ndarray, target: np.ndarray) -> dict:
    out = {}
    for index, name in enumerate(TARGETS):
        residual = prediction[:, index] - target[:, index]
        mae = float(np.abs(residual).mean())
        total = float(((target[:, index] - target[:, index].mean()) ** 2).sum())
        r2 = float(1.0 - (residual**2).sum() / total) if total > 0 else float("nan")
        out[name] = {"mae": mae, "r2": r2}
    out["average"] = {
        "mae": float(np.mean([out[name]["mae"] for name in TARGETS])),
        "r2": float(np.mean([out[name]["r2"] for name in TARGETS])),
    }
    return out


def source_indices(graphs) -> list[int]:
    return [int(graph.source_idx.view(-1)[0]) for graph in graphs]


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("SchNetPack training requires a visible DCU accelerator")
    ensure_dirs(args.model_out.parent, args.metrics_out.parent, args.split_out.parent)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda")
    all_graphs = torch.load(args.graphs, weights_only=False)
    if len(all_graphs) < args.max_samples:
        raise RuntimeError(f"Requested {args.max_samples} graphs, found {len(all_graphs)}")
    selector = np.random.RandomState(args.seed)
    selected = [all_graphs[index] for index in selector.permutation(len(all_graphs))[:args.max_samples]]
    splitter = np.random.RandomState(args.seed)
    split = splitter.permutation(len(selected))
    train_end, val_end = int(0.8 * len(selected)), int(0.9 * len(selected))
    train = [selected[index] for index in split[:train_end]]
    validation = [selected[index] for index in split[train_end:val_end]]
    test = [selected[index] for index in split[val_end:]]
    torch.save(
        {"seed": args.seed, "train": source_indices(train), "val": source_indices(validation), "test": source_indices(test)},
        args.split_out,
    )

    model_config = {
        "hidden_channels": args.hidden_channels,
        "num_interactions": args.num_interactions,
        "num_gaussians": args.num_gaussians,
        "cutoff": args.cutoff,
    }
    model = SchNetPackRegressor(**model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = nn.L1Loss()
    best_val, best_epoch, best_state, wait = float("inf"), -1, None, 0
    history = []
    print(
        f"device={torch.cuda.get_device_name(0)} graphs={len(selected)} "
        f"split={len(train)}/{len(validation)}/{len(test)} config={model_config}",
        flush=True,
    )

    train_rng = np.random.RandomState(args.seed)
    for epoch in range(args.epochs):
        model.train()
        started = time.time()
        total, count = 0.0, 0
        for batch in batches(train, args.batch_size, True, train_rng):
            prediction, target = model(batch, device)
            loss = criterion(prediction, target)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss at epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            total += float(loss.detach().cpu()) * len(batch)
            count += len(batch)
        train_loss = total / max(count, 1)
        _, _, val_mae = evaluate(model, validation, args.batch_size, device)
        scheduler.step()
        improved = val_mae < best_val
        if improved:
            best_val, best_epoch, wait = val_mae, epoch, 0
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        else:
            wait += 1
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_mae": val_mae,
            "lr": optimizer.param_groups[0]["lr"],
            "time_s": time.time() - started,
        }
        history.append(row)
        print(
            f"ep{epoch:03d} train={train_loss:.5f} val={val_mae:.5f} "
            f"best={best_val:.5f}@{best_epoch} time={row['time_s']:.1f}s{' *' if improved else ''}",
            flush=True,
        )
        if wait >= args.patience:
            print(f"early_stop={epoch}", flush=True)
            break

    if best_state is None:
        raise RuntimeError("No finite checkpoint was produced")
    model.load_state_dict(best_state)
    prediction, target, test_mae = evaluate(model, test, args.batch_size, device)
    test_metrics = metrics(prediction, target)
    torch.save({"state_dict": best_state, "config": model_config, "seed": args.seed}, args.model_out)
    torch.cuda.synchronize()
    result = {
        "implementation": "schnetpack-2.x",
        "schnetpack": importlib.metadata.version("schnetpack"),
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(0),
        "graph_path": str(args.graphs),
        "n_graphs": len(selected),
        "split_sizes": {"train": len(train), "val": len(validation), "test": len(test)},
        "seed": args.seed,
        "model_config": model_config,
        "params": {"lr": args.lr, "weight_decay": args.weight_decay, "batch_size": args.batch_size},
        "best_val_mae": best_val,
        "best_epoch": best_epoch,
        "test_l1": test_mae,
        "test_metrics": test_metrics,
        "history": history,
    }
    args.metrics_out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
