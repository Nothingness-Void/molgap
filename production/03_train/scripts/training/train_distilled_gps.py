"""Distill a fixed dual-expert teacher into one fusion-compatible GPS encoder."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

from molgap.distillation import (
    TeacherEmbeddingSpec,
    atomic_json_write,
    atomic_torch_save,
    build_teacher_target_parts,
    extract_gps_embedding_parts,
    load_teacher_targets,
    merge_embedding_prefix,
    sha256_file,
)
from molgap.gps import GPSWrapper
from molgap.multi2d import metric_block


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graphs", type=Path, required=True)
    parser.add_argument("--student-init", type=Path, required=True)
    parser.add_argument(
        "--teacher",
        nargs=4,
        action="append",
        metavar=("NAME", "GPS7_DIR", "GPS9_DIR", "HEAD"),
        required=True,
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model-out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=4e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--teacher-weight", type=float, default=0.70)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fusion-prefix-rows", type=int, default=997445)
    return parser.parse_args()


def _evaluate(model, graphs, indices, teacher_targets, batch_size, device):
    loader = DataLoader(Subset(graphs, indices.tolist()), batch_size=batch_size, shuffle=False, num_workers=0)
    predictions, targets, teachers = [], [], []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            source_idx = batch.source_idx.view(-1).long()
            batch = batch.to(device)
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                prediction = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            predictions.append(prediction.float().cpu())
            targets.append(batch.y.float().cpu())
            teachers.append(teacher_targets[source_idx])
    return torch.cat(predictions).numpy(), torch.cat(targets).numpy(), torch.cat(teachers).numpy()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.teacher_weight <= 1.0:
        raise ValueError("--teacher-weight must be in [0, 1]")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.run_dir / "training_state.pt"
    metrics_path = args.run_dir / "metrics.json"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    specs = [
        TeacherEmbeddingSpec(name, Path(gps7), Path(gps9), Path(head))
        for name, gps7, gps9, head in args.teacher
    ]
    target_manifest = build_teacher_target_parts(
        specs,
        out_dir=args.run_dir / "teacher_targets",
        device=device,
    )
    teacher_targets = load_teacher_targets(args.run_dir / "teacher_targets/manifest.json")
    print(f"teacher targets: {tuple(teacher_targets.shape)}", flush=True)

    graphs = torch.load(args.graphs, map_location="cpu", weights_only=False)
    if len(graphs) != len(teacher_targets):
        raise ValueError(f"graphs={len(graphs):,}, teacher targets={len(teacher_targets):,}")
    source_idx = torch.as_tensor(
        [int(graph.source_idx.view(-1)[0]) for graph in graphs], dtype=torch.long
    )
    if not torch.equal(source_idx, torch.arange(len(graphs), dtype=torch.long)):
        raise ValueError("Graph source_idx is not contiguous")

    permutation = np.random.RandomState(args.seed).permutation(len(graphs))
    n_train, n_val = int(0.8 * len(graphs)), int(0.1 * len(graphs))
    train_idx = permutation[:n_train]
    val_idx = permutation[n_train:n_train + n_val]
    test_idx = permutation[n_train + n_val:]
    train_loader = DataLoader(Subset(graphs, train_idx.tolist()), batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(Subset(graphs, val_idx.tolist()), batch_size=args.eval_batch_size, shuffle=False, num_workers=0)

    model = GPSWrapper(hidden_channels=192, num_layers=7, num_heads=4, dropout=0.05).to(device)
    model.load_state_dict(torch.load(args.student_init, map_location=device, weights_only=True))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())
    criterion = nn.L1Loss()
    start_epoch, best_val, best_epoch, wait, best_state, log = 0, float("inf"), -1, 0, None, []
    config = {
        "graphs": str(args.graphs),
        "n_graphs": len(graphs),
        "student_init_sha256": sha256_file(args.student_init),
        "teacher_manifest_sha256": sha256_file(args.run_dir / "teacher_targets/manifest.json"),
        "teacher_weight": args.teacher_weight,
        "seed": args.seed,
    }
    if checkpoint_path.is_file():
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if state.get("config") != config:
            raise ValueError("Resume checkpoint configuration differs")
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])
        scaler.load_state_dict(state["scaler_state"])
        start_epoch = int(state["next_epoch"])
        best_val, best_epoch, wait = float(state["best_val"]), int(state["best_epoch"]), int(state["wait"])
        best_state, log = state["best_state"], list(state["log"])
        print(f"resuming epoch {start_epoch}", flush=True)

    for epoch in range(start_epoch, args.epochs):
        started = time.time()
        model.train()
        train_total, train_rows = 0.0, 0
        for batch in train_loader:
            indices = batch.source_idx.view(-1).long()
            soft_target = teacher_targets[indices].to(device, non_blocking=True)
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                prediction = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                teacher_loss = criterion(prediction, soft_target)
                label_loss = criterion(prediction, batch.y)
                loss = args.teacher_weight * teacher_loss + (1.0 - args.teacher_weight) * label_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            scaler.step(optimizer)
            scaler.update()
            train_total += loss.item() * batch.num_graphs
            train_rows += batch.num_graphs

        model.eval()
        val_total, val_rows = 0.0, 0
        with torch.inference_mode():
            for batch in val_loader:
                batch = batch.to(device)
                with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                    prediction = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                    val_loss = criterion(prediction, batch.y)
                val_total += val_loss.item() * batch.num_graphs
                val_rows += batch.num_graphs
        val_mae = val_total / max(val_rows, 1)
        scheduler.step()
        improved = val_mae < best_val
        if improved:
            best_val, best_epoch, wait = val_mae, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            wait += 1
        row = {
            "epoch": epoch,
            "train_distillation_loss": train_total / max(train_rows, 1),
            "val_label_mae_eV": val_mae,
            "best_val_label_mae_eV": best_val,
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": time.time() - started,
        }
        log.append(row)
        atomic_torch_save(
            {
                "config": config,
                "next_epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "scaler_state": scaler.state_dict(),
                "best_val": best_val,
                "best_epoch": best_epoch,
                "wait": wait,
                "best_state": best_state,
                "log": log,
            },
            checkpoint_path,
        )
        atomic_json_write(
            {
                "complete": False,
                "config": config,
                "next_epoch": epoch + 1,
                "best_val_label_mae_eV": best_val,
                "best_epoch": best_epoch,
                "log": log,
            },
            metrics_path,
        )
        print(
            f"ep{epoch:03d} train={row['train_distillation_loss']:.5f} "
            f"val={val_mae:.5f} best={best_val:.5f}@{best_epoch} "
            f"{row['seconds']:.1f}s{' *' if improved else ''}",
            flush=True,
        )
        if wait >= args.patience:
            print(f"early stop at epoch {epoch}", flush=True)
            break

    if best_state is None:
        raise RuntimeError("No finite student checkpoint was produced")
    model.load_state_dict(best_state)
    atomic_torch_save(best_state, args.model_out)
    prediction, target, teacher = _evaluate(
        model, graphs, test_idx, teacher_targets, args.eval_batch_size, device
    )
    prediction_path = args.run_dir / "test_predictions.npz"
    temporary_prediction_path = args.run_dir / ".test_predictions.npz.tmp"
    with temporary_prediction_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            source_idx=test_idx,
            target=target,
            teacher=teacher,
            student=prediction,
        )
    os.replace(temporary_prediction_path, prediction_path)
    embedding_manifest = extract_gps_embedding_parts(
        model,
        graphs,
        model_path=args.model_out,
        out_dir=args.run_dir / "student_embeddings",
        device=device,
        batch_size=args.eval_batch_size,
        chunk_size=50_000,
    )
    fusion_prefix = merge_embedding_prefix(
        args.run_dir / "student_embeddings/manifest.json",
        rows=args.fusion_prefix_rows,
        out_path=args.run_dir / "student_1m_embeddings_fp16.pt",
    )
    result = {
        "complete": True,
        "config": config,
        "split": {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)},
        "training": {"best_epoch": best_epoch, "best_val_label_mae_eV": best_val, "epochs_completed": len(log)},
        "test_metrics": metric_block(target, prediction),
        "teacher_test_metrics": metric_block(target, teacher),
        "student_teacher_mae_eV": float(np.abs(prediction - teacher).mean()),
        "model": {"path": str(args.model_out), "sha256": sha256_file(args.model_out)},
        "test_predictions": {
            "path": str(prediction_path),
            "sha256": sha256_file(prediction_path),
        },
        "teacher_targets": target_manifest,
        "student_embeddings": {"manifest": str(args.run_dir / "student_embeddings/manifest.json"), "rows": embedding_manifest["rows"]},
        "fusion_prefix": fusion_prefix,
        "log": log,
    }
    atomic_json_write(result, metrics_path)
    atomic_json_write(
        {
            "complete": True,
            "outputs": [
                str(args.model_out),
                str(metrics_path),
                str(checkpoint_path),
                str(prediction_path),
                fusion_prefix["path"],
            ],
            "model_sha256": result["model"]["sha256"],
            "fusion_prefix_sha256": fusion_prefix["sha256"],
        },
        args.run_dir / "completion_manifest.json",
    )
    print(json.dumps({key: result[key] for key in ("training", "test_metrics", "teacher_test_metrics", "student_teacher_mae_eV", "fusion_prefix")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
