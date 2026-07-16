"""Reusable expert-training loop for the dual-2D static candidate architecture pilot."""

from __future__ import annotations

import copy
import time
from collections.abc import Callable

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler
from torch_geometric.loader import DataLoader


TARGET_WEIGHTS = (0.25, 0.25, 0.50)


def expert_forward(kind: str, model, batch):
    if kind in {"local", "global"}:
        return model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
    if kind == "geometry":
        return model(
            batch.z,
            batch.pos,
            batch.batch,
            charges=getattr(batch, "charges", None),
        )
    raise ValueError(f"Unknown expert kind: {kind}")


def weighted_smooth_l1(prediction, target):
    loss = torch.nn.functional.smooth_l1_loss(
        prediction, target, reduction="none", beta=0.1
    )
    weights = prediction.new_tensor(TARGET_WEIGHTS)
    return (loss * weights).sum(dim=-1).mean()


@torch.no_grad()
def predict_expert(kind: str, model, loader, device):
    model.eval()
    predictions, targets, source_indices = [], [], []
    for batch in loader:
        batch = batch.to(device)
        predictions.append(expert_forward(kind, model, batch).float().cpu().numpy())
        targets.append(batch.y.float().cpu().numpy())
        source_indices.append(batch.source_idx.view(-1).cpu().numpy())
    return (
        np.concatenate(predictions),
        np.concatenate(targets),
        np.concatenate(source_indices).astype(np.int64),
    )


@torch.no_grad()
def encode_expert(kind: str, model, loader, device):
    model.eval()
    embeddings, predictions, source_indices = [], [], []
    for batch in loader:
        batch = batch.to(device)
        if kind in {"local", "global"}:
            embedding = model.encode(
                batch.x, batch.edge_index, batch.edge_attr, batch.batch
            )
        elif kind == "geometry":
            embedding = model.encode(
                batch.z,
                batch.pos,
                batch.batch,
                charges=getattr(batch, "charges", None),
            )
        else:
            raise ValueError(kind)
        embeddings.append(embedding.float().cpu().numpy())
        predictions.append(model.head(embedding).float().cpu().numpy())
        source_indices.append(batch.source_idx.view(-1).cpu().numpy())
    return (
        np.concatenate(embeddings),
        np.concatenate(predictions),
        np.concatenate(source_indices).astype(np.int64),
    )


def validation_metrics(prediction, target, source_indices, source_lookup):
    errors = np.abs(prediction - target)
    metrics = {
        "overall": {
            "homo_mae": float(errors[:, 0].mean()),
            "lumo_mae": float(errors[:, 1].mean()),
            "gap_mae": float(errors[:, 2].mean()),
        }
    }
    labels = np.asarray([source_lookup[int(index)] for index in source_indices])
    for source in sorted(set(labels)):
        mask = labels == source
        metrics[source] = {
            "n": int(mask.sum()),
            "gap_mae": float(errors[mask, 2].mean()),
        }
    random_gap = metrics["random"]["gap_mae"]
    diverse_gap = metrics["descriptor_diverse"]["gap_mae"]
    hard_mask = np.isin(labels, ["narrow_conjugated", "flexible", "large_heteroatom"])
    hard_gap = float(errors[hard_mask, 2].mean())
    orbital = float(errors[:, :2].mean())
    metrics["robust"] = {
        "random_gap_mae": random_gap,
        "diverse_gap_mae": diverse_gap,
        "hard_gap_mae": hard_gap,
        "orbital_mae": orbital,
        "score": 0.45 * random_gap + 0.20 * diverse_gap + 0.25 * hard_gap + 0.10 * orbital,
    }
    return metrics


def train_expert(
    *,
    kind: str,
    model,
    train_graphs,
    validation_graphs,
    source_lookup: dict[int, str],
    sample_weight: Callable[[str], float],
    device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    patience: int = 8,
):
    generator = torch.Generator().manual_seed(seed)
    weights = [sample_weight(source_lookup[int(graph.source_idx.item())]) for graph in train_graphs]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True, generator=generator)
    train_loader = DataLoader(train_graphs, batch_size=batch_size, sampler=sampler, num_workers=0)
    validation_loader = DataLoader(
        validation_graphs, batch_size=batch_size, shuffle=False, num_workers=0
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1), eta_min=1e-6
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_score, best_random = np.inf, np.inf
    best_state, best_metrics, best_epoch = None, None, -1
    wait, log = 0, []
    for epoch in range(epochs):
        started = time.perf_counter()
        model.train()
        total, count = 0.0, 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                loss = weighted_smooth_l1(expert_forward(kind, model, batch), batch.y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach()) * batch.num_graphs
            count += batch.num_graphs
        scheduler.step()
        prediction, target, indices = predict_expert(
            kind, model, validation_loader, device
        )
        metrics = validation_metrics(prediction, target, indices, source_lookup)
        score = metrics["robust"]["score"]
        random_gap = metrics["robust"]["random_gap_mae"]
        eligible = random_gap <= best_random + 0.001
        improved = eligible and score < best_score
        if improved:
            best_score, best_random = score, random_gap
            best_state = copy.deepcopy(model.state_dict())
            best_metrics, best_epoch, wait = metrics, epoch, 0
        else:
            wait += 1
        row = {
            "epoch": epoch,
            "train_loss": total / max(count, 1),
            "robust_score": score,
            "random_gap_mae": random_gap,
            "lr": optimizer.param_groups[0]["lr"],
            "elapsed_s": time.perf_counter() - started,
            "selected": improved,
        }
        log.append(row)
        print(
            f"{kind} ep{epoch:02d} train={row['train_loss']:.5f} "
            f"score={score:.5f} random_gap={random_gap:.5f} "
            f"{row['elapsed_s']:.1f}s{' *' if improved else ''}",
            flush=True,
        )
        if wait >= patience:
            break
    if best_state is None:
        raise RuntimeError(f"{kind} produced no eligible checkpoint")
    model.load_state_dict(best_state)
    return {
        "state_dict": best_state,
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
        "log": log,
    }
