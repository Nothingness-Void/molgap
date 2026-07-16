"""Training helpers for frozen-embedding dual-2D controls."""

from __future__ import annotations

import copy

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .dual2d import Dual2DConcatFusion, Dual2DTargetGate


def robust_score(y, prediction, sources):
    """Validation score shared by the active dual-2D heads."""
    error = np.abs(prediction - y)
    sources = np.asarray(sources)
    random_gap = float(error[sources == "random", 2].mean())
    diverse_gap = float(error[sources == "descriptor_diverse", 2].mean())
    hard = np.isin(
        sources, ["narrow_conjugated", "flexible", "large_heteroatom"]
    )
    hard_gap = float(error[hard, 2].mean())
    orbital = float(error[:, :2].mean())
    return {
        "score": 0.45 * random_gap + 0.20 * diverse_gap + 0.25 * hard_gap + 0.10 * orbital,
        "random_gap_mae": random_gap,
        "diverse_gap_mae": diverse_gap,
        "hard_gap_mae": hard_gap,
        "orbital_mae": orbital,
    }


TARGET_WEIGHTS = torch.tensor([0.25, 0.25, 0.50])


def train_dual2d(
    *,
    kind,
    local_embedding,
    global_embedding,
    expert_predictions,
    targets,
    train_indices,
    validation_indices,
    validation_sources,
    seed,
    epochs=80,
    patience=12,
    prior_weights=None,
):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = (
        Dual2DConcatFusion()
        if kind == "concat_fusion"
        else Dual2DTargetGate(prior_weights=prior_weights)
    ).to(device)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(np.asarray(train_indices, dtype=np.int64))),
        batch_size=512,
        shuffle=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    weights = TARGET_WEIGHTS.to(device)
    target_mean = targets[train_indices].mean(axis=0).astype(np.float32)
    target_std = targets[train_indices].std(axis=0).clip(1e-6).astype(np.float32)
    best_state, best_metrics, best_epoch = None, None, -1
    best_score, best_random, wait = np.inf, np.inf, 0
    log = []
    for epoch in range(epochs):
        model.train()
        temperature = 2.0 - epoch / max(epochs - 1, 1)
        total, count = 0.0, 0
        for (selection,) in loader:
            index = selection.numpy()
            local = torch.from_numpy(local_embedding[index]).to(device)
            global_value = torch.from_numpy(global_embedding[index]).to(device)
            target = torch.from_numpy(targets[index]).to(device)
            optimizer.zero_grad(set_to_none=True)
            if kind == "concat_fusion":
                normalized_target = (target - torch.from_numpy(target_mean).to(device)) / torch.from_numpy(target_std).to(device)
                output = model(local, global_value)
                loss_matrix = torch.nn.functional.smooth_l1_loss(
                    output, normalized_target, reduction="none", beta=0.1
                )
            else:
                experts = torch.from_numpy(expert_predictions[index]).to(device)
                output, router_weights = model(
                    local, global_value, experts, temperature=temperature
                )
                loss_matrix = torch.nn.functional.smooth_l1_loss(
                    output, target, reduction="none", beta=0.1
                )
            loss = (loss_matrix * weights).sum(dim=-1).mean()
            if kind != "concat_fusion":
                mean_usage = router_weights.mean(dim=0)
                prior = torch.softmax(model.prior_logits, dim=-1)
                loss = loss + 0.002 * ((mean_usage - prior) ** 2).sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach()) * len(index)
            count += len(index)
        prediction, _ = predict_dual2d(
            model=model,
            kind=kind,
            local_embedding=local_embedding,
            global_embedding=global_embedding,
            expert_predictions=expert_predictions,
            indices=validation_indices,
            target_mean=target_mean,
            target_std=target_std,
        )
        metrics = robust_score(
            targets[validation_indices], prediction, validation_sources
        )
        eligible = metrics["random_gap_mae"] <= best_random + 0.001
        improved = eligible and metrics["score"] < best_score
        if improved:
            best_state = copy.deepcopy(model.state_dict())
            best_metrics, best_epoch = metrics, epoch
            best_score, best_random, wait = metrics["score"], metrics["random_gap_mae"], 0
        else:
            wait += 1
        log.append({
            "epoch": epoch,
            "train_loss": total / max(count, 1),
            "temperature": temperature,
            **metrics,
            "selected": improved,
        })
        if wait >= patience:
            break
    if best_state is None:
        raise RuntimeError(f"{kind} produced no eligible checkpoint")
    model.load_state_dict(best_state)
    return model, {
        "best_epoch": best_epoch,
        "best_validation": best_metrics,
        "target_mean": target_mean.tolist(),
        "target_std": target_std.tolist(),
        "log": log,
    }


@torch.no_grad()
def predict_dual2d(
    *,
    model,
    kind,
    local_embedding,
    global_embedding,
    expert_predictions,
    indices,
    target_mean,
    target_std,
):
    device = next(model.parameters()).device
    model.eval()
    outputs, weights = [], []
    for start in range(0, len(indices), 2048):
        index = indices[start:start + 2048]
        local = torch.from_numpy(local_embedding[index]).to(device)
        global_value = torch.from_numpy(global_embedding[index]).to(device)
        if kind == "concat_fusion":
            prediction = model(local, global_value)
            prediction = (
                prediction * torch.as_tensor(target_std, device=device)
                + torch.as_tensor(target_mean, device=device)
            )
        else:
            experts = torch.from_numpy(expert_predictions[index]).to(device)
            prediction, gate = model(local, global_value, experts, temperature=1.0)
            weights.append(gate.cpu().numpy())
        outputs.append(prediction.cpu().numpy())
    return np.concatenate(outputs), (np.concatenate(weights) if weights else None)
