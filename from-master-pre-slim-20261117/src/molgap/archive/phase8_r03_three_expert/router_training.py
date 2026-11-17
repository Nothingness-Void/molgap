"""Frozen-feature Router training used by the archive-r03 feasibility gate."""

from __future__ import annotations

import copy

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .hetero_moe import FrozenExpertMixer


TARGET_WEIGHTS = np.asarray([0.25, 0.25, 0.50])


def robust_score(y, prediction, sources):
    error = np.abs(prediction - y)
    sources = np.asarray(sources)
    random_gap = float(error[sources == "random", 2].mean())
    diverse_gap = float(error[sources == "descriptor_diverse", 2].mean())
    hard = np.isin(sources, ["narrow_conjugated", "flexible", "large_heteroatom"])
    hard_gap = float(error[hard, 2].mean())
    orbital = float(error[:, :2].mean())
    return {
        "score": 0.45 * random_gap + 0.20 * diverse_gap + 0.25 * hard_gap + 0.10 * orbital,
        "random_gap_mae": random_gap,
        "diverse_gap_mae": diverse_gap,
        "hard_gap_mae": hard_gap,
        "orbital_mae": orbital,
    }


def _forward_batches(model, arrays, indices, device, batch_size=2048):
    predictions, weights, residuals = [], [], []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            selection = indices[start:start + batch_size]
            embeddings = [
                torch.from_numpy(value[selection]).to(device) for value in arrays["embeddings"]
            ]
            expert_predictions = torch.from_numpy(
                arrays["expert_predictions"][selection]
            ).to(device)
            descriptors = torch.from_numpy(arrays["descriptors"][selection]).to(device)
            output = model(
                embeddings, expert_predictions, descriptors,
                temperature=1.0, residual_enabled=True,
            )
            predictions.append(output["prediction"].cpu().numpy())
            weights.append(output["router_weights"].cpu().numpy())
            residuals.append(output["residual"].cpu().numpy())
    return np.concatenate(predictions), np.concatenate(weights), np.concatenate(residuals)


def train_frozen_router(
    *,
    arrays,
    train_indices,
    validation_indices,
    validation_sources,
    n_descriptors,
    shared_targets,
    use_residual,
    seed,
    epochs=12,
):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FrozenExpertMixer(
        n_descriptors=n_descriptors,
        use_residual=use_residual,
        shared_targets=shared_targets,
    ).to(device)
    dataset = TensorDataset(torch.from_numpy(np.asarray(train_indices, dtype=np.int64)))
    loader = DataLoader(dataset, batch_size=512, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    best_state, best_metrics, best_epoch = None, None, -1
    best_score, best_random = np.inf, np.inf
    log = []
    target_weights = torch.tensor(TARGET_WEIGHTS, device=device, dtype=torch.float32)
    for epoch in range(epochs):
        model.train()
        temperature = 2.0 - epoch / max(epochs - 1, 1)
        total, count = 0.0, 0
        for (selection,) in loader:
            selection_np = selection.numpy()
            embeddings = [
                torch.from_numpy(value[selection_np]).to(device)
                for value in arrays["embeddings"]
            ]
            expert_predictions = torch.from_numpy(
                arrays["expert_predictions"][selection_np]
            ).to(device)
            descriptors = torch.from_numpy(arrays["descriptors"][selection_np]).to(device)
            target = torch.from_numpy(arrays["targets"][selection_np]).to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(
                embeddings,
                expert_predictions,
                descriptors,
                temperature=temperature,
                residual_enabled=epoch >= 2,
            )
            main = (
                torch.nn.functional.smooth_l1_loss(
                    output["prediction"], target, reduction="none", beta=0.1
                ) * target_weights
            ).sum(dim=-1).mean()
            mean_usage = output["router_weights"].mean(dim=0)
            balance_weight = 0.02 if epoch < 2 else 0.01 * (1 - epoch / epochs)
            balance = ((mean_usage - 1 / 3) ** 2).sum()
            loss = main + balance_weight * balance
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach()) * len(selection_np)
            count += len(selection_np)
        prediction, _, _ = _forward_batches(
            model, arrays, validation_indices, device
        )
        metrics = robust_score(
            arrays["targets"][validation_indices], prediction, validation_sources
        )
        eligible = metrics["random_gap_mae"] <= best_random + 0.001
        improved = eligible and metrics["score"] < best_score
        if improved:
            best_score = metrics["score"]
            best_random = metrics["random_gap_mae"]
            best_state = copy.deepcopy(model.state_dict())
            best_metrics = metrics
            best_epoch = epoch
        log.append({
            "epoch": epoch,
            "train_loss": total / count,
            "temperature": temperature,
            **metrics,
            "selected": improved,
        })
    if best_state is None:
        raise RuntimeError("Router produced no eligible checkpoint")
    model.load_state_dict(best_state)
    return model, {
        "best_epoch": best_epoch,
        "best_validation": best_metrics,
        "log": log,
    }


def predict_frozen_router(model, arrays, indices):
    device = next(model.parameters()).device
    return _forward_batches(model, arrays, np.asarray(indices), device)
