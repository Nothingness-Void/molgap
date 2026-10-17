"""Small learned gates for frozen GPS7/GPS9/GPS11 prediction experts."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


TARGETS = ("homo", "lumo", "gap")
EXPERTS = ("gps7", "gps9", "gps11_160")
DESCRIPTOR_COLUMNS = (
    "mw",
    "heavy_atoms",
    "aromatic_rings",
    "rotatable_bonds",
    "hetero_atoms",
)


@dataclass(frozen=True)
class GateTrainingConfig:
    hidden_channels: int = 64
    dropout: float = 0.10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 4096
    epochs: int = 80
    patience: int = 10
    expert_call_cost_eV: float = 0.0
    seed: int = 42


def validate_prediction_stack(
    predictions: np.ndarray,
    targets: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    predictions = np.asarray(predictions, dtype=np.float32)
    if predictions.ndim != 3 or predictions.shape[1:] != (len(EXPERTS), len(TARGETS)):
        raise ValueError(
            "predictions must have shape [rows, 3 experts, 3 targets], "
            f"found {predictions.shape}"
        )
    if not np.isfinite(predictions).all():
        raise ValueError("predictions contain non-finite values")
    if targets is None:
        return predictions, None
    targets = np.asarray(targets, dtype=np.float32)
    if targets.shape != (len(predictions), len(TARGETS)):
        raise ValueError(
            f"targets must have shape {(len(predictions), len(TARGETS))}, "
            f"found {targets.shape}"
        )
    if not np.isfinite(targets).all():
        raise ValueError("targets contain non-finite values")
    return predictions, targets


def dense_gate_features(predictions: np.ndarray) -> np.ndarray:
    """Features available after all three experts have run."""
    predictions, _ = validate_prediction_stack(predictions)
    pairwise = np.concatenate(
        (
            predictions[:, 1] - predictions[:, 0],
            predictions[:, 2] - predictions[:, 0],
            predictions[:, 2] - predictions[:, 1],
        ),
        axis=1,
    )
    summary = np.concatenate(
        (predictions.mean(axis=1), predictions.std(axis=1)),
        axis=1,
    )
    physics = predictions[:, :, 2] - (
        predictions[:, :, 1] - predictions[:, :, 0]
    )
    return np.concatenate(
        (predictions.reshape(len(predictions), -1), pairwise, summary, physics),
        axis=1,
    ).astype(np.float32)


def predispatch_features(
    gps7_predictions: np.ndarray,
    descriptors: np.ndarray,
) -> np.ndarray:
    """Features available before deciding whether GPS9/GPS11 are worth calling."""
    gps7_predictions = np.asarray(gps7_predictions, dtype=np.float32)
    descriptors = np.asarray(descriptors, dtype=np.float32)
    if gps7_predictions.ndim != 2 or gps7_predictions.shape[1] != len(TARGETS):
        raise ValueError("gps7_predictions must have shape [rows, 3]")
    if descriptors.ndim != 2 or len(descriptors) != len(gps7_predictions):
        raise ValueError("descriptors must be a row-aligned 2D array")
    physics = gps7_predictions[:, 2:3] - (
        gps7_predictions[:, 1:2] - gps7_predictions[:, 0:1]
    )
    features = np.concatenate((gps7_predictions, physics, descriptors), axis=1)
    if not np.isfinite(features).all():
        raise ValueError("predispatch features contain non-finite values")
    return features.astype(np.float32)


class _NormalizedGate(nn.Module):
    def __init__(
        self,
        input_dim: int,
        feature_mean: np.ndarray,
        feature_std: np.ndarray,
        config: GateTrainingConfig,
    ):
        super().__init__()
        feature_mean = np.asarray(feature_mean, dtype=np.float32)
        feature_std = np.asarray(feature_std, dtype=np.float32)
        if feature_mean.shape != (input_dim,) or feature_std.shape != (input_dim,):
            raise ValueError("Feature normalization shape does not match input_dim")
        self.register_buffer("feature_mean", torch.from_numpy(feature_mean))
        self.register_buffer(
            "feature_std", torch.from_numpy(np.maximum(feature_std, 1e-6))
        )
        self.network = nn.Sequential(
            nn.Linear(input_dim, config.hidden_channels),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_channels, config.hidden_channels),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(
                config.hidden_channels,
                len(TARGETS) * len(EXPERTS),
            ),
        )

    def logits(self, features: torch.Tensor) -> torch.Tensor:
        normalized = (features - self.feature_mean) / self.feature_std
        return self.network(normalized).view(-1, len(TARGETS), len(EXPERTS))


class DenseSoftGate(_NormalizedGate):
    """Target-specific soft blend after all three GPS experts have run."""

    def forward(
        self,
        features: torch.Tensor,
        predictions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weights = torch.softmax(self.logits(features), dim=-1)
        expert_first = predictions.permute(0, 2, 1)
        fused = (weights * expert_first).sum(dim=-1)
        return fused, weights


class PreDispatchRouter(_NormalizedGate):
    """Target-specific expert selector using only GPS7 and cheap descriptors."""

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.logits(features), dim=-1)


def _indices(values: Sequence[int], n_rows: int, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.int64)
    if result.ndim != 1 or len(result) == 0:
        raise ValueError(f"{name} indices must be a non-empty 1D array")
    if result.min() < 0 or result.max() >= n_rows or len(np.unique(result)) != len(result):
        raise ValueError(f"{name} indices are invalid or duplicated")
    return result


def _loader(
    features: np.ndarray,
    predictions: np.ndarray,
    targets: np.ndarray,
    indices: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    tensors = TensorDataset(
        torch.from_numpy(features[indices]),
        torch.from_numpy(predictions[indices]),
        torch.from_numpy(targets[indices]),
    )
    return DataLoader(
        tensors,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
    )


def _state_to_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def fit_dense_soft_gate(
    predictions: np.ndarray,
    targets: np.ndarray,
    train_indices: Sequence[int],
    validation_indices: Sequence[int],
    *,
    config: GateTrainingConfig = GateTrainingConfig(),
    device: torch.device | str = "cpu",
) -> tuple[DenseSoftGate, dict]:
    predictions, targets = validate_prediction_stack(predictions, targets)
    features = dense_gate_features(predictions)
    train_indices = _indices(train_indices, len(features), "train")
    validation_indices = _indices(validation_indices, len(features), "validation")
    if np.intersect1d(train_indices, validation_indices).size:
        raise ValueError("Train and validation indices overlap")
    torch.manual_seed(config.seed)
    device = torch.device(device)
    model = DenseSoftGate(
        features.shape[1],
        features[train_indices].mean(axis=0),
        features[train_indices].std(axis=0),
        config,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    train_loader = _loader(
        features,
        predictions,
        targets,
        train_indices,
        batch_size=config.batch_size,
        shuffle=True,
    )
    validation_loader = _loader(
        features,
        predictions,
        targets,
        validation_indices,
        batch_size=config.batch_size,
        shuffle=False,
    )
    best_loss, best_epoch, wait, best_state = float("inf"), -1, 0, None
    log = []
    for epoch in range(config.epochs):
        model.train()
        train_total, train_rows = 0.0, 0
        for feature, prediction, target in train_loader:
            feature, prediction, target = (
                feature.to(device),
                prediction.to(device),
                target.to(device),
            )
            optimizer.zero_grad()
            fused, _ = model(feature, prediction)
            loss = torch.nn.functional.l1_loss(fused, target)
            loss.backward()
            optimizer.step()
            train_total += float(loss.item()) * len(feature)
            train_rows += len(feature)
        model.eval()
        validation_total, validation_rows = 0.0, 0
        with torch.inference_mode():
            for feature, prediction, target in validation_loader:
                feature, prediction, target = (
                    feature.to(device),
                    prediction.to(device),
                    target.to(device),
                )
                fused, _ = model(feature, prediction)
                loss = torch.nn.functional.l1_loss(fused, target)
                validation_total += float(loss.item()) * len(feature)
                validation_rows += len(feature)
        train_loss = train_total / train_rows
        validation_loss = validation_total / validation_rows
        log.append(
            {
                "epoch": epoch,
                "train_mae_eV": train_loss,
                "validation_mae_eV": validation_loss,
            }
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = _state_to_cpu(model)
            wait = 0
        else:
            wait += 1
            if wait >= config.patience:
                break
    if best_state is None:
        raise RuntimeError("Dense gate produced no finite checkpoint")
    model.load_state_dict(best_state)
    return model.to(device).eval(), {
        "kind": "dense_soft_gate",
        "config": asdict(config),
        "best_epoch": best_epoch,
        "best_validation_mae_eV": best_loss,
        "log": log,
    }


def fit_predispatch_router(
    predictions: np.ndarray,
    targets: np.ndarray,
    descriptors: np.ndarray,
    train_indices: Sequence[int],
    validation_indices: Sequence[int],
    *,
    config: GateTrainingConfig = GateTrainingConfig(),
    device: torch.device | str = "cpu",
) -> tuple[PreDispatchRouter, dict]:
    predictions, targets = validate_prediction_stack(predictions, targets)
    descriptors = np.asarray(descriptors, dtype=np.float32)
    features = predispatch_features(predictions[:, 0], descriptors)
    train_indices = _indices(train_indices, len(features), "train")
    validation_indices = _indices(validation_indices, len(features), "validation")
    if np.intersect1d(train_indices, validation_indices).size:
        raise ValueError("Train and validation indices overlap")
    torch.manual_seed(config.seed)
    device = torch.device(device)
    model = PreDispatchRouter(
        features.shape[1],
        features[train_indices].mean(axis=0),
        features[train_indices].std(axis=0),
        config,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    train_loader = _loader(
        features,
        predictions,
        targets,
        train_indices,
        batch_size=config.batch_size,
        shuffle=True,
    )
    validation_loader = _loader(
        features,
        predictions,
        targets,
        validation_indices,
        batch_size=config.batch_size,
        shuffle=False,
    )

    def loss_block(
        probability: torch.Tensor,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        errors = torch.abs(prediction.permute(0, 2, 1) - target.unsqueeze(-1))
        expected_error = (probability * errors).sum(dim=-1).mean()
        extra_call_probability = probability[:, :, 1:].sum(dim=-1).mean()
        return expected_error + config.expert_call_cost_eV * extra_call_probability

    best_loss, best_epoch, wait, best_state = float("inf"), -1, 0, None
    log = []
    for epoch in range(config.epochs):
        model.train()
        train_total, train_rows = 0.0, 0
        for feature, prediction, target in train_loader:
            feature, prediction, target = (
                feature.to(device),
                prediction.to(device),
                target.to(device),
            )
            optimizer.zero_grad()
            loss = loss_block(model(feature), prediction, target)
            loss.backward()
            optimizer.step()
            train_total += float(loss.item()) * len(feature)
            train_rows += len(feature)
        model.eval()
        validation_total, validation_rows = 0.0, 0
        with torch.inference_mode():
            for feature, prediction, target in validation_loader:
                feature, prediction, target = (
                    feature.to(device),
                    prediction.to(device),
                    target.to(device),
                )
                loss = loss_block(model(feature), prediction, target)
                validation_total += float(loss.item()) * len(feature)
                validation_rows += len(feature)
        train_loss = train_total / train_rows
        validation_loss = validation_total / validation_rows
        log.append(
            {
                "epoch": epoch,
                "train_utility_loss": train_loss,
                "validation_utility_loss": validation_loss,
            }
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = _state_to_cpu(model)
            wait = 0
        else:
            wait += 1
            if wait >= config.patience:
                break
    if best_state is None:
        raise RuntimeError("Predispatch router produced no finite checkpoint")
    model.load_state_dict(best_state)
    return model.to(device).eval(), {
        "kind": "predispatch_router",
        "config": asdict(config),
        "best_epoch": best_epoch,
        "best_validation_utility_loss": best_loss,
        "log": log,
    }


@torch.inference_mode()
def predict_dense_gate(
    model: DenseSoftGate,
    predictions: np.ndarray,
    *,
    batch_size: int = 8192,
) -> tuple[np.ndarray, np.ndarray]:
    predictions, _ = validate_prediction_stack(predictions)
    features = dense_gate_features(predictions)
    device = next(model.parameters()).device
    outputs, weights = [], []
    loader = DataLoader(
        TensorDataset(torch.from_numpy(features), torch.from_numpy(predictions)),
        batch_size=batch_size,
        shuffle=False,
    )
    model.eval()
    for feature, prediction in loader:
        output, weight = model(feature.to(device), prediction.to(device))
        outputs.append(output.cpu())
        weights.append(weight.cpu())
    return torch.cat(outputs).numpy(), torch.cat(weights).numpy()


def load_dense_gate_checkpoint(
    path: Path,
    *,
    device: torch.device | str = "cpu",
) -> DenseSoftGate:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("kind") != "three_gps_dense_soft_gate":
        raise ValueError(f"Unsupported dense gate checkpoint: {checkpoint.get('kind')}")
    if tuple(checkpoint.get("experts", ())) != EXPERTS:
        raise ValueError("Dense gate expert order differs")
    if tuple(checkpoint.get("targets", ())) != TARGETS:
        raise ValueError("Dense gate target order differs")
    state = checkpoint["state_dict"]
    config = GateTrainingConfig(**checkpoint["config"])
    input_dim = int(state["feature_mean"].numel())
    model = DenseSoftGate(
        input_dim,
        np.zeros(input_dim, dtype=np.float32),
        np.ones(input_dim, dtype=np.float32),
        config,
    )
    model.load_state_dict(state, strict=True)
    if not all(torch.isfinite(value).all() for value in model.state_dict().values()):
        raise ValueError("Dense gate checkpoint contains non-finite values")
    return model.to(torch.device(device)).eval()


@torch.inference_mode()
def predict_hard_route(
    model: PreDispatchRouter,
    predictions: np.ndarray,
    descriptors: np.ndarray,
    *,
    batch_size: int = 8192,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    predictions, _ = validate_prediction_stack(predictions)
    features = predispatch_features(predictions[:, 0], descriptors)
    device = next(model.parameters()).device
    selections, probabilities = [], []
    loader = DataLoader(
        TensorDataset(torch.from_numpy(features)),
        batch_size=batch_size,
        shuffle=False,
    )
    model.eval()
    for (feature,) in loader:
        probability = model(feature.to(device))
        probabilities.append(probability.cpu())
        selections.append(probability.argmax(dim=-1).cpu())
    selected = torch.cat(selections).numpy()
    probability = torch.cat(probabilities).numpy()
    expert_first = predictions.transpose(0, 2, 1)
    routed = np.take_along_axis(expert_first, selected[..., None], axis=2)[..., 0]
    return routed, selected, probability


def metric_block(
    targets: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, Mapping[str, float]]:
    targets = np.asarray(targets, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)
    if targets.shape != predictions.shape or targets.shape[1] != len(TARGETS):
        raise ValueError("Metric inputs must have aligned [rows, 3] shapes")
    result: dict[str, Mapping[str, float]] = {}
    for index, target in enumerate(TARGETS):
        result[target] = {
            "mae_eV": float(np.abs(predictions[:, index] - targets[:, index]).mean())
        }
    result["average"] = {
        "mae_eV": float(np.abs(predictions - targets).mean())
    }
    return result


def route_cost(selected: np.ndarray) -> dict[str, object]:
    selected = np.asarray(selected, dtype=np.int64)
    if selected.ndim != 2 or selected.shape[1] != len(TARGETS):
        raise ValueError("selected must have shape [rows, 3 targets]")
    call_gps9 = np.any(selected == 1, axis=1)
    call_gps11 = np.any(selected == 2, axis=1)
    counts = {
        target: {
            expert: float(np.mean(selected[:, target_index] == expert_index))
            for expert_index, expert in enumerate(EXPERTS)
        }
        for target_index, target in enumerate(TARGETS)
    }
    return {
        "expected_encoder_passes": float(
            1.0 + call_gps9.mean() + call_gps11.mean()
        ),
        "gps9_call_fraction": float(call_gps9.mean()),
        "gps11_call_fraction": float(call_gps11.mean()),
        "target_route_fractions": counts,
    }
