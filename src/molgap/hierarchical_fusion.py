"""Bounded 3D correction on top of an already-frozen 2D prediction system."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from typing import Sequence
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


TARGETS = ("homo", "lumo", "gap")


@dataclass(frozen=True)
class HierarchicalFusionConfig:
    hidden_channels: int = 128
    correction_scale_eV: float = 0.10
    dropout: float = 0.10
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    batch_size: int = 2048
    epochs: int = 100
    patience: int = 12
    seed: int = 42


@dataclass(frozen=True)
class ConservativeFusionConfig:
    """Safety-first settings for a frozen-2D plus optional-3D correction."""

    hidden_channels: int = 64
    correction_scale_eV: float = 0.03
    gate_init: float = 0.10
    normalization_clip: float = 5.0
    correction_penalty: float = 0.05
    minimum_validation_improvement_eV: float = 0.00025
    dropout: float = 0.05
    learning_rate: float = 2e-4
    weight_decay: float = 1e-5
    batch_size: int = 2048
    epochs: int = 80
    patience: int = 10
    seed: int = 42


class HierarchicalBoundedResidualHead(nn.Module):
    """Keep the frozen 2D output exact and learn only a bounded 3D correction."""

    def __init__(
        self,
        context_dim: int,
        feature_mean: np.ndarray,
        feature_std: np.ndarray,
        config: HierarchicalFusionConfig,
    ):
        super().__init__()
        if config.correction_scale_eV <= 0:
            raise ValueError("correction_scale_eV must be positive")
        feature_mean = np.asarray(feature_mean, dtype=np.float32)
        feature_std = np.asarray(feature_std, dtype=np.float32)
        if feature_mean.shape != (context_dim,) or feature_std.shape != (context_dim,):
            raise ValueError("Normalization shape differs from context_dim")
        self.correction_scale_eV = float(config.correction_scale_eV)
        self.register_buffer("feature_mean", torch.from_numpy(feature_mean))
        self.register_buffer(
            "feature_std", torch.from_numpy(np.maximum(feature_std, 1e-6))
        )
        self.correction = nn.Sequential(
            nn.Linear(context_dim, config.hidden_channels),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_channels, config.hidden_channels),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_channels, len(TARGETS)),
        )

    def forward(
        self,
        base_prediction: torch.Tensor,
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = (context - self.feature_mean) / self.feature_std
        correction = self.correction_scale_eV * torch.tanh(
            self.correction(normalized)
        )
        return base_prediction + correction, correction


class ConservativeHierarchicalResidualHead(nn.Module):
    """An exact-identity, confidence-gated 3D correction for a frozen 2D base."""

    def __init__(
        self,
        context_dim: int,
        feature_center: np.ndarray,
        feature_scale: np.ndarray,
        config: ConservativeFusionConfig,
    ):
        super().__init__()
        if context_dim <= 0:
            raise ValueError("context_dim must be positive")
        if config.correction_scale_eV <= 0:
            raise ValueError("correction_scale_eV must be positive")
        if not 0.0 < config.gate_init < 1.0:
            raise ValueError("gate_init must fall strictly between 0 and 1")
        if config.normalization_clip <= 0:
            raise ValueError("normalization_clip must be positive")
        feature_center = np.asarray(feature_center, dtype=np.float32)
        feature_scale = np.asarray(feature_scale, dtype=np.float32)
        if feature_center.shape != (context_dim,) or feature_scale.shape != (context_dim,):
            raise ValueError("Normalization shape differs from context_dim")
        self.correction_scale_eV = float(config.correction_scale_eV)
        self.normalization_clip = float(config.normalization_clip)
        self.register_buffer("feature_center", torch.from_numpy(feature_center))
        self.register_buffer(
            "feature_scale",
            torch.from_numpy(np.maximum(feature_scale, 1e-6)),
        )
        self.context_norm = nn.LayerNorm(context_dim, elementwise_affine=False)
        self.correction = nn.Sequential(
            nn.Linear(context_dim, config.hidden_channels),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_channels, len(TARGETS)),
        )
        self.gate = nn.Sequential(
            nn.Linear(context_dim, config.hidden_channels),
            nn.SiLU(),
            nn.Linear(config.hidden_channels, len(TARGETS)),
        )
        correction_output = self.correction[-1]
        gate_output = self.gate[-1]
        with torch.no_grad():
            correction_output.weight.zero_()
            correction_output.bias.zero_()
            gate_output.weight.zero_()
            gate_output.bias.fill_(float(torch.logit(torch.tensor(config.gate_init))))

    def forward(
        self,
        base_prediction: torch.Tensor,
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        normalized = (context - self.feature_center) / self.feature_scale
        normalized = normalized.clamp(
            min=-self.normalization_clip,
            max=self.normalization_clip,
        )
        normalized = self.context_norm(normalized)
        confidence = torch.sigmoid(self.gate(normalized))
        correction = (
            self.correction_scale_eV
            * confidence
            * torch.tanh(self.correction(normalized))
        )
        return base_prediction + correction, correction, confidence


def hierarchical_context(
    expert_predictions: np.ndarray,
    dense_weights: np.ndarray,
    primary_embeddings: np.ndarray,
    augmented_embeddings: np.ndarray,
) -> np.ndarray:
    expert_predictions = np.asarray(expert_predictions, dtype=np.float32)
    dense_weights = np.asarray(dense_weights, dtype=np.float32)
    primary_embeddings = np.asarray(primary_embeddings, dtype=np.float32)
    augmented_embeddings = np.asarray(augmented_embeddings, dtype=np.float32)
    rows = len(expert_predictions)
    if expert_predictions.shape != (rows, 3, 3):
        raise ValueError("expert_predictions must have shape [rows, 3, 3]")
    if dense_weights.shape != (rows, 3, 3):
        raise ValueError("dense_weights must have shape [rows, 3, 3]")
    if (
        primary_embeddings.ndim != 2
        or augmented_embeddings.ndim != 2
        or len(primary_embeddings) != rows
        or len(augmented_embeddings) != rows
    ):
        raise ValueError("SchNet embeddings must be row-aligned 2D arrays")
    pairwise = np.concatenate(
        (
            expert_predictions[:, 1] - expert_predictions[:, 0],
            expert_predictions[:, 2] - expert_predictions[:, 0],
            expert_predictions[:, 2] - expert_predictions[:, 1],
        ),
        axis=1,
    )
    context = np.concatenate(
        (
            expert_predictions.reshape(rows, -1),
            dense_weights.reshape(rows, -1),
            pairwise,
            primary_embeddings,
            augmented_embeddings,
        ),
        axis=1,
    )
    if not np.isfinite(context).all():
        raise ValueError("Hierarchical context contains non-finite values")
    return context.astype(np.float32)


def _checked_indices(
    values: Sequence[int],
    rows: int,
    name: str,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.int64)
    if (
        values.ndim != 1
        or len(values) == 0
        or values.min() < 0
        or values.max() >= rows
        or len(np.unique(values)) != len(values)
    ):
        raise ValueError(f"Invalid {name} indices")
    return values


def _state_to_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def _atomic_torch(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _atomic_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def fit_hierarchical_fusion(
    base_prediction: np.ndarray,
    context: np.ndarray,
    targets: np.ndarray,
    train_indices: Sequence[int],
    validation_indices: Sequence[int],
    *,
    config: HierarchicalFusionConfig = HierarchicalFusionConfig(),
    device: torch.device | str = "cpu",
    checkpoint_path: Path | None = None,
    progress_path: Path | None = None,
    resume: bool = False,
    contract_id: str | None = None,
) -> tuple[HierarchicalBoundedResidualHead, dict]:
    base_prediction = np.asarray(base_prediction, dtype=np.float32)
    context = np.asarray(context, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)
    rows = len(base_prediction)
    if base_prediction.shape != (rows, len(TARGETS)):
        raise ValueError("base_prediction must have shape [rows, 3]")
    if targets.shape != base_prediction.shape:
        raise ValueError("targets must align with base_prediction")
    if context.ndim != 2 or len(context) != rows:
        raise ValueError("context must be a row-aligned 2D array")
    if not (
        np.isfinite(base_prediction).all()
        and np.isfinite(context).all()
        and np.isfinite(targets).all()
    ):
        raise ValueError("Fusion inputs contain non-finite values")
    train_indices = _checked_indices(train_indices, rows, "train")
    validation_indices = _checked_indices(validation_indices, rows, "validation")
    if np.intersect1d(train_indices, validation_indices).size:
        raise ValueError("Train and validation indices overlap")

    torch.manual_seed(config.seed)
    device = torch.device(device)
    model = HierarchicalBoundedResidualHead(
        context.shape[1],
        context[train_indices].mean(axis=0),
        context[train_indices].std(axis=0),
        config,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    def loader(indices: np.ndarray, shuffle: bool) -> DataLoader:
        return DataLoader(
            TensorDataset(
                torch.from_numpy(base_prediction[indices]),
                torch.from_numpy(context[indices]),
                torch.from_numpy(targets[indices]),
            ),
            batch_size=config.batch_size,
            shuffle=shuffle,
            num_workers=0,
        )

    train_loader = loader(train_indices, True)
    validation_loader = loader(validation_indices, False)
    best_loss, best_epoch, best_state, wait, start_epoch = (
        float("inf"),
        -1,
        None,
        0,
        0,
    )
    log = []
    if resume:
        if checkpoint_path is None or not checkpoint_path.is_file():
            raise FileNotFoundError("resume=True requires an existing checkpoint")
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
        expected_contract = {
            "contract_id": contract_id,
            "rows": rows,
            "context_dim": context.shape[1],
            "config": asdict(config),
        }
        if checkpoint.get("contract") != expected_contract:
            raise ValueError("Hierarchical Fusion resume contract differs")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        best_loss = float(checkpoint["best_loss"])
        best_epoch = int(checkpoint["best_epoch"])
        best_state = checkpoint["best_state"]
        wait = int(checkpoint["wait"])
        start_epoch = int(checkpoint["next_epoch"])
        log = list(checkpoint["log"])
    for epoch in range(start_epoch, config.epochs):
        model.train()
        train_total, train_rows = 0.0, 0
        for base, features, target in train_loader:
            base, features, target = (
                base.to(device),
                features.to(device),
                target.to(device),
            )
            optimizer.zero_grad()
            prediction, _ = model(base, features)
            loss = nn.functional.l1_loss(prediction, target)
            loss.backward()
            optimizer.step()
            train_total += float(loss.item()) * len(base)
            train_rows += len(base)
        model.eval()
        validation_total, validation_rows = 0.0, 0
        with torch.inference_mode():
            for base, features, target in validation_loader:
                base, features, target = (
                    base.to(device),
                    features.to(device),
                    target.to(device),
                )
                prediction, _ = model(base, features)
                loss = nn.functional.l1_loss(prediction, target)
                validation_total += float(loss.item()) * len(base)
                validation_rows += len(base)
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
        if checkpoint_path is not None:
            contract = {
                "contract_id": contract_id,
                "rows": rows,
                "context_dim": context.shape[1],
                "config": asdict(config),
            }
            _atomic_torch(
                {
                    "contract": contract,
                    "next_epoch": epoch + 1,
                    "model": _state_to_cpu(model),
                    "optimizer": optimizer.state_dict(),
                    "best_loss": best_loss,
                    "best_epoch": best_epoch,
                    "best_state": best_state,
                    "wait": wait,
                    "log": log,
                },
                checkpoint_path,
            )
        if progress_path is not None:
            _atomic_json(
                {
                    "status": "training",
                    "next_epoch": epoch + 1,
                    "best_epoch": best_epoch,
                    "best_validation_mae_eV": best_loss,
                    "wait": wait,
                },
                progress_path,
            )
        if wait >= config.patience:
            break
    if best_state is None:
        raise RuntimeError("No finite hierarchical Fusion checkpoint")
    model.load_state_dict(best_state)
    return model.to(device).eval(), {
        "kind": "hierarchical_2d_3d_bounded_residual",
        "config": asdict(config),
        "best_epoch": best_epoch,
        "best_validation_mae_eV": best_loss,
        "log": log,
    }


@torch.inference_mode()
def predict_hierarchical_fusion(
    model: HierarchicalBoundedResidualHead,
    base_prediction: np.ndarray,
    context: np.ndarray,
    *,
    batch_size: int = 8192,
) -> tuple[np.ndarray, np.ndarray]:
    base_prediction = np.asarray(base_prediction, dtype=np.float32)
    context = np.asarray(context, dtype=np.float32)
    if len(base_prediction) != len(context):
        raise ValueError("Prediction and context rows differ")
    device = next(model.parameters()).device
    predictions, corrections = [], []
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(base_prediction),
            torch.from_numpy(context),
        ),
        batch_size=batch_size,
        shuffle=False,
    )
    model.eval()
    for base, features in loader:
        prediction, correction = model(base.to(device), features.to(device))
        predictions.append(prediction.cpu())
        corrections.append(correction.cpu())
    return torch.cat(predictions).numpy(), torch.cat(corrections).numpy()


def _robust_location_scale(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.median(values, axis=0).astype(np.float32)
    lower, upper = np.quantile(values, (0.25, 0.75), axis=0)
    scale = ((upper - lower) / 1.349).astype(np.float32)
    fallback = np.std(values, axis=0).astype(np.float32)
    scale = np.where(scale > 1e-6, scale, fallback)
    return center, np.maximum(scale, 1e-6)


def fit_conservative_hierarchical_fusion(
    base_prediction: np.ndarray,
    context: np.ndarray,
    targets: np.ndarray,
    train_indices: Sequence[int],
    validation_indices: Sequence[int],
    *,
    config: ConservativeFusionConfig = ConservativeFusionConfig(),
    device: torch.device | str = "cpu",
    checkpoint_path: Path | None = None,
    progress_path: Path | None = None,
    resume: bool = False,
    contract_id: str | None = None,
    progress_label: str | None = None,
) -> tuple[ConservativeHierarchicalResidualHead, dict]:
    """Fit a conservative correction while retaining exact 2D identity as a candidate."""
    base_prediction = np.asarray(base_prediction, dtype=np.float32)
    context = np.asarray(context, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)
    rows = len(base_prediction)
    if base_prediction.shape != (rows, len(TARGETS)):
        raise ValueError("base_prediction must have shape [rows, 3]")
    if targets.shape != base_prediction.shape:
        raise ValueError("targets must align with base_prediction")
    if context.ndim != 2 or len(context) != rows:
        raise ValueError("context must be a row-aligned 2D array")
    if not all(
        np.isfinite(value).all() for value in (base_prediction, context, targets)
    ):
        raise ValueError("Fusion inputs contain non-finite values")
    if config.correction_penalty < 0 or config.minimum_validation_improvement_eV < 0:
        raise ValueError("Fusion penalties and thresholds must be non-negative")
    train_indices = _checked_indices(train_indices, rows, "train")
    validation_indices = _checked_indices(validation_indices, rows, "validation")
    if np.intersect1d(train_indices, validation_indices).size:
        raise ValueError("Train and validation indices overlap")

    torch.manual_seed(config.seed)
    device = torch.device(device)
    center, scale = _robust_location_scale(context[train_indices])
    model = ConservativeHierarchicalResidualHead(
        context.shape[1], center, scale, config
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    def loader(indices: np.ndarray, shuffle: bool) -> DataLoader:
        return DataLoader(
            TensorDataset(
                torch.from_numpy(base_prediction[indices]),
                torch.from_numpy(context[indices]),
                torch.from_numpy(targets[indices]),
            ),
            batch_size=config.batch_size,
            shuffle=shuffle,
            num_workers=0,
        )

    train_loader = loader(train_indices, True)
    validation_loader = loader(validation_indices, False)
    identity_validation_mae = float(
        np.abs(base_prediction[validation_indices] - targets[validation_indices]).mean()
    )
    best_loss = identity_validation_mae
    best_epoch = -1
    best_state = _state_to_cpu(model)
    wait = 0
    start_epoch = 0
    log: list[dict[str, float | int]] = []
    contract = {
        "contract_id": contract_id,
        "rows": rows,
        "context_dim": context.shape[1],
        "config": asdict(config),
        "identity_validation_mae_eV": identity_validation_mae,
    }
    if resume:
        if checkpoint_path is None or not checkpoint_path.is_file():
            raise FileNotFoundError("resume=True requires an existing checkpoint")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if checkpoint.get("contract") != contract:
            raise ValueError("Conservative Fusion resume contract differs")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        best_loss = float(checkpoint["best_loss"])
        best_epoch = int(checkpoint["best_epoch"])
        best_state = checkpoint["best_state"]
        wait = int(checkpoint["wait"])
        start_epoch = int(checkpoint["next_epoch"])
        log = list(checkpoint["log"])

    for epoch in range(start_epoch, config.epochs):
        model.train()
        train_total, train_mae_total, correction_total, train_rows = 0.0, 0.0, 0.0, 0
        for base, features, target in train_loader:
            base, features, target = base.to(device), features.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction, correction, _ = model(base, features)
            mae = nn.functional.l1_loss(prediction, target)
            correction_penalty = correction.abs().mean()
            loss = mae + config.correction_penalty * correction_penalty
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_total += float(loss.item()) * len(base)
            train_mae_total += float(mae.item()) * len(base)
            correction_total += float(correction_penalty.item()) * len(base)
            train_rows += len(base)

        model.eval()
        validation_total, validation_rows = 0.0, 0
        with torch.inference_mode():
            for base, features, target in validation_loader:
                prediction, _, _ = model(
                    base.to(device),
                    features.to(device),
                )
                validation_total += float(
                    nn.functional.l1_loss(prediction, target.to(device)).item()
                ) * len(base)
                validation_rows += len(base)
        validation_loss = validation_total / validation_rows
        improved = (
            validation_loss
            <= best_loss - config.minimum_validation_improvement_eV
        )
        if improved:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = _state_to_cpu(model)
            wait = 0
        else:
            wait += 1
        log.append(
            {
                "epoch": epoch,
                "train_objective": train_total / train_rows,
                "train_mae_eV": train_mae_total / train_rows,
                "train_mean_abs_correction_eV": correction_total / train_rows,
                "validation_mae_eV": validation_loss,
            }
        )
        if progress_label:
            print(
                f"{progress_label} ep{epoch:03d} "
                f"train={train_mae_total / train_rows:.6f} "
                f"val={validation_loss:.6f} "
                f"best={best_loss:.6f}@{best_epoch} wait={wait}",
                flush=True,
            )
        if checkpoint_path is not None:
            _atomic_torch(
                {
                    "contract": contract,
                    "next_epoch": epoch + 1,
                    "model": _state_to_cpu(model),
                    "optimizer": optimizer.state_dict(),
                    "best_loss": best_loss,
                    "best_epoch": best_epoch,
                    "best_state": best_state,
                    "wait": wait,
                    "log": log,
                },
                checkpoint_path,
            )
        if progress_path is not None:
            _atomic_json(
                {
                    "status": "training",
                    "next_epoch": epoch + 1,
                    "best_epoch": best_epoch,
                    "identity_validation_mae_eV": identity_validation_mae,
                    "best_validation_mae_eV": best_loss,
                    "wait": wait,
                },
                progress_path,
            )
        if wait >= config.patience:
            break

    model.load_state_dict(best_state)
    report = {
        "kind": "conservative_hierarchical_2d_3d_residual",
        "config": asdict(config),
        "identity_validation_mae_eV": identity_validation_mae,
        "best_epoch": best_epoch,
        "selected_identity": best_epoch < 0,
        "best_validation_mae_eV": best_loss,
        "log": log,
    }
    if progress_path is not None:
        _atomic_json({"status": "complete", **report}, progress_path)
    return model.to(device).eval(), report


@torch.inference_mode()
def predict_conservative_hierarchical_fusion(
    model: ConservativeHierarchicalResidualHead,
    base_prediction: np.ndarray,
    context: np.ndarray,
    *,
    batch_size: int = 8192,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base_prediction = np.asarray(base_prediction, dtype=np.float32)
    context = np.asarray(context, dtype=np.float32)
    if len(base_prediction) != len(context):
        raise ValueError("Prediction and context rows differ")
    device = next(model.parameters()).device
    predictions, corrections, confidences = [], [], []
    loader = DataLoader(
        TensorDataset(torch.from_numpy(base_prediction), torch.from_numpy(context)),
        batch_size=batch_size,
        shuffle=False,
    )
    model.eval()
    for base, features in loader:
        prediction, correction, confidence = model(
            base.to(device), features.to(device)
        )
        predictions.append(prediction.cpu())
        corrections.append(correction.cpu())
        confidences.append(confidence.cpu())
    return (
        torch.cat(predictions).numpy(),
        torch.cat(corrections).numpy(),
        torch.cat(confidences).numpy(),
    )
