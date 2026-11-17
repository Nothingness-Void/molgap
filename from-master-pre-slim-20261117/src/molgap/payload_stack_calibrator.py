"""Train a small, leakage-free calibrator on retained 30k QM9 payloads.

The encoder payloads are produced by independent 30k/3k/3k seed-42
preflights.  This module only learns from their saved train predictions and
embeddings, so the GPU is used for fusion-head training rather than another
graph encoder.  The primary warmblend prediction is an identity path and the
new correction is zero-initialized.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


DEFAULT_SOURCES = (
    "tgt_egt_hybrid_warmblend_etkdg",
    "tgt_egt_hybrid_plus_etkdg",
    "tgt_egt_hybrid_warmblend_frozen_etkdg",
    "tgt_egt_hybrid_etkdg",
    "tgt_egt_rich_etkdg",
    "pair_triplet_2d_rich_topology",
)


class PayloadStackCalibrator(nn.Module):
    """Identity warmblend predictions plus a small learned correction."""

    def __init__(self, feature_dim: int, hidden_channels: int = 128, dropout: float = 0.05):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_channels),
            nn.SiLU(),
            nn.LayerNorm(hidden_channels),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 3),
        )
        last = self.net[-1]
        assert isinstance(last, nn.Linear)
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, features: Tensor, base_predictions: Tensor) -> Tensor:
        return base_predictions + self.net(features)


def _atomic_torch_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _atomic_json_save(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def _load_payloads(cache_dir: Path, sources: tuple[str, ...]) -> dict[str, dict[str, dict[str, Tensor]]]:
    payloads: dict[str, dict[str, dict[str, Tensor]]] = {}
    for source in sources:
        path = cache_dir / "embeddings" / "n30000_3000_3000" / source / "seed42" / "payload.pt"
        if not path.exists():
            raise FileNotFoundError(f"missing retained payload: {path}")
        payloads[source] = torch.load(path, map_location="cpu", weights_only=False)
    return payloads


def _align_split(
    payloads: dict[str, dict[str, dict[str, Tensor]]],
    split: str,
    sources: tuple[str, ...],
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Align all retained experts by molecule source index."""

    primary = payloads[sources[0]][split]
    reference_ids = [int(value) for value in primary["source_idx"].tolist()]
    positions = {
        source: {int(value): index for index, value in enumerate(payloads[source][split]["source_idx"].tolist())}
        for source in sources
    }
    common_ids = [
        value for value in reference_ids if all(value in positions[source] for source in sources)
    ]
    if not common_ids:
        raise RuntimeError(f"no common source_idx rows for split={split}")

    reference_positions = torch.tensor(
        [positions[sources[0]][value] for value in common_ids], dtype=torch.long
    )
    target = primary["targets"].index_select(0, reference_positions).float()
    base_predictions = primary["predictions"].index_select(0, reference_positions).float()
    embedding = primary["embeddings"].index_select(0, reference_positions).float()
    prediction_parts = [
        payloads[source][split]["predictions"]
        .index_select(0, torch.tensor([positions[source][value] for value in common_ids], dtype=torch.long))
        .float()
        for source in sources
    ]
    for source in sources[1:]:
        source_target = payloads[source][split]["targets"].index_select(
            0, torch.tensor([positions[source][value] for value in common_ids], dtype=torch.long)
        ).float()
        if not torch.allclose(target, source_target, atol=1e-5, rtol=1e-5):
            raise ValueError(f"target mismatch in retained payload source={source} split={split}")
    features = torch.cat([embedding, *prediction_parts], dim=1)
    source_index = torch.tensor(common_ids, dtype=torch.long)
    return features, base_predictions, target, source_index, embedding


def _metrics(predictions: Tensor, targets: Tensor) -> dict[str, dict[str, float]]:
    values = torch.abs(predictions - targets).mean(dim=0)
    return {
        "HOMO": {"mae": float(values[0])},
        "LUMO": {"mae": float(values[1])},
        "Gap": {"mae": float(values[2])},
        "average": {"mae": float(values.mean())},
    }


@torch.no_grad()
def _evaluate(
    model: PayloadStackCalibrator,
    loader: DataLoader[tuple[Tensor, Tensor, Tensor]],
    device: torch.device,
) -> tuple[Tensor, dict[str, dict[str, float]]]:
    model.eval()
    predictions: list[Tensor] = []
    targets: list[Tensor] = []
    for features, base_predictions, target in loader:
        features = features.to(device, non_blocking=True)
        base_predictions = base_predictions.to(device, non_blocking=True)
        predictions.append(model(features, base_predictions).cpu())
        targets.append(target)
    joined_predictions = torch.cat(predictions)
    joined_targets = torch.cat(targets)
    return joined_predictions, _metrics(joined_predictions, joined_targets)


def run(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(args.cpu_threads)

    sources = tuple(args.sources)
    payloads = _load_payloads(Path(args.cache_dir), sources)
    aligned = {split: _align_split(payloads, split, sources) for split in ("train", "validation", "test")}
    train_features, train_base, train_targets, train_source_idx, _ = aligned["train"]
    validation_features, validation_base, validation_targets, validation_source_idx, _ = aligned["validation"]
    test_features, test_base, test_targets, test_source_idx, _ = aligned["test"]

    feature_mean = train_features.mean(dim=0)
    feature_std = train_features.std(dim=0).clamp_min(1e-6)
    def standardize(features: Tensor) -> Tensor:
        return (features - feature_mean) / feature_std

    train_features = standardize(train_features)
    validation_features = standardize(validation_features)
    test_features = standardize(test_features)

    train_loader = DataLoader(
        TensorDataset(train_features, train_base, train_targets),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    validation_loader = DataLoader(
        TensorDataset(validation_features, validation_base, validation_targets),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        TensorDataset(test_features, test_base, test_targets),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PayloadStackCalibrator(
        feature_dim=train_features.shape[1],
        hidden_channels=args.hidden_channels,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    result_root = Path(args.results_dir) / f"n{args.train_size}_{args.validation_size}_{args.test_size}" / "payload_stack_calibrator" / f"seed{args.seed}"
    model_root = Path(args.models_dir) / f"n{args.train_size}_{args.validation_size}_{args.test_size}" / "payload_stack_calibrator" / f"seed{args.seed}"
    checkpoint_path = model_root / "checkpoint.pt"
    model_path = model_root / "model.pt"
    payload_path = result_root / "payload.pt"
    metrics_path = result_root / "metrics.json"

    initial_validation_predictions, initial_validation_metrics = _evaluate(model, validation_loader, device)
    best_average = initial_validation_metrics["average"]["mae"]
    best_epoch = -1
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    _atomic_torch_save(
        {
            "epoch": -1,
            "model_state": best_state,
            "optimizer_state": optimizer.state_dict(),
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "sources": sources,
        },
        checkpoint_path,
    )

    for epoch in range(args.epochs):
        model.train()
        for features, base_predictions, targets in train_loader:
            features = features.to(device, non_blocking=True)
            base_predictions = base_predictions.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(features, base_predictions)
            loss = F.smooth_l1_loss(predictions, targets, beta=args.smooth_l1_beta)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

        validation_predictions, validation_metrics = _evaluate(model, validation_loader, device)
        validation_average = validation_metrics["average"]["mae"]
        marker = " *" if validation_average < best_average else ""
        print(
            f"payload_stack_calibrator ep{epoch:02d} train_loss={float(loss):.6f} "
            f"val={validation_average:.5f}eV{marker}",
            flush=True,
        )
        if validation_average < best_average:
            best_average = validation_average
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            _atomic_torch_save(
                {
                    "epoch": epoch,
                    "model_state": best_state,
                    "optimizer_state": optimizer.state_dict(),
                    "feature_mean": feature_mean,
                    "feature_std": feature_std,
                    "sources": sources,
                },
                checkpoint_path,
            )

    model.load_state_dict(best_state)
    model.cpu()
    validation_predictions, validation_metrics = _evaluate(model, validation_loader, torch.device("cpu"))
    test_predictions, test_metrics = _evaluate(model, test_loader, torch.device("cpu"))
    train_predictions, train_metrics = _evaluate(
        model,
        DataLoader(TensorDataset(train_features, train_base, train_targets), batch_size=args.batch_size),
        torch.device("cpu"),
    )
    _atomic_torch_save(
        {
            "candidate": "payload_stack_calibrator",
            "model_state": model.state_dict(),
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "sources": sources,
            "feature_dim": train_features.shape[1],
        },
        model_path,
    )
    _atomic_torch_save(
        {
            "epoch": best_epoch,
            "model_state": best_state,
            "optimizer_state": optimizer.state_dict(),
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "sources": sources,
        },
        checkpoint_path,
    )
    _atomic_torch_save(
        {
            "train": {"predictions": train_predictions, "targets": train_targets, "source_idx": train_source_idx},
            "validation": {"predictions": validation_predictions, "targets": validation_targets, "source_idx": validation_source_idx},
            "test": {"predictions": test_predictions, "targets": test_targets, "source_idx": test_source_idx},
        },
        payload_path,
    )
    report = {
        "experiment": "qm9_payload_stack_calibrator",
        "candidate": "payload_stack_calibrator",
        "geometry": "etkdg-compatible retained payloads",
        "seed": args.seed,
        "split_seed": 42,
        "sources": list(sources),
        "requested_rows": {
            "train": args.train_size,
            "validation": args.validation_size,
            "test": args.test_size,
        },
        "split_rows": {
            "train": int(train_targets.shape[0]),
            "validation": int(validation_targets.shape[0]),
            "test": int(test_targets.shape[0]),
        },
        "model_config": {
            "kind": "payload_stack_calibrator",
            "feature_dim": int(train_features.shape[1]),
            "hidden_channels": args.hidden_channels,
            "dropout": args.dropout,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "device": str(device),
        },
        "n_params": sum(parameter.numel() for parameter in model.parameters()),
        "best_epoch": best_epoch,
        "initial_validation_average_mae_eV": initial_validation_metrics["average"]["mae"],
        "best_validation_average_mae_eV": validation_metrics["average"]["mae"],
        "metrics": {"train": train_metrics, "validation": validation_metrics, "test": test_metrics},
    }
    _atomic_json_save(report, metrics_path)
    print(json.dumps(report, indent=2), flush=True)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--models-dir", required=True)
    parser.add_argument("--train-size", type=int, default=30000)
    parser.add_argument("--validation-size", type=int, default=3000)
    parser.add_argument("--test-size", type=int, default=3000)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-channels", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--smooth-l1-beta", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sources", nargs="+", default=list(DEFAULT_SOURCES))
    return parser


def main(argv: list[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


__all__ = ["PayloadStackCalibrator", "build_parser", "main", "run"]
