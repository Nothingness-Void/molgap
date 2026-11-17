"""Train a leakage-free multi-payload calibrator on the retained QM9 screens.

Unlike the single-embedding calibrator, this module keeps every row supplied by
the complete pure-2D payloads and represents missing ETKDG rows with explicit
availability masks.  All normalization statistics and learned corrections use
the train split only.  The reported common metrics remain restricted to rows
covered by the warmblend ETKDG payload so the preflight gate stays comparable.
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
    "edge_global_2d_topology",
    "pair_triplet_2d_topology",
    "pair_triplet_2d_rich_topology",
    "tgt_egt_compact_etkdg",
    "tgt_egt_hybrid_etkdg",
    "tgt_egt_hybrid_frozen_etkdg",
    "tgt_egt_hybrid_plus_etkdg",
    "tgt_egt_hybrid_warmblend_etkdg",
    "tgt_egt_hybrid_warmblend_frozen_etkdg",
    "tgt_egt_rich_etkdg",
    "tgt_egt_stable_etkdg",
)
PRIMARY_SOURCE = "tgt_egt_hybrid_warmblend_etkdg"
FALLBACK_SOURCE = "pair_triplet_2d_rich_topology"


class PayloadMultiStackCalibrator(nn.Module):
    """Warmblend/fallback identity prediction plus a zero-initialized correction."""

    def __init__(self, feature_dim: int, hidden_channels: int = 256, dropout: float = 0.1):
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


def _load_payloads(cache_dir: Path, sources: tuple[str, ...]) -> dict[str, Any]:
    payloads: dict[str, Any] = {}
    for source in sources:
        path = cache_dir / "embeddings" / "n30000_3000_3000" / source / "seed42" / "payload.pt"
        if not path.exists():
            raise FileNotFoundError(f"missing retained payload: {path}")
        payloads[source] = torch.load(path, map_location="cpu", weights_only=False)
    return payloads


def _align_union(
    payloads: dict[str, Any], split: str, sources: tuple[str, ...]
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Align the union rows, filling absent source views and returning a mask."""

    reference = payloads[FALLBACK_SOURCE][split]
    source_ids = {
        source: [int(value) for value in payloads[source][split]["source_idx"].tolist()]
        for source in sources
    }
    positions = {
        source: {value: index for index, value in enumerate(source_ids[source])}
        for source in sources
    }
    common_ids = source_ids[FALLBACK_SOURCE]
    reference_positions = torch.arange(len(common_ids), dtype=torch.long)
    target = reference["targets"].index_select(0, reference_positions).float()

    embedding_parts: list[Tensor] = []
    prediction_parts: list[Tensor] = []
    availability_parts: list[Tensor] = []
    for source in sources:
        source_part = payloads[source][split]
        embedding_dim = int(source_part["embeddings"].shape[1])
        embeddings = torch.zeros((len(common_ids), embedding_dim), dtype=torch.float32)
        predictions = torch.zeros((len(common_ids), 3), dtype=torch.float32)
        available = torch.zeros((len(common_ids), 1), dtype=torch.float32)
        present_ids = [value for value in common_ids if value in positions[source]]
        if present_ids:
            output_rows = torch.tensor(
                [index for index, value in enumerate(common_ids) if value in positions[source]],
                dtype=torch.long,
            )
            input_rows = torch.tensor(
                [positions[source][value] for value in present_ids], dtype=torch.long
            )
            embeddings.index_copy_(0, output_rows, source_part["embeddings"].index_select(0, input_rows).float())
            predictions.index_copy_(0, output_rows, source_part["predictions"].index_select(0, input_rows).float())
            available.index_fill_(0, output_rows, 1.0)
            source_target = source_part["targets"].index_select(0, input_rows).float()
            if not torch.allclose(target.index_select(0, output_rows), source_target, atol=1e-5, rtol=1e-5):
                raise ValueError(f"target mismatch in retained payload source={source} split={split}")
        embedding_parts.append(embeddings)
        prediction_parts.append(predictions)
        availability_parts.append(available)

    primary_positions = positions[PRIMARY_SOURCE]
    fallback_positions = positions[FALLBACK_SOURCE]
    base_predictions = prediction_parts[sources.index(FALLBACK_SOURCE)].clone()
    primary_predictions = torch.zeros_like(base_predictions)
    primary_rows = [index for index, value in enumerate(common_ids) if value in primary_positions]
    if primary_rows:
        primary_output = torch.tensor(primary_rows, dtype=torch.long)
        primary_input = torch.tensor([primary_positions[common_ids[index]] for index in primary_rows], dtype=torch.long)
        primary_predictions.index_copy_(
            0,
            primary_output,
            payloads[PRIMARY_SOURCE][split]["predictions"].index_select(0, primary_input).float(),
        )
        base_predictions.index_copy_(0, primary_output, primary_predictions.index_select(0, primary_output))

    features = torch.cat([*embedding_parts, *prediction_parts, *availability_parts], dim=1)
    source_index = torch.tensor(common_ids, dtype=torch.long)
    common_mask = torch.zeros((len(common_ids),), dtype=torch.bool)
    common_mask[torch.tensor(primary_rows, dtype=torch.long)] = True
    return features, base_predictions, target, source_index, common_mask


def _metrics(predictions: Tensor, targets: Tensor) -> dict[str, dict[str, float]]:
    values = torch.abs(predictions - targets).mean(dim=0)
    return {
        "HOMO": {"mae": float(values[0])},
        "LUMO": {"mae": float(values[1])},
        "Gap": {"mae": float(values[2])},
        "average": {"mae": float(values.mean())},
    }


@torch.no_grad()
def _evaluate(model: PayloadMultiStackCalibrator, loader: DataLoader, device: torch.device) -> tuple[Tensor, Tensor]:
    model.eval()
    predictions: list[Tensor] = []
    targets: list[Tensor] = []
    for features, base_predictions, target in loader:
        predictions.append(model(features.to(device), base_predictions.to(device)).cpu())
        targets.append(target)
    return torch.cat(predictions), torch.cat(targets)


def run(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(args.cpu_threads)

    sources = tuple(args.sources)
    if PRIMARY_SOURCE not in sources or FALLBACK_SOURCE not in sources:
        raise ValueError("sources must contain the primary warmblend and pure-2D fallback")
    payloads = _load_payloads(Path(args.cache_dir), sources)
    aligned = {split: _align_union(payloads, split, sources) for split in ("train", "validation", "test")}
    train_features, train_base, train_targets, train_source_idx, train_common = aligned["train"]
    validation_features, validation_base, validation_targets, validation_source_idx, validation_common = aligned["validation"]
    test_features, test_base, test_targets, test_source_idx, test_common = aligned["test"]

    feature_mean = train_features.mean(dim=0)
    feature_std = train_features.std(dim=0).clamp_min(1e-6)
    train_features = (train_features - feature_mean) / feature_std
    validation_features = (validation_features - feature_mean) / feature_std
    test_features = (test_features - feature_mean) / feature_std

    def loader(features: Tensor, base: Tensor, targets: Tensor, shuffle: bool) -> DataLoader:
        return DataLoader(
            TensorDataset(features, base, targets),
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )

    train_loader = loader(train_features, train_base, train_targets, True)
    validation_loader = loader(validation_features, validation_base, validation_targets, False)
    test_loader = loader(test_features, test_base, test_targets, False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PayloadMultiStackCalibrator(
        feature_dim=train_features.shape[1], hidden_channels=args.hidden_channels, dropout=args.dropout
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    result_root = Path(args.results_dir) / f"n{args.train_size}_{args.validation_size}_{args.test_size}" / "payload_multistack_calibrator" / f"seed{args.seed}"
    model_root = Path(args.models_dir) / f"n{args.train_size}_{args.validation_size}_{args.test_size}" / "payload_multistack_calibrator" / f"seed{args.seed}"
    checkpoint_path = model_root / "checkpoint.pt"
    model_path = model_root / "model.pt"
    payload_path = result_root / "payload.pt"
    metrics_path = result_root / "metrics.json"

    initial_validation_predictions, initial_validation_targets = _evaluate(model, validation_loader, device)
    best_average = _metrics(initial_validation_predictions[validation_common], initial_validation_targets[validation_common])["average"]["mae"]
    best_epoch = -1
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    _atomic_torch_save(
        {"epoch": -1, "model_state": best_state, "optimizer_state": optimizer.state_dict(), "feature_mean": feature_mean, "feature_std": feature_std, "sources": sources},
        checkpoint_path,
    )

    for epoch in range(args.epochs):
        model.train()
        for features, base_predictions, targets in train_loader:
            features = features.to(device, non_blocking=True)
            base_predictions = base_predictions.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = F.smooth_l1_loss(model(features, base_predictions), targets, beta=args.smooth_l1_beta)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
        validation_predictions, validation_targets_seen = _evaluate(model, validation_loader, device)
        validation_average = _metrics(validation_predictions[validation_common], validation_targets_seen[validation_common])["average"]["mae"]
        marker = " *" if validation_average < best_average else ""
        print(f"payload_multistack_calibrator ep{epoch:02d} train_loss={float(loss):.6f} val_common={validation_average:.5f}eV{marker}", flush=True)
        if validation_average < best_average:
            best_average = validation_average
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            _atomic_torch_save(
                {"epoch": epoch, "model_state": best_state, "optimizer_state": optimizer.state_dict(), "feature_mean": feature_mean, "feature_std": feature_std, "sources": sources},
                checkpoint_path,
            )

    model.load_state_dict(best_state)
    model.cpu()
    train_predictions, train_targets_seen = _evaluate(model, loader(train_features, train_base, train_targets, False), torch.device("cpu"))
    validation_predictions, validation_targets_seen = _evaluate(model, validation_loader, torch.device("cpu"))
    test_predictions, test_targets_seen = _evaluate(model, test_loader, torch.device("cpu"))
    test_common_metrics = _metrics(test_predictions[test_common], test_targets_seen[test_common])
    report = {
        "experiment": "qm9_payload_multistack_calibrator",
        "candidate": "payload_multistack_calibrator",
        "geometry": "union of retained pure-2D and ETKDG-compatible payloads",
        "seed": args.seed,
        "split_seed": 42,
        "sources": list(sources),
        "requested_rows": {"train": args.train_size, "validation": args.validation_size, "test": args.test_size},
        "split_rows": {"train": int(train_targets.shape[0]), "validation": int(validation_targets.shape[0]), "test": int(test_targets.shape[0])},
        "common_etkdg_rows": {"train": int(train_common.sum()), "validation": int(validation_common.sum()), "test": int(test_common.sum())},
        "model_config": {"kind": "payload_multistack_calibrator", "feature_dim": int(train_features.shape[1]), "hidden_channels": args.hidden_channels, "dropout": args.dropout, "learning_rate": args.learning_rate, "weight_decay": args.weight_decay, "batch_size": args.batch_size, "epochs": args.epochs, "device": str(device)},
        "n_params": sum(parameter.numel() for parameter in model.parameters()),
        "best_epoch": best_epoch,
        "initial_validation_common_average_mae_eV": _metrics(initial_validation_predictions[validation_common], initial_validation_targets[validation_common])["average"]["mae"],
        "best_validation_common_average_mae_eV": best_average,
        "metrics": {"train": _metrics(train_predictions, train_targets_seen), "validation": _metrics(validation_predictions, validation_targets_seen), "test": _metrics(test_predictions, test_targets_seen)},
        "common_etkdg_metrics": {"train": _metrics(train_predictions[train_common], train_targets_seen[train_common]), "validation": _metrics(validation_predictions[validation_common], validation_targets_seen[validation_common]), "test": test_common_metrics},
        "gate": {"average_max_eV": 0.0678138843, "Gap_max_eV": 0.082272936, "average_margin_eV": 0.0678138843 - test_common_metrics["average"]["mae"], "Gap_margin_eV": 0.082272936 - test_common_metrics["Gap"]["mae"], "passed": test_common_metrics["average"]["mae"] <= 0.0678138843 and test_common_metrics["Gap"]["mae"] <= 0.082272936},
    }
    _atomic_torch_save({"candidate": "payload_multistack_calibrator", "model_state": model.state_dict(), "feature_mean": feature_mean, "feature_std": feature_std, "sources": sources, "feature_dim": train_features.shape[1]}, model_path)
    _atomic_torch_save({"train": {"predictions": train_predictions, "targets": train_targets_seen, "source_idx": train_source_idx, "common_mask": train_common}, "validation": {"predictions": validation_predictions, "targets": validation_targets_seen, "source_idx": validation_source_idx, "common_mask": validation_common}, "test": {"predictions": test_predictions, "targets": test_targets_seen, "source_idx": test_source_idx, "common_mask": test_common}}, payload_path)
    _atomic_torch_save({"epoch": best_epoch, "model_state": best_state, "optimizer_state": optimizer.state_dict(), "feature_mean": feature_mean, "feature_std": feature_std, "sources": sources}, checkpoint_path)
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
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-channels", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--smooth-l1-beta", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sources", nargs="+", default=list(DEFAULT_SOURCES))
    return parser


def main(argv: list[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


__all__ = ["PayloadMultiStackCalibrator", "build_parser", "main", "run"]
